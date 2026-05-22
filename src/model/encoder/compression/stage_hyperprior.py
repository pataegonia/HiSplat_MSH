import os

import torch
import torch.nn as nn
from compressai.models.base import CompressionModel, EntropyBottleneck, GaussianConditional

from .auxt import WLS, iWLS
from .compressor import calc_bits, get_scale_table
from .ops import GDN, quantize_ste


class StageMeanScaleHyperprior(CompressionModel):
    def __init__(self, in_channels: int, N: int, M: int, analysis_depth: int = 4):
        super().__init__()
        if analysis_depth < 2:
            raise ValueError("analysis_depth must be at least 2.")

        self.N = N
        self.M = M
        self.analysis_depth = analysis_depth

        self.g_a_convs = nn.ModuleList()
        self.g_a_gdns = nn.ModuleList()
        self.AuxT_enc = nn.ModuleList()
        self.enc_aux_gates = nn.Parameter(torch.full((analysis_depth,), -4.0))
        current_channels = in_channels
        for i in range(analysis_depth):
            next_channels = M if i == analysis_depth - 1 else N
            self.g_a_convs.append(nn.Conv2d(current_channels, next_channels, 5, 2, 2))
            if i < analysis_depth - 1:
                self.g_a_gdns.append(GDN(next_channels))
            self.AuxT_enc.append(WLS(current_channels, next_channels))
            current_channels = next_channels

        self.g_s_deconvs = nn.ModuleList()
        self.g_s_gdns = nn.ModuleList()
        self.AuxT_dec = nn.ModuleList()
        self.dec_aux_gates = nn.Parameter(torch.full((analysis_depth,), -4.0))
        current_channels = M
        for i in range(analysis_depth):
            next_channels = in_channels if i == analysis_depth - 1 else N
            self.g_s_deconvs.append(nn.ConvTranspose2d(current_channels, next_channels, 5, 2, 2, 1))
            if i < analysis_depth - 1:
                self.g_s_gdns.append(GDN(next_channels, inverse=True))
            self.AuxT_dec.append(iWLS(current_channels, next_channels))
            current_channels = next_channels

        self.h_a = nn.Sequential(
            nn.Conv2d(M, N, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N, N, 5, 2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N, N, 5, 2, padding=2),
        )

        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(N, M, 5, 2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(M, M * 3 // 2, 5, 2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 3 // 2, M * 2, 3, padding=1),
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def update(self, scale_table=None, force: bool = False, update_quantiles: bool = False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        try:
            updated |= self.entropy_bottleneck.update(force=force, update_quantiles=update_quantiles)
        except TypeError as exc:
            if update_quantiles:
                raise TypeError(
                    "Installed CompressAI EntropyBottleneck.update() does not support "
                    "update_quantiles=True. Upgrade CompressAI or disable fast quantile update."
                ) from exc
            updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def _build_stable_indexes(self, scales):
        if os.environ.get("MSH_CODEC_STABLE_INDEXES", "1") != "0":
            # Actual entropy coding recomputes h_s(z_hat) at decode time. CUDA
            # transposed convolutions can differ by ~1e-6 between calls, which
            # is enough to flip a GaussianConditional scale-table boundary and
            # desynchronize the range decoder. Snapping is tiny relative to the
            # log-spaced scale table, but makes index selection reproducible.
            scales = torch.round(scales * 1024.0) / 1024.0
        return self.gaussian_conditional.build_indexes(scales)

    @staticmethod
    def _scalar(tensor, reducer):
        if tensor is None:
            return float("nan")
        tensor = tensor.detach()
        if tensor.numel() == 0:
            return float("nan")
        return float(reducer(tensor).item())

    @staticmethod
    def _format_debug_info(debug_info):
        if not debug_info:
            return "stage=? scene=? step=? input=?"
        scene = debug_info.get("scene_names", debug_info.get("scene", "?"))
        if isinstance(scene, (list, tuple)):
            scene = ",".join(str(item)[:20] for item in scene)
        return (
            f"stage={debug_info.get('stage_id', '?')} "
            f"scene={scene} "
            f"step={debug_info.get('global_step', '?')} "
            f"input={debug_info.get('codec_input', '?')}"
        )

    def _debug_compress_roundtrip(
        self,
        *,
        debug_info,
        z,
        z_hat,
        scales_hat,
        means_hat,
        y,
        y_strings,
        z_strings,
        shape,
        indexes,
    ):
        print(
            f"[MSH_PROBE_ENTER] {self._format_debug_info(debug_info)}",
            flush=True,
        )
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat_expected = torch.round(z - z_offset) + z_offset
        z_rt_max = (z_hat - z_hat_expected).detach().abs().amax()

        scale_table = getattr(self.gaussian_conditional, "scale_table", None)
        scale_table_max = self._scalar(scale_table, torch.amax)
        scale_table_min = self._scalar(scale_table, torch.amin)
        scales_max = self._scalar(scales_hat, torch.amax)
        scales_min = self._scalar(scales_hat, torch.amin)
        scales_finite = bool(torch.isfinite(scales_hat).all().detach().item())

        z_symbols = torch.round(z - z_offset).detach()
        y_symbols = torch.round(y - means_hat).detach()
        z_sym_min = self._scalar(z_symbols, torch.amin)
        z_sym_max = self._scalar(z_symbols, torch.amax)
        y_sym_min = self._scalar(y_symbols, torch.amin)
        y_sym_max = self._scalar(y_symbols, torch.amax)
        idx_min = self._scalar(indexes, torch.amin)
        idx_max = self._scalar(indexes, torch.amax)

        pre_alert = (
            (not scales_finite)
            or z_rt_max.item() > 1.0e-4
            or (scale_table is not None and scale_table.numel() > 0 and scales_max > scale_table_max)
        )
        y_rt_max = torch.full((), float("nan"), device=y.device)
        x_hat_abs_max = torch.full((), float("nan"), device=y.device)
        try:
            y_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=means_hat)
            y_hat_expected = torch.round(y - means_hat) + means_hat
            y_rt_max = (y_hat - y_hat_expected).detach().abs().amax()
            x_hat_abs_max = self._synthesis(y_hat).detach().abs().amax()
        except Exception as exc:
            print(
                "[MSH_CODEC_DIAG] y-decode exception "
                f"{self._format_debug_info(debug_info)} "
                f"exc={type(exc).__name__}: {exc} "
                f"z_rt_max={z_rt_max.item():.6g} "
                f"scales_minmax=({scales_min:.6g},{scales_max:.6g}) "
                f"scale_table_minmax=({scale_table_min:.6g},{scale_table_max:.6g}) "
                f"z_sym_minmax=({z_sym_min:.6g},{z_sym_max:.6g}) "
                f"y_sym_minmax=({y_sym_min:.6g},{y_sym_max:.6g}) "
                f"idx_minmax=({idx_min:.6g},{idx_max:.6g})",
                flush=True,
            )
            raise

        # Emulate the FULL real `decompress()` path inside the diagnostic so we
        # can compare cached (compress-time) tensors against freshly recomputed
        # ones. This pinpoints which call (eb.decompress / h_s / build_indexes /
        # gc.decompress / _synthesis) diverges between inline and real paths.
        z_hat_diff = torch.full((), float("nan"), device=y.device)
        means_diff = torch.full((), float("nan"), device=y.device)
        scales_diff = torch.full((), float("nan"), device=y.device)
        indexes_diff = torch.full((), float("nan"), device=y.device)
        y_hat_real_diff = torch.full((), float("nan"), device=y.device)
        x_hat_real_max = torch.full((), float("nan"), device=y.device)
        try:
            z_hat_2 = self.entropy_bottleneck.decompress(z_strings, shape)
            z_hat_diff = (z_hat - z_hat_2).detach().abs().amax()
            scales_2, means_2 = self.h_s(z_hat_2).chunk(2, 1)
            scales_diff = (scales_hat - scales_2).detach().abs().amax()
            means_diff = (means_hat - means_2).detach().abs().amax()
            indexes_2 = self._build_stable_indexes(scales_2)
            indexes_diff = (indexes.float() - indexes_2.float()).detach().abs().amax()
            y_hat_real = self.gaussian_conditional.decompress(y_strings, indexes_2, means=means_2)
            y_hat_real_diff = (y_hat - y_hat_real).detach().abs().amax()
            x_hat_real_max = self._synthesis(y_hat_real).detach().abs().amax()
        except Exception as exc:
            print(f"[MSH_CODEC_DIAG_REAL_EMU_EXC] {type(exc).__name__}: {exc}", flush=True)

        alert = (
            pre_alert
            or y_rt_max.item() > 1.0e-4
            or x_hat_abs_max.item() > 1.0e4
            or indexes_diff.item() > 0
            or y_hat_real_diff.item() > 1.0e-4
            or x_hat_real_max.item() > 1.0e4
        )
        if os.environ.get("MSH_CODEC_DEBUG", "0") == "1" or alert:
            print(
                f"[MSH_CODEC_DIAG] alert={alert} "
                f"{self._format_debug_info(debug_info)} "
                f"z_rt_max={z_rt_max.item():.6g} "
                f"y_rt_max={y_rt_max.item():.6g} "
                f"x_hat_abs_max={x_hat_abs_max.item():.6g} "
                f"x_hat_real_max={x_hat_real_max.item():.6g} "
                f"z_hat_diff={z_hat_diff.item():.6g} "
                f"scales_diff={scales_diff.item():.6g} "
                f"means_diff={means_diff.item():.6g} "
                f"indexes_diff={indexes_diff.item():.6g} "
                f"y_hat_real_diff={y_hat_real_diff.item():.6g} "
                f"scales_finite={scales_finite} "
                f"scales_minmax=({scales_min:.6g},{scales_max:.6g}) "
                f"scale_table_minmax=({scale_table_min:.6g},{scale_table_max:.6g}) "
                f"z_sym_minmax=({z_sym_min:.6g},{z_sym_max:.6g}) "
                f"y_sym_minmax=({y_sym_min:.6g},{y_sym_max:.6g}) "
                f"idx_minmax=({idx_min:.6g},{idx_max:.6g})",
                flush=True,
            )

    def _analysis(self, x):
        y_aux = x
        y_main = x
        for i, conv in enumerate(self.g_a_convs):
            y_main = conv(y_main)
            if i < len(self.g_a_gdns):
                y_main = self.g_a_gdns[i](y_main)
            y_aux = self.AuxT_enc[i](y_aux)
            y_main = y_main + torch.sigmoid(self.enc_aux_gates[i]) * y_aux
        return y_main

    def _synthesis(self, y_hat):
        y_aux = y_hat
        y_main = y_hat
        for i, deconv in enumerate(self.g_s_deconvs):
            y_main = deconv(y_main)
            if i < len(self.g_s_gdns):
                y_main = self.g_s_gdns[i](y_main)
            y_aux = self.AuxT_dec[i](y_aux)
            y_main = y_main + torch.sigmoid(self.dec_aux_gates[i]) * y_aux
        return y_main

    def forward(self, x, mode: str = "full"):
        if mode == "entropy":
            mode = "full"
        if mode not in {"transform", "quantize", "quantize_hp", "full"}:
            raise ValueError(f"Unknown codec training mode: {mode}")

        y = self._analysis(x)
        if mode == "transform":
            return {
                "x_hat": self._synthesis(y),
                "estimated_bits": None,
            }

        if mode == "quantize":
            y_hat = quantize_ste(y)
            return {
                "x_hat": self._synthesis(y_hat),
                "estimated_bits": None,
            }

        z = self.h_a(y)

        offset = self.entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - offset) + offset

        scales, means = self.h_s(z_hat).chunk(2, 1)
        y_hat = quantize_ste(y - means) + means
        if mode == "quantize_hp":
            return {
                "x_hat": self._synthesis(y_hat),
                "estimated_bits": None,
            }

        _, z_likelihoods = self.entropy_bottleneck(z)
        z_bits = calc_bits(z_likelihoods)
        _, y_likelihoods = self.gaussian_conditional(y, scales, means)
        y_bits = calc_bits(y_likelihoods)

        return {
            "x_hat": self._synthesis(y_hat),
            "estimated_bits": {"y": y_bits, "z": z_bits},
            "y_likelihoods": y_likelihoods,
        }

    def compress(self, x, debug_info=None):
        y = self._analysis(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat, means_hat = self.h_s(z_hat).chunk(2, 1)
        indexes = self._build_stable_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        if os.environ.get("MSH_CODEC_DEBUG", "0") == "1":
            self._debug_compress_roundtrip(
                debug_info=debug_info,
                z=z,
                z_hat=z_hat,
                scales_hat=scales_hat,
                means_hat=means_hat,
                y=y,
                y_strings=y_strings,
                z_strings=z_strings,
                shape=tuple(z.size()[-2:]),
                indexes=indexes,
            )
        return {"strings": [y_strings, z_strings], "shape": tuple(z.size()[-2:])}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat, means_hat = self.h_s(z_hat).chunk(2, 1)
        indexes = self._build_stable_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        return {"x_hat": self._synthesis(y_hat)}

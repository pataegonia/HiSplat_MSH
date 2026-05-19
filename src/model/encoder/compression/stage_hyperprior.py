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

    def update(self, scale_table=None, force: bool = False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

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

    def compress(self, x):
        y = self._analysis(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat, means_hat = self.h_s(z_hat).chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": tuple(z.size()[-2:])}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat, means_hat = self.h_s(z_hat).chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        return {"x_hat": self._synthesis(y_hat)}

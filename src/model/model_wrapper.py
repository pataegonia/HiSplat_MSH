import copy
import json
import os
import time
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import cv2
import moviepy.editor as mpy
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_batch_images, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..utils.my_utils import AverageMeter, format_duration
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int
    test_all_ckpt: bool
    compress: bool = False


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    align_2d: bool | float
    align_3d: bool | float
    align_depth: bool | float
    normal_norm: bool
    lambda_rd: float | None = None
    phase: str = "standard"
    lr_codec: float | None = None
    lr_aux: float | None = None
    lr_generator: float | None = None
    warmup_steps: int = 0
    finetune_steps: int = 0
    cooldown_steps: int = 0
    lr_codec_cooldown: float | None = None
    lr_generator_cooldown: float | None = None
    codec_lr_warmup_steps: int = 500
    codec_lr_warmup_start_factor: float = 0.1
    codec_recon_weight: float = 1.0
    codec_warmup_bypass: bool = True


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class RateLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.attribute_num = 7 + 3 + 1

    def forward(self, estimated_bits, target_size):
        n, k, c, l = target_size
        num_pixels = n * k * (c * l + self.attribute_num)
        return sum(bits / num_pixels for bits in estimated_bits.values())


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None
    train_time: AverageMeter
    bg_time: float

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)
        self.rate_loss = RateLoss()
        self._aux_optimizer = None
        self.train_time = AverageMeter()
        self.bg_time = 0
        # For testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

    def _warmup_codec_bypass(self) -> bool:
        return self.train_cfg.phase == "warmup" and self.train_cfg.codec_warmup_bypass

    def _train_forward_codec_mode(self) -> str:
        return "bypass" if self._warmup_codec_bypass() else "forward"

    def training_step(self, batch, batch_idx):
        max_steps = get_cfg().trainer.max_steps
        batch: BatchedExample = self.data_shim(batch)
        b, tar_v, c, h, w = batch["target"]["image"].shape
        _, con_v, _, con_h, con_w = batch["context"]["image"].shape
        # Run the model and get gaussians
        gaussian_dict, result_dict = self.encoder(
            batch["context"],
            self.global_step,
            False,
            scene_names=batch["scene"],
            codec_mode=self._train_forward_codec_mode(),
        )
        target_gt = batch["target"]["image"]
        # For three resolutions, render them
        total_loss = 0
        total_bpp = None
        total_estimated_bits = None
        total_estimated_kb = None
        total_codec_recon = None
        loss_dict = {}
        for i in range(len(gaussian_dict)):
            gaussians = gaussian_dict[f"stage{i}"]["gaussians"]
            pre_output = None if i == 0 else output
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=self.train_cfg.depth_mode,
            )
            # Compute metrics.
            psnr_probabilistic = compute_psnr(
                rearrange(target_gt, "b v c h w -> (b v) c h w"),
                rearrange(output.color, "b v c h w -> (b v) c h w"),
            )
            self.log(f"train/psnr_probabilistic_stage{i}", psnr_probabilistic.mean())
            sup_batch = copy.deepcopy(batch)
            # Compute and log loss.
            for loss_fn in self.losses:
                loss = loss_fn.forward(output, sup_batch, gaussians, self.global_step)
                self.log(f"loss/{loss_fn.name}_{i}", loss)
                loss_dict[f"{loss_fn.name}_{i}"] = loss.item()
                if not self._warmup_codec_bypass():
                    total_loss = total_loss + loss
            codec_recon = result_dict[f"stage{i}"].get("codec_recon_loss")
            if codec_recon is not None:
                self.log(f"loss/codec_recon_stage{i}", codec_recon)
                loss_dict[f"codec_recon_{i}"] = codec_recon.item()
                total_codec_recon = codec_recon if total_codec_recon is None else total_codec_recon + codec_recon
            estimated_bits = result_dict[f"stage{i}"].get("estimated_bits")
            if estimated_bits is not None:
                estimated_num_bits = self._estimated_bits_num_bits(estimated_bits)
                total_estimated_bits = (
                    estimated_num_bits
                    if total_estimated_bits is None
                    else total_estimated_bits + estimated_num_bits
                )
                estimated_kb_i = self._bits_to_kbytes(estimated_num_bits)
                self.log(f"loss/estimated_kb_stage{i}", estimated_kb_i)
                loss_dict[f"estimated_kb_{i}"] = estimated_kb_i.detach().item()
                total_estimated_kb = (
                    estimated_kb_i if total_estimated_kb is None else total_estimated_kb + estimated_kb_i
                )

                bpp_i = self.rate_loss(estimated_bits, gaussians.harmonics.shape)
                self.log(f"loss/bpp_attr_stage{i}", bpp_i)
                self.log(f"loss/bpp_stage{i}", bpp_i)
                loss_dict[f"bpp_{i}"] = bpp_i.item()
                total_bpp = bpp_i if total_bpp is None else total_bpp + bpp_i
        if total_estimated_kb is not None:
            self.log("loss/estimated_kb", total_estimated_kb)
        if total_bpp is not None:
            # legacy per-Gaussian-attribute proxy, diagnostic only (NOT in Lagrangian)
            self.log("loss/bpp_attr_total", total_bpp)
            self.log("loss/bpp_total", total_bpp)
        if total_estimated_bits is not None and self.train_cfg.lambda_rd is not None:
            # standard LIC rate: total estimated bits over a single consistent
            # denominator (context input pixels), so the optimizer is free to
            # allocate bits across stages.
            rate_bpp = total_estimated_bits / (b * con_v * con_h * con_w)
            self.log("loss/rate_bpp_total", rate_bpp)
            total_loss = total_loss + self.train_cfg.lambda_rd * rate_bpp
        if total_codec_recon is not None and self.train_cfg.codec_recon_weight > 0:
            self.log("loss/codec_recon_total", total_codec_recon)
            total_loss = total_loss + self.train_cfg.codec_recon_weight * total_codec_recon
        if self._warmup_codec_bypass() and total_codec_recon is None:
            raise ValueError("Warmup codec bypass requires codec reconstruction losses.")
        if self._warmup_codec_bypass() and self.train_cfg.codec_recon_weight <= 0:
            raise ValueError("Warmup codec bypass requires train.codec_recon_weight > 0.")
        if not torch.isfinite(total_loss.detach()):
            raise FloatingPointError(
                f"Non-finite training loss at step {self.global_step}. "
                f"Logged components: {loss_dict}"
            )
        self.log("loss/total", total_loss)
        if self.global_rank == 0 and self.global_step % self.train_cfg.print_log_every_n_steps == 0:
            print(
                f"train step[{self.global_step}/{get_cfg().trainer.max_steps}] ; "
                f"used: {format_duration(self.train_time.sum)}; "
                f"eta: {format_duration((get_cfg().trainer.max_steps - self.global_step) * self.train_time.avg)}; "
                f"loss = {total_loss:.6f}; "
                f"{[n + f'={l:.6f}; ' for n, l in loss_dict.items()]}"
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    """ Log the time"""

    def on_train_batch_start(self, batch, batch_idex):
        if self.train_time.avg == 0:
            self.train_time.update(0.0001)
        else:
            self.train_time.update(time.time() - self.bg_time)
        self.bg_time = time.time()

    def on_test_start(self):
        if self.test_cfg.compress and hasattr(self.encoder, "update_codecs"):
            self.encoder.update_codecs(force=True)

    @staticmethod
    def _estimated_bits_num_bits(estimated_bits):
        total_bits = None
        for bits in estimated_bits.values():
            total_bits = bits if total_bits is None else total_bits + bits
        if total_bits is None:
            raise ValueError("estimated_bits must contain at least one stream.")
        return total_bits

    @staticmethod
    def _bits_to_kbytes(num_bits):
        return num_bits / 8 / 1024

    @staticmethod
    def _bytes_to_kbytes(num_bytes):
        return num_bytes / 1024

    @classmethod
    def _estimated_kbytes_by_stage(cls, result_dict) -> dict[str, float]:
        kbytes_by_stage = {}
        for stage, stage_result in result_dict.items():
            estimated_bits = stage_result.get("estimated_bits")
            if estimated_bits is None:
                continue
            estimated_num_bits = cls._estimated_bits_num_bits(estimated_bits)
            kbytes_by_stage[stage] = cls._bits_to_kbytes(estimated_num_bits).detach().item()
        return kbytes_by_stage

    @staticmethod
    def _codec_payload_num_bytes(codec_payloads) -> int:
        stage_payloads = codec_payloads.get("stages", codec_payloads)
        total_bytes = 0
        for payload in stage_payloads.values():
            if payload is None:
                continue
            for stream_group in payload["strings"]:
                for stream in stream_group:
                    total_bytes += len(stream)
        return total_bytes

    @staticmethod
    def _codec_payload_num_bits(codec_payloads) -> int:
        return ModelWrapper._codec_payload_num_bytes(codec_payloads) * 8

    @staticmethod
    def _codec_payload_bytes_by_stage(codec_payloads) -> dict[str, int]:
        stage_payloads = codec_payloads.get("stages", codec_payloads)
        bytes_by_stage = {}
        for stage, payload in stage_payloads.items():
            total_bytes = 0
            if payload is not None:
                for stream_group in payload["strings"]:
                    for stream in stream_group:
                        total_bytes += len(stream)
            bytes_by_stage[stage] = total_bytes
        return bytes_by_stage

    @staticmethod
    def _codec_payload_bits_by_stage(codec_payloads) -> dict[str, int]:
        return {
            stage: num_bytes * 8
            for stage, num_bytes in ModelWrapper._codec_payload_bytes_by_stage(codec_payloads).items()
        }

    def test_step(self, batch, batch_idx):
        # extrinsic: [b, mv, 4, 4]
        # intrinsic: [b, mv, 3, 3]
        # image: [b, mv, c, h, w]
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1

        codec_payloads = None
        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            if self.test_cfg.compress:
                codec_outputs = self.encoder.compress(
                    batch["context"],
                    self.global_step,
                    False,
                    scene_names=batch["scene"],
                )
                codec_payloads = codec_outputs
                gaussian_dict = codec_outputs["gaussian_dict"]
                result_dict = codec_outputs["result_dict"]
            else:
                gaussian_dict, result_dict = self.encoder(
                    batch["context"], self.global_step, False, scene_names=batch["scene"]
                )
        with self.benchmarker.time("decoder", num_calls=v):
            gaussians = gaussian_dict[f"stage2"]["gaussians"]
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=self.train_cfg.depth_mode,
            )
        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        images_prob = output.color[0]
        rgb_gt = batch["target"]["image"][0]

        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in images_prob],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v
            rgb = images_prob

            if f"psnr" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr"] = []
            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = []
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []
            psnr, ssim, lpips = compute_psnr(rgb_gt, rgb), compute_ssim(rgb_gt, rgb), compute_lpips(rgb_gt, rgb)
            self.test_step_outputs[f"psnr"].append(psnr.mean().item())
            self.test_step_outputs[f"ssim"].append(ssim.mean().item())
            self.test_step_outputs[f"lpips"].append(lpips.mean().item())

            estimated_kbytes_by_stage = self._estimated_kbytes_by_stage(result_dict)
            if estimated_kbytes_by_stage:
                if "estimated_kb" not in self.test_step_outputs:
                    self.test_step_outputs["estimated_kb"] = []
                self.test_step_outputs["estimated_kb"].append(sum(estimated_kbytes_by_stage.values()))
                for stage, stage_kbytes in estimated_kbytes_by_stage.items():
                    kb_key = f"estimated_kb_{stage}"
                    if kb_key not in self.test_step_outputs:
                        self.test_step_outputs[kb_key] = []
                    self.test_step_outputs[kb_key].append(stage_kbytes)

            if codec_payloads is not None:
                total_bytes = self._codec_payload_num_bytes(codec_payloads)
                total_bits = self._codec_payload_num_bits(codec_payloads)
                _, context_views, _, context_h, context_w = batch["context"]["image"].shape
                denom = b * context_views * context_h * context_w
                if "compressed_kb" not in self.test_step_outputs:
                    self.test_step_outputs["compressed_kb"] = []
                if "compressed_bits" not in self.test_step_outputs:
                    self.test_step_outputs["compressed_bits"] = []
                if "compressed_mbytes" not in self.test_step_outputs:
                    self.test_step_outputs["compressed_mbytes"] = []
                if "compressed_bpp" not in self.test_step_outputs:
                    self.test_step_outputs["compressed_bpp"] = []
                self.test_step_outputs["compressed_kb"].append(self._bytes_to_kbytes(total_bytes))
                self.test_step_outputs["compressed_bits"].append(total_bits)
                self.test_step_outputs["compressed_mbytes"].append(total_bits / 8 / (1024 * 1024))
                self.test_step_outputs["compressed_bpp"].append(total_bits / denom)
                for stage, stage_bytes in self._codec_payload_bytes_by_stage(codec_payloads).items():
                    stage_bits = stage_bytes * 8
                    kb_key = f"compressed_kb_{stage}"
                    bits_key = f"compressed_bits_{stage}"
                    bpp_key = f"compressed_bpp_{stage}"
                    if kb_key not in self.test_step_outputs:
                        self.test_step_outputs[kb_key] = []
                    if bits_key not in self.test_step_outputs:
                        self.test_step_outputs[bits_key] = []
                    if bpp_key not in self.test_step_outputs:
                        self.test_step_outputs[bpp_key] = []
                    self.test_step_outputs[kb_key].append(self._bytes_to_kbytes(stage_bytes))
                    self.test_step_outputs[bits_key].append(stage_bits)
                    self.test_step_outputs[bpp_key].append(stage_bits / denom)
            # Create the parent directory if it doesn't already exist.
            log_path = path / scene / "psnr.txt"
            psnr_log = [f"example{j}: {psnr[j].item():.2f} \n" for j in range(len(psnr))]
            psnr_log = reduce(lambda a, b: a + b, psnr_log)
            os.makedirs(str(path / scene), exist_ok=True)
            with open(log_path, "w") as f:
                f.write(psnr_log)

        # Save images.
        if self.test_cfg.save_image:
            for index, color in zip(batch["target"]["index"][0], images_prob):
                save_image(color, path / scene / f"color/{index:0>6}.png")
            comparison = hcat(
                add_label(vcat(*batch["context"]["image"][0]), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*images_prob), "Target (Softmax)"),
            )
            save_batch_images(rgb_gt, str(path / scene / "output.png"))
            save_image(add_border(comparison), path / scene / "compare.png")

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
            self.benchmarker.dump_memory(self.test_cfg.output_path / name / "peak_memory.json")
            self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        self.eval_cnt += 1
        if self.global_rank == 0:
            print(
                f"validation step on {self.global_step} {self.eval_cnt}/{len(self.trainer.val_dataloaders)}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        # Run the model and get gaussians
        gaussian_dict, result_dict = self.encoder(
            batch["context"],
            self.global_step,
            False,
            scene_names=batch["scene"],
            codec_mode=self._train_forward_codec_mode(),
        )
        output_list = []
        # for debug
        render_img_debug_list = []
        depth_fine_debug_list = []
        depth_coarse_debug_list = []
        for i in range(len(gaussian_dict)):
            gaussians = gaussian_dict[f"stage{i}"]["gaussians"]
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=self.train_cfg.depth_mode,
            )
            rgb_softmax = output.color[0]
            # for debug
            v = result_dict[f"stage{i}"]["depths"].size(1)
            fine_depth_i = F.interpolate(
                result_dict[f"stage{i}"]["depths"].reshape(b * v, 64 * 2**i, 64 * 2**i)[:, None],
                size=(256, 256),
                mode="bilinear",
            )[0, 0]
            fine_depth_i = cv2.applyColorMap(
                ((fine_depth_i.clip(1, 10) / 10).detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            fine_depth_i = torch.from_numpy(fine_depth_i).permute(2, 0, 1)
            depth_fine_debug_list.append(fine_depth_i.flip(0) / 255)

            coarse_depth_i = F.interpolate(
                1 / result_dict[f"stage{i}"]["coarse_disps"], size=(256, 256), mode="bilinear"
            )[0, 0]
            coarse_depth_i = cv2.applyColorMap(
                ((coarse_depth_i.clip(1, 10) / 10).detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            coarse_depth_i = torch.from_numpy(coarse_depth_i).permute(2, 0, 1)
            depth_coarse_debug_list.append(coarse_depth_i.flip(0) / 255)
            render_img_debug_list.append(output.color[0, 0])
            # Compute validation metrics.
            rgb_gt = batch["target"]["image"][0]
            for tag, rgb in zip(("val",), (rgb_softmax,)):
                psnr = compute_psnr(rgb_gt, rgb).mean()
                lpips = compute_lpips(rgb_gt, rgb).mean()
                ssim = compute_ssim(rgb_gt, rgb).mean()
                if i == len(gaussian_dict) - 1:
                    self.log(f"val/psnr_{tag}", psnr)
                    self.log(f"val/lpips_{tag}", lpips)
                    self.log(f"val/ssim_{tag}", ssim)
                self.log(f"val/psnr_{tag}_{i}", psnr)
                self.log(f"val/lpips_{tag}_{i}", lpips)
                self.log(f"val/ssim_{tag}_{i}", ssim)

            if self.eval_cnt == len(self.trainer.val_dataloaders) or self.eval_cnt == 0:
                # Construct comparison image.
                comparison = hcat(
                    add_label(vcat(*batch["context"]["image"][0]), "Context"),
                    add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                    add_label(vcat(*rgb_softmax), "Target (Softmax)"),
                )
                self.logger.log_image(
                    f"comparison_{i}",
                    [prep_image(add_border(comparison))],
                    step=self.global_step,
                    caption=batch["scene"],
                )

                # Render projections and construct projection image.
                projections = hcat(
                    *render_projections(
                        gaussians,
                        256,
                        extra_label="(Softmax)",
                    )[0]
                )
                self.logger.log_image(
                    f"projection_{i}",
                    [prep_image(add_border(projections))],
                    step=self.global_step,
                )

                # Draw cameras.
                cameras = hcat(*render_cameras(batch, 256))
                self.logger.log_image(f"cameras_{i}", [prep_image(add_border(cameras))], step=self.global_step)
        if self.eval_cnt == len(self.trainer.val_dataloaders) or self.eval_cnt == 0:
            fine_depth = add_label(vcat(*depth_fine_debug_list), label="fine_d")
            coarse_depth = add_label(vcat(*depth_coarse_debug_list), label="coarse_d")
            render_img = add_label(vcat(*render_img_debug_list), label="render")
            gt = vcat(
                add_label(batch["context"]["image"][0, 0], label="context"),
                add_label(batch["target"]["image"][0, 0], label="target"),
            )
            self.logger.log_image(
                "depth_compare", [prep_image(hcat(gt, render_img, fine_depth, coarse_depth))], step=self.global_step
            )
        if self.eval_cnt == len(self.trainer.val_dataloaders):
            self.eval_cnt = 0

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (batch["context"]["extrinsics"][0, 1] if v == 2 else batch["target"]["extrinsics"][0, 0]),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (batch["context"]["intrinsics"][0, 1] if v == 2 else batch["target"]["intrinsics"][0, 0]),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (batch["context"]["extrinsics"][0, 1] if v == 2 else batch["target"]["extrinsics"][0, 0]),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (batch["context"]["intrinsics"][0, 1] if v == 2 else batch["target"]["intrinsics"][0, 0]),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob, depths, scales, rotations = self.encoder(batch["context"], self.global_step, False)
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth")
        images_prob = [vcat(rgb, depth) for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))]
        # output_det = self.decoder.forward(
        #     gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
        # )
        # images_det = [
        #     vcat(rgb, depth)
        #     for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
        # ]
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Softmax"),
                    # add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")}

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(str(dir / f"{self.global_step:0>6}.mp4"), logger=None)

    def _codec_train_mode(self):
        mode = getattr(self.encoder, "codec_train_mode", "full")
        return "full" if mode == "entropy" else mode

    def _codec_uses_entropy(self):
        return self._codec_train_mode() == "full"

    def _codec_aux_loss(self):
        loss = torch.zeros((), device=self.device)
        for module in self.encoder.modules():
            codec = getattr(module, "gauss_feature_codec", None)
            if codec is not None and hasattr(codec, "aux_loss"):
                loss = loss + codec.aux_loss()
        return loss

    def on_before_optimizer_step(self, optimizer):
        if self._aux_optimizer is None or not self._codec_uses_entropy():
            return
        aux_loss = self._codec_aux_loss()
        self._aux_optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(aux_loss.detach()):
            self.log("loss/aux", aux_loss.detach(), prog_bar=False)
            return
        aux_loss.backward()
        aux_params = [
            param
            for group in self._aux_optimizer.param_groups
            for param in group["params"]
        ]
        torch.nn.utils.clip_grad_norm_(aux_params, 1.0)
        self._aux_optimizer.step()
        self.log("loss/aux", aux_loss.detach(), prog_bar=False)

    def _split_codec_params(self):
        codec_params, aux_params, generator_params = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".quantiles"):
                aux_params.append(param)
            elif "gauss_feature_codec" in name:
                codec_params.append(param)
            else:
                generator_params.append(param)
        return codec_params, aux_params, generator_params

    def _codec_optimizer_config(self, optimizer):
        warmup_steps = self.train_cfg.codec_lr_warmup_steps
        if warmup_steps <= 0:
            return optimizer
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.train_cfg.codec_lr_warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @staticmethod
    def _first_non_none(*values):
        for value in values:
            if value is not None:
                return value
        return None

    def configure_optimizers(self):
        codec_params, aux_params, generator_params = self._split_codec_params()
        phase = self.train_cfg.phase
        self._aux_optimizer = None

        if codec_params and phase == "standard":
            raise ValueError(
                "train.phase='standard' is unsafe when MSH codecs are enabled. "
                "Use warmup, finetune, or cooldown so codec/generator learning rates are explicit."
            )
        if phase != "standard" and not codec_params:
            raise ValueError(f"train.phase='{phase}' requires model.encoder.use_gauss_feature_compression=true.")

        aux_lr = self.train_cfg.lr_aux
        if aux_lr is None:
            aux_lr = self.train_cfg.lr_codec
        if aux_lr is None:
            aux_lr = self.optimizer_cfg.lr
        if aux_params and self._codec_uses_entropy() and aux_lr > 0:
            self._aux_optimizer = optim.Adam(
                aux_params,
                lr=aux_lr,
            )

        if phase == "warmup" and codec_params:
            optimizer = optim.Adam(
                codec_params,
                lr=self._first_non_none(self.train_cfg.lr_codec, self.optimizer_cfg.lr),
            )
            return self._codec_optimizer_config(optimizer)

        if phase in {"finetune", "cooldown"} and codec_params:
            if phase == "cooldown":
                lr_codec = self._first_non_none(
                    self.train_cfg.lr_codec_cooldown,
                    self.train_cfg.lr_codec,
                    self.optimizer_cfg.lr,
                )
                lr_generator = self._first_non_none(
                    self.train_cfg.lr_generator_cooldown,
                    self.train_cfg.lr_generator,
                    self.optimizer_cfg.lr,
                )
            else:
                lr_codec = self._first_non_none(self.train_cfg.lr_codec, self.optimizer_cfg.lr)
                lr_generator = self._first_non_none(self.train_cfg.lr_generator, self.optimizer_cfg.lr)
            param_groups = []
            if generator_params:
                param_groups.append({"params": generator_params, "lr": lr_generator})
            param_groups.append({"params": codec_params, "lr": lr_codec})
            return self._codec_optimizer_config(optim.Adam(param_groups))

        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                self.optimizer_cfg.lr,
                self.trainer.max_steps + 10,
                pct_start=0.01,
                cycle_momentum=False,
                anneal_strategy="cos",
            )
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }

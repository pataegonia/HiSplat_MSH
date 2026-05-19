# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import pickle

import torch
import torch.nn as nn
from torch import Tensor
from compressai.models.base import (
    CompressionModel,
    EntropyBottleneck,
    GaussianConditional,
)
from compressai.models.utils import conv

from .ops import GDN, quantize_ste, conv1x1, ste_round
from .auxt import WLS, iWLS, OLP
from typing import cast

def calc_bits(likelihood):
    likelihood = torch.nan_to_num(likelihood, nan=1e-9, posinf=1.0, neginf=1e-9)
    likelihood = likelihood.clamp(min=1e-9, max=1.0)
    return -torch.log2(likelihood).sum()



class MeanScaleHyperpriorBase(CompressionModel):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        GC        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │1
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │                 means_hat
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        GC = Gaussian conditional entropy model

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, in_channels: int, N: int, M: int):
        super().__init__()
        self.N, self.M = N, M

        self.g_a = nn.Sequential(
            nn.Conv2d(in_channels, N, 5, 2, 2),
            GDN(N),
            nn.Conv2d(N, N, 5, 2, 2),
            GDN(N),
            nn.Conv2d(N, N, 5, 2, 2),
            GDN(N),
            nn.Conv2d(N, M, 5, 2, 2),
        )

        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(M, N, 5, 2, 2, 1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, 2, 2, 1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, 2, 2, 1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, in_channels, 5, 2, 2, 1),
        )

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
        
        self.AuxT_enc = nn.Sequential(
            WLS(in_channels,N),
            WLS(N,N),
            WLS(N,N),
            WLS(N,M),  
        )  
        self.AuxT_dec = nn.Sequential(
            iWLS(M,N),
            iWLS(N,N),
            iWLS(N,N),
            iWLS(N,in_channels),  
        )
        
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def ortho_loss(self) -> Tensor:
        loss = sum(m.loss() for m in self.modules() if isinstance(m, OLP))
        return cast(Tensor, loss)
    
    def forward(self, x):
        y_enc_aux = x
        y_enc_main = x
        aux_index = 0
        for i,layer in enumerate(self.g_a): 
            y_enc_main = layer(y_enc_main) 
            if i in [0,2,4,6]: #shorcut position for encoder
                y_enc_aux = self.AuxT_enc[aux_index](y_enc_aux)
                y_enc_main += y_enc_aux
                aux_index += 1
        y = y_enc_main
        
        z = self.h_a(y)

        offset = self.entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - offset) + offset

        _, z_likelihoods = self.entropy_bottleneck(z)
        z_bits = calc_bits(z_likelihoods)

        scales, means = self.h_s(z_hat).chunk(2, 1)

        y_hat = quantize_ste(y - means) + means

        _, y_likelihoods = self.gaussian_conditional(y, scales, means)
        
        y_bits = calc_bits(y_likelihoods)

        y_dec_aux  = y_hat
        y_dec_main = y_hat
        
        aux_index = 0
        for i,layer in enumerate(self.g_s):
            y_dec_main = layer(y_dec_main)
            if i in [0,2,4,6]:   # shorcut position for decoder
                y_dec_aux = self.AuxT_dec[aux_index](y_dec_aux)
                aux_index += 1
                y_dec_main += y_dec_aux
        x_hat = y_dec_main

        return {
            "x_hat": x_hat,
            "estimated_bits": {"y": y_bits, "z": z_bits},
            "y_likelihoods": y_likelihoods,  # [B*V, M, H_l, W_l]
        }

    def compress(self, x):
        # self.update()
        y_enc_aux = x
        y_enc_main = x
        aux_index = 0
        for i,layer in enumerate(self.g_a): 
            y_enc_main = layer(y_enc_main) 
            if i in [0,2,4,6]:   # shorcut position for decoder
                y_enc_aux = self.AuxT_enc[aux_index](y_enc_aux)
                y_enc_main += y_enc_aux
                aux_index += 1
        y = y_enc_main
        
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        
        y_dec_aux  = y_hat
        y_dec_main = y_hat
        aux_index = 0
        for i, layer in enumerate(self.g_s):
            y_dec_main = layer(y_dec_main)
            # ConvTranspose 가 있는 위치마다 skip
            if i in [0, 2, 4, 6]:
                y_dec_aux = self.AuxT_dec[aux_index](y_dec_aux)
                aux_index += 1
                y_dec_main = y_dec_main + y_dec_aux
        
        # x_hat = y_dec_main.clamp_(0, 1)
        x_hat = y_dec_main
        
        return {"x_hat": x_hat}


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class Quantizer():
    def quantize(self, inputs, quantize_type="noise"):
        if quantize_type == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        elif quantize_type == "ste":
            return torch.round(inputs) - inputs.detach() + inputs
        else:
            return torch.round(inputs)
        
class CheckboardMaskedConv2d(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask:
        [[0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        self.mask[:, :, 0::2, 1::2] = 1
        self.mask[:, :, 1::2, 0::2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)

        return out
    
 
class MeanScaleHyperprior(MeanScaleHyperpriorBase):
    def __init__(
        self,
        N: int = 128,
        M: int = 192,
        num_slices: int = 2,
        groups: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(N=N, M=M, **kwargs)
        self.N = int(N)
        self.M = int(M)
        self.num_slices = num_slices

        # --------- groups 자동 생성 (추천) ---------
        if groups is None:
            # M을 num_slices개로 거의 균등하게 나누기
            base = M // num_slices
            rest = M - base * num_slices  # 나머지는 앞에서 1씩 더해줌

            groups = [0]
            for i in range(num_slices):
                g = base + (1 if i < rest else 0)
                groups.append(g)
            # 예: M=192, num_slices=5 → [0, 39, 39, 38, 38, 38] 이런 식
        self.groups = groups

        assert len(self.groups) == self.num_slices + 1
        assert sum(self.groups[1:]) == self.M, \
            f"sum(groups[1:])={sum(self.groups[1:])} must equal M={self.M}"

        # ---------- ELIC-style entropy coder 모듈들 ----------

        # cross-channel transforms (support_slices -> mean/scale support)
        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(
                    # 입력 채널 수:
                    #   이전 slice(s)의 support 채널들의 합 (조건에 따라 0, g1, g1+g{i-1})
                    self.groups[min(1, i) if i > 0 else 0]
                    + self.groups[i if i > 1 else 0],
                    224,
                    stride=1,
                    kernel_size=5,
                ),
                nn.ReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=5),
                nn.ReLU(inplace=True),
                conv(128, self.groups[i + 1] * 2, stride=1, kernel_size=5),
            )
            for i in range(1, self.num_slices)
        )

        # checkerboard masked conv for context prediction
        self.context_prediction = nn.ModuleList(
            CheckboardMaskedConv2d(
                self.groups[i + 1],
                2 * self.groups[i + 1],
                kernel_size=5,
                padding=2,
                stride=1,
            )
            for i in range(self.num_slices)
        )

        # ParamAggregation (GEP)
        # 입력: [anchor/non-anchor context (2*g_i), support(latent_means/scales + cc_support)]
        # support 채널 수 = 2*M (+ 2*prev_groups if i>0)
        self.ParamAggregation = nn.ModuleList()
        for i in range(self.num_slices):
            prev_group = self.groups[i + 1 if i > 0 else 0]
            this_group = self.groups[i + 1]
            in_ch = 2 * self.M + 2 * prev_group + 2 * this_group
            hidden1 = 2 * self.M
            hidden2 = 2 * self.M
            self.ParamAggregation.append(
                nn.Sequential(
                    conv1x1(in_ch, hidden1),
                    nn.ReLU(inplace=True),
                    conv1x1(hidden1, hidden2),
                    nn.ReLU(inplace=True),
                    conv1x1(hidden2, this_group * 2),
                )
            )

        # 양자화 helper (ELIC 코드 그대로 사용)
        self.quantizer = Quantizer()

        # ELIC 스타일과 동일하게 GaussianConditional 사용
        self.gaussian_conditional = GaussianConditional(None)
        
        self.AuxT_enc = nn.Sequential(
            WLS(3,N),
            WLS(N,N),
            WLS(N,N),
            WLS(N,M),  
        )  
        self.AuxT_dec = nn.Sequential(
            iWLS(M,N),
            iWLS(N,N),
            iWLS(N,N),
            iWLS(N,3),  
        )

    def ortho_loss(self) -> Tensor:
        loss = sum(m.loss() for m in self.modules() if isinstance(m, OLP))
        return cast(Tensor, loss)

    # --------- z / y 공통용 scale_table 업데이트 ---------
    def update(self, scale_table=None, force: bool = False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(
            scale_table, force=force
        )
        updated |= super().update(force=force)
        return updated

    # --------- ELIC-style forward (entropy estimation) ---------
    def forward(self, x, noisequant: bool = False):
        # encoder / hyper encoder 는 부모 클래스의 g_a / h_a 사용

        y_enc_aux = x
        y_enc_main = x
        aux_index = 0
        for i,layer in enumerate(self.g_a): 
            y_enc_main = layer(y_enc_main) 
            if i in [0,2,4,6]: #shorcut position for encoder
                y_enc_aux = self.AuxT_enc[aux_index](y_enc_aux)
                y_enc_main += y_enc_aux
                aux_index += 1
        y = y_enc_main
        
        B, C, H, W = y.size()

        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if not noisequant:
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = quantize_ste(z_tmp) + z_offset

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        # anchor / non-anchor checkerboard 분리
        anchor = torch.zeros_like(y, device=x.device)
        non_anchor = torch.zeros_like(y, device=x.device)

        anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
        non_anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        non_anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]

        y_slices = torch.split(y, self.groups[1:], 1)
        anchor_split = torch.split(anchor, self.groups[1:], 1)
        non_anchor_split = torch.split(non_anchor, self.groups[1:], 1)
        ctx_params_anchor_split = torch.split(
            torch.zeros(B, C * 2, H, W, device=x.device),
            [2 * g for g in self.groups[1:]],
            1,
        )

        y_hat_slices = []
        y_hat_slices_for_gs = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # support_slices 구성
            if slice_index == 0:
                support = torch.cat([latent_means, latent_scales], dim=1)
            else:
                if slice_index == 1:
                    support_slices = y_hat_slices[0]
                else:
                    support_slices = torch.cat(
                        [y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1
                    )
                support_slices_ch = self.cc_transforms[slice_index - 1](
                    support_slices
                )
                support_slices_ch_mean, support_slices_ch_scale = \
                    support_slices_ch.chunk(2, 1)
                support = torch.cat(
                    [
                        support_slices_ch_mean,
                        support_slices_ch_scale,
                        latent_means,
                        latent_scales,
                    ],
                    dim=1,
                )

            # ---------- checkerboard process 1 (anchor) ----------
            y_anchor = anchor_split[slice_index]
            means_anchor, scales_anchor = self.ParamAggregation[slice_index](
                torch.cat(
                    [ctx_params_anchor_split[slice_index], support], dim=1
                )
            ).chunk(2, 1)

            scales_hat_split = torch.zeros_like(y_anchor, device=x.device)
            means_hat_split = torch.zeros_like(y_anchor, device=x.device)

            scales_hat_split[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
            scales_hat_split[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
            means_hat_split[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
            means_hat_split[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]

            if noisequant:
                y_anchor_q = self.quantizer.quantize(y_anchor, "noise")
                y_anchor_q_gs = self.quantizer.quantize(y_anchor, "ste")
            else:
                y_anchor_q = (
                    self.quantizer.quantize(y_anchor - means_anchor, "ste")
                    + means_anchor
                )
                y_anchor_q_gs = (
                    self.quantizer.quantize(y_anchor - means_anchor, "ste")
                    + means_anchor
                )

            y_anchor_q[:, :, 0::2, 1::2] = 0
            y_anchor_q[:, :, 1::2, 0::2] = 0
            y_anchor_q_gs[:, :, 0::2, 1::2] = 0
            y_anchor_q_gs[:, :, 1::2, 0::2] = 0

            # ---------- checkerboard process 2 (non-anchor) ----------
            masked_context = self.context_prediction[slice_index](y_anchor_q)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[
                slice_index
            ](
                torch.cat([masked_context, support], dim=1)
            ).chunk(2, 1)

            scales_hat_split[:, :, 0::2, 1::2] = scales_non_anchor[
                :, :, 0::2, 1::2
            ]
            scales_hat_split[:, :, 1::2, 0::2] = scales_non_anchor[
                :, :, 1::2, 0::2
            ]
            means_hat_split[:, :, 0::2, 1::2] = means_non_anchor[
                :, :, 0::2, 1::2
            ]
            means_hat_split[:, :, 1::2, 0::2] = means_non_anchor[
                :, :, 1::2, 0::2
            ]

            # likelihood for this slice
            _, y_slice_likelihood = self.gaussian_conditional(
                y_slice, scales_hat_split, means=means_hat_split
            )

            y_non_anchor = non_anchor_split[slice_index]
            if noisequant:
                y_non_anchor_q = self.quantizer.quantize(y_non_anchor, "noise")
                y_non_anchor_q_gs = self.quantizer.quantize(
                    y_non_anchor, "ste"
                )
            else:
                y_non_anchor_q = (
                    self.quantizer.quantize(
                        y_non_anchor - means_non_anchor, "ste"
                    )
                    + means_non_anchor
                )
                y_non_anchor_q_gs = (
                    self.quantizer.quantize(
                        y_non_anchor - means_non_anchor, "ste"
                    )
                    + means_non_anchor
                )

            y_non_anchor_q[:, :, 0::2, 0::2] = 0
            y_non_anchor_q[:, :, 1::2, 1::2] = 0
            y_non_anchor_q_gs[:, :, 0::2, 0::2] = 0
            y_non_anchor_q_gs[:, :, 1::2, 1::2] = 0

            y_hat_slice = y_anchor_q + y_non_anchor_q
            y_hat_slice_gs = y_anchor_q_gs + y_non_anchor_q_gs

            y_hat_slices.append(y_hat_slice)
            y_hat_slices_for_gs.append(y_hat_slice_gs)
            y_likelihood.append(y_slice_likelihood)

        y_likelihoods = torch.cat(y_likelihood, dim=1)
        y_hat = torch.cat(y_hat_slices_for_gs, dim=1)

        y_dec_aux  = y_hat
        y_dec_main = y_hat
        
        aux_index = 0
        for i,layer in enumerate(self.g_s):
            y_dec_main = layer(y_dec_main)
            if i in [0,2,4,6]:   # shorcut position for decoder
                y_dec_aux = self.AuxT_dec[aux_index](y_dec_aux)
                aux_index += 1
                y_dec_main += y_dec_aux
        x_hat = y_dec_main


        # 🔹 기존 MeanScaleHyperprior랑 동일하게 bit 수도 계산
        y_bits = calc_bits(y_likelihoods)
        z_bits = calc_bits(z_likelihoods)

        return {
            "x_hat": x_hat,
            "estimated_bits": {"y": y_bits, "z": z_bits},   # 🔹 train.py가 기대하는 키
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},  # 분석용 유지
        }
        

    def compress(self, x):
        self.update()

        # ==========================
        # 1) Encoder (g_a + AuxT_enc)  ->  y
        # ==========================
        y_enc_aux = x
        y_enc_main = x
        aux_index = 0
        for i, layer in enumerate(self.g_a):
            y_enc_main = layer(y_enc_main)
            if i in [0, 2, 4, 6]:
                y_enc_aux = self.AuxT_enc[aux_index](y_enc_aux)
                y_enc_main = y_enc_main + y_enc_aux
                aux_index += 1
        y = y_enc_main

        B, C, H, W = y.size()

        # ==========================
        # 2) Hyper encoder / EB  ->  z_strings, z_hat, latent params
        # ==========================
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        # ==========================
        # 3) Slice-wise + checkerboard entropy coding (AR context)
        # ==========================
        y_slices = torch.split(y, self.groups[1:], dim=1)
        ctx_params_anchor_split = torch.split(
            torch.zeros(B, C * 2, H, W, device=x.device),
            [2 * g for g in self.groups[1:]],
            dim=1,
        )

        # ⚠️ 이 y_strings는 "파이썬 구조" (slice별 anchor/non-anchor bitstream)
        y_strings_struct = []
        y_hat_slices = []

        for slice_index, y_slice in enumerate(y_slices):
            # ---- support 구성 (latent + 이전 slice) ----
            if slice_index == 0:
                support = torch.cat([latent_means, latent_scales], dim=1)
            else:
                if slice_index == 1:
                    support_slices = y_hat_slices[0]
                else:
                    support_slices = torch.cat(
                        [y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1
                    )
                support_slices_ch = self.cc_transforms[slice_index - 1](
                    support_slices
                )
                support_slices_ch_mean, support_slices_ch_scale = \
                    support_slices_ch.chunk(2, 1)
                support = torch.cat(
                    [
                        support_slices_ch_mean,
                        support_slices_ch_scale,
                        latent_means,
                        latent_scales,
                    ],
                    dim=1,
                )

            # ---------- checkerboard 1: anchor ----------
            y_anchor = y_slice.clone()
            means_anchor, scales_anchor = self.ParamAggregation[slice_index](
                torch.cat(
                    [ctx_params_anchor_split[slice_index], support], dim=1
                )
            ).chunk(2, 1)

            B_a, C_a, H_a, W_a = y_anchor.size()

            y_anchor_encode = torch.zeros(
                B_a, C_a, H_a, W_a // 2, device=x.device
            )
            means_anchor_encode = torch.zeros_like(y_anchor_encode)
            scales_anchor_encode = torch.zeros_like(y_anchor_encode)

            y_anchor_encode[:, :, 0::2, :] = y_anchor[:, :, 0::2, 0::2]
            y_anchor_encode[:, :, 1::2, :] = y_anchor[:, :, 1::2, 1::2]
            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            indexes_anchor = self.gaussian_conditional.build_indexes(
                scales_anchor_encode
            )
            anchor_strings = self.gaussian_conditional.compress(
                y_anchor_encode, indexes_anchor, means=means_anchor_encode
            )
            anchor_quantized = self.gaussian_conditional.decompress(
                anchor_strings, indexes_anchor, means=means_anchor_encode
            )

            y_anchor_decode = torch.zeros_like(y_anchor)
            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            # ---------- checkerboard 2: non-anchor ----------
            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[
                slice_index
            ](
                torch.cat([masked_context, support], dim=1)
            ).chunk(2, 1)

            y_non_anchor_encode = torch.zeros_like(y_anchor_encode)
            means_non_anchor_encode = torch.zeros_like(y_anchor_encode)
            scales_non_anchor_encode = torch.zeros_like(y_anchor_encode)

            non_anchor = y_slice.clone()
            y_non_anchor_encode[:, :, 0::2, :] = non_anchor[:, :, 0::2, 1::2]
            y_non_anchor_encode[:, :, 1::2, :] = non_anchor[:, :, 1::2, 0::2]
            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[
                :, :, 0::2, 1::2
            ]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[
                :, :, 1::2, 0::2
            ]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[
                :, :, 0::2, 1::2
            ]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[
                :, :, 1::2, 0::2
            ]

            indexes_non_anchor = self.gaussian_conditional.build_indexes(
                scales_non_anchor_encode
            )
            non_anchor_strings = self.gaussian_conditional.compress(
                y_non_anchor_encode,
                indexes_non_anchor,
                means=means_non_anchor_encode,
            )
            non_anchor_quantized = self.gaussian_conditional.decompress(
                non_anchor_strings,
                indexes_non_anchor,
                means=means_non_anchor_encode,
            )

            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[
                :, :, 0::2, :
            ]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[
                :, :, 1::2, :
            ]

            y_slice_hat = y_anchor_decode + y_non_anchor_quantized
            y_hat_slices.append(y_slice_hat)

            # 🔹 slice마다 [anchor_strings, non_anchor_strings] 저장
            #   anchor_strings / non_anchor_strings 둘 다 list[bytes] (batch 크기만큼)
            y_strings_struct.append([anchor_strings, non_anchor_strings])

        # 🔹 여기서 전체 구조를 pickle로 bytes로 싸서,
        #   ops.write_body가 기대하는 형식: [ [bytes], [bytes] ] 로 맞춰줌
        y_packed = pickle.dumps(y_strings_struct)
        y_strings_for_file = [y_packed]   # batch=1 가정

        return {
            "strings": [y_strings_for_file, z_strings],
            "shape": z.size()[-2:],
        }


    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        # ==========================
        # 1) z 복호화 + latent params
        # ==========================
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        B, _, H_z, W_z = z_hat.size()

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        # y의 공간 해상도 (stride=16 이라고 가정하는 기존 코드 유지)
        H_y, W_y = H_z * 4, W_z * 4

        # 🔹 파일에서 읽힌 y_strings는 [ packed_bytes ] 형태 (batch=1 가정)
        raw_y_strings = strings[0]
        assert len(raw_y_strings) == 1, "현재 구현은 batch=1만 지원"
        # 🔹 compress에서 pickle.dumps 했던 구조 복원
        y_strings_struct = pickle.loads(raw_y_strings[0])

        C_total = sum(self.groups[1:])
        ctx_params_anchor = torch.zeros(
            B, 2 * C_total, H_y, W_y, device=z_hat.device
        )
        ctx_params_anchor_split = torch.split(
            ctx_params_anchor,
            [2 * g for g in self.groups[1:]],
            dim=1,
        )

        # ==========================
        # 2) slice-wise y 복호화 (checkerboard)
        # ==========================
        y_hat_slices = []
        for slice_index in range(len(self.groups) - 1):
            # ---- support 구성 ----
            if slice_index == 0:
                support = torch.cat([latent_means, latent_scales], dim=1)
            else:
                if slice_index == 1:
                    support_slices = y_hat_slices[0]
                else:
                    support_slices = torch.cat(
                        [y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1
                    )
                support_slices_ch = self.cc_transforms[slice_index - 1](
                    support_slices
                )
                support_slices_ch_mean, support_slices_ch_scale = \
                    support_slices_ch.chunk(2, 1)
                support = torch.cat(
                    [
                        support_slices_ch_mean,
                        support_slices_ch_scale,
                        latent_means,
                        latent_scales,
                    ],
                    dim=1,
                )

            # ---------- checkerboard 1: anchor ----------
            means_anchor, scales_anchor = self.ParamAggregation[slice_index](
                torch.cat(
                    [ctx_params_anchor_split[slice_index], support], dim=1
                )
            ).chunk(2, 1)

            B_a, C_a, H_a, W_a = means_anchor.size()
            means_anchor_encode = torch.zeros(
                B_a, C_a, H_a, W_a // 2, device=z_hat.device
            )
            scales_anchor_encode = torch.zeros_like(means_anchor_encode)
            y_anchor_decode = torch.zeros(
                B_a, C_a, H_a, W_a, device=z_hat.device
            )

            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            indexes_anchor = self.gaussian_conditional.build_indexes(
                scales_anchor_encode
            )

            # 🔹 compress에서 저장한 slice별 bitstream 복원
            anchor_strings, non_anchor_strings = y_strings_struct[slice_index]
            anchor_quantized = self.gaussian_conditional.decompress(
                anchor_strings, indexes_anchor, means=means_anchor_encode
            )

            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            # ---------- checkerboard 2: non-anchor ----------
            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[
                slice_index
            ](
                torch.cat([masked_context, support], dim=1)
            ).chunk(2, 1)

            means_non_anchor_encode = torch.zeros_like(means_anchor_encode)
            scales_non_anchor_encode = torch.zeros_like(means_anchor_encode)

            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[
                :, :, 0::2, 1::2
            ]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[
                :, :, 1::2, 0::2
            ]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[
                :, :, 0::2, 1::2
            ]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[
                :, :, 1::2, 0::2
            ]

            indexes_non_anchor = self.gaussian_conditional.build_indexes(
                scales_non_anchor_encode
            )
            non_anchor_quantized = self.gaussian_conditional.decompress(
                non_anchor_strings,
                indexes_non_anchor,
                means=means_non_anchor_encode,
            )

            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[
                :, :, 0::2, :
            ]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[
                :, :, 1::2, :
            ]

            y_slice_hat = y_anchor_decode + y_non_anchor_quantized
            y_hat_slices.append(y_slice_hat)

        # ==========================
        # 3) Synthesis (g_s + AuxT_dec)  ->  x_hat
        # ==========================
        y_hat = torch.cat(y_hat_slices, dim=1)

        y_dec_aux = y_hat
        y_dec_main = y_hat
        aux_index = 0
        for i, layer in enumerate(self.g_s):
            y_dec_main = layer(y_dec_main)
            if i in [0, 2, 4, 6]:
                y_dec_aux = self.AuxT_dec[aux_index](y_dec_aux)
                aux_index += 1
                y_dec_main = y_dec_main + y_dec_aux

        x_hat = y_dec_main
        return {"x_hat": x_hat}

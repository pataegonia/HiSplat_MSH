# Findings

## F1. Anchor Gap Source

The MSH anchor with M320 and depth `[2,4,4]` scores 26.916 PSNR against the vanilla HiSplat ceiling of 27.194 PSNR. The −0.278 dB gap is structural.

Rescue ablation at 30k:

| arm | change | PSNR | conclusion |
|---|---|---:|---|
| 1 | longer training, M320 d244 | 26.974 | training length is not the main cause |
| 2 | depth `[2,2,2]` | 27.162 | downsample depth is the main lever |
| 3 | M `[512,512,512]` | 27.062 | width helps less and was still not enough |

Interpretation: depth `[2,4,4]` creates a strong RD frontier. Reducing depth recovers render quality but increases rate sharply.

## F2. Probe RD Shape

Forward eval at step 15000 shows the useful lambda region is below 0.02:

| lambda | estimated KB | PSNR |
|---:|---:|---:|
| 0 | 304.1 | 26.418 |
| 0.02 | 1.44 | 25.909 |
| 0.2 | 0.134 | 25.523 |

Use eval `estimated_kb` for plots. Training-log `estimated_kb` differs because it is batch/random-crop dependent.

## F3. Real-Bytes Failure Root Cause

`compress=true` failures were not caused by EntropyBottleneck quantile staleness. Aux convergence and `update_quantiles=True` did not fix the explosions.

The decisive diagnostic was job 345379:

- `scales_diff=4.76837e-07`
- `indexes_diff=1`
- `y_hat_real_diff=3.95764e+08`
- `x_hat_real_max=390912`

Root cause: cuDNN produced tiny differences in `h_s(z_hat)` between compress-time and decompress-time recomputation. Near a GaussianConditional scale-table boundary, that changed one integer CDF index and desynchronized the range decoder.

Fix: snap scales to a 1/1024 grid before `build_indexes(scales)` in both `compress()` and `decompress()`. Job 345396 then completed the 6474-scene real eval with 25.9041 PSNR and 1.5071 KB.

## F4. Warmstart RD Curve Shift

Anchor warm-start (5k bypass+recon_w=1 codec, the `msh_proj_full_highrate_bypass_5k` checkpoint) followed by 70k forward+recon_w=0+λ training shifts the entire RD curve up by +0.08 to +0.22 dB compared to fresh `hisplat_re10k` start, real arithmetic-coded bytes:

| λ | fresh 50k PSNR/KB | warmstart 70k PSNR/KB | Δ PSNR |
|---:|---|---|---:|
| 0.0   | 26.616 / 491.6 | 26.835 / 640.5 | +0.219 |
| 0.003 | 26.415 / 9.45  | 26.542 / 10.43 | +0.127 |
| 0.007 | 26.350 / 6.25  | 26.428 / 6.64  | +0.078 |
| 0.012 | 26.199 / 3.23  | 26.354 / 4.94  | +0.155 |
| 0.02  | 26.085 / 1.91  | 26.269 / 3.64  | +0.184 |

Interpretation: learning setup is a real lever, not just architecture. Anchor warm-start gives the codec a feature-fidelity initialization before being forced through the render path, so the RD curve sits on a better operating point. KB drifts up by 6 to 90% at the same λ because the warm-started codec keeps a richer representation.

## F5. recon_w Ablation in Warmup Setup

With λ=0 fixed and anchor warmstart, sweeping `CODEC_RECON_WEIGHT` from 0.0 to 0.3 improves PSNR, SSIM, and LPIPS simultaneously while KB grows:

| recon_w | PSNR | SSIM | LPIPS | compressed_kb |
|---:|---:|---:|---:|---:|
| 0.0 | 26.835 | 0.8753 | 0.1244 | 640.5 |
| 0.01 | 26.911 | 0.8767 | 0.1227 | 893.1 |
| 0.03 | 26.947 | 0.8773 | 0.1223 | 957.9 |
| 0.1  | 26.974 | 0.8778 | 0.1221 | 1008.8 |
| 0.3  | 26.992 | 0.8780 | 0.1222 | 1040.1 |

This contradicts [F1 mechanism #1](findings.md) (recon-render misalignment, recon_w↑ degrades render PSNR). The resolution: F1 was derived from finetune setup where generator and codec train jointly, which triggers the misalignment. The warmup setup here has generator frozen, so misalignment does not arise and recon_w acts as a clean feature-fidelity regularizer.

Also: 26.992 at recon_w=0.3 is the new measured forward ceiling for d244, slightly above rescue ① 26.974. The vanilla 27.194 ceiling is still unreachable without depth reduction.

## F6. λ=0.001 Fragile Zone in Forward + recon_w=0

With `CODEC_WARMUP_BYPASS=false` and `CODEC_RECON_WEIGHT=0`, the λ=0.001 point exploded twice (jobs 345702 at step 41k, 345978 resumed from step 40k exploded at step 14.5k). All other λ values (0.0, 0.003, 0.007, 0.012, 0.02) completed 70k without explosion under the same setup.

`loss/aux` traces show λ↑ accelerates aux drift (λ=0.003: aux ~150k at 70k; λ=0.02: ~320k), but only λ=0.001 explodes. Hypothesis: rate pressure at λ=0.001 is just strong enough to drift the z distribution but not strong enough to settle the codec into a new operating point. Larger λ forces a clear new operating point; λ=0 keeps the anchor operating point untouched.

Operational fix: recon_w≥0.01 holds aux at the anchor value (23,615) across all four ablation runs (recon_w 0.01/0.03/0.1/0.3), confirming that a small recon term acts as a quantile drift dampener. The λ=0.001 retrain uses recon_w=0.01 + LR_AUX=1e-3 as a safety net.


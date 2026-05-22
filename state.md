# CURRENT STATE (2026-05-22)

HiSplat + per-stage MSH has the headline RD curve plus a recon_w lever ablation, both validated on real arithmetic-coded bytes.

Three locked findings now in place:

- Rescue ablation (2026-05-19): the −0.28 dB anchor gap is caused by AE downsample depth `[2,4,4]`. Reducing depth to `[2,2,2]` reaches near-vanilla quality but costs about 9× more rate.
- Real-bytes compression validated (2026-05-20): symmetric 1/1024 scale snapping before `build_indexes` in `compress()` and `decompress()`, plus deterministic cuDNN during `test.compress=true`. Job 345396 finished full 6474-scene eval at 25.9041 PSNR / 1.5071 KB.
- Warmstart RD shift + recon_w ablation (2026-05-22): anchor-warmstart 70k training shifts the entire RD curve up by +0.08 to +0.22 dB versus fresh 50k. In warmup setup, recon_w↑ improves PSNR, SSIM, and LPIPS simultaneously, contradicting the F1 misalignment claim that was derived from finetune setup.

## Headline RD frontier (warmstart 70k, recon_w=0, real bytes)

| λ | PSNR | SSIM | LPIPS | compressed_kb |
|---:|---:|---:|---:|---:|
| 0.0   | 26.835 | 0.8753 | 0.1244 | 640.5 |
| 0.001 | — | — | — | — |
| 0.003 | 26.542 | 0.8698 | 0.1327 | 10.43 |
| 0.007 | 26.428 | 0.8673 | 0.1359 | 6.64 |
| 0.012 | 26.354 | 0.8658 | 0.1379 | 4.94 |
| 0.02  | 26.269 | 0.8642 | 0.1404 | 3.64 |

λ=0.001 is currently retraining with recon_w=0.01 + LR_AUX=1e-3 safety net; recon_w=0 exploded twice at this λ.

Compared to the previous fresh 50k curve: warmstart adds +0.08 to +0.22 dB at every λ and shifts KB slightly up.

## recon_w ablation (warmstart 70k, λ=0.0, real bytes)

| recon_w | PSNR | SSIM | LPIPS | compressed_kb |
|---:|---:|---:|---:|---:|
| 0.0 | 26.835 | 0.8753 | 0.1244 | 640.5 |
| 0.01 | 26.911 | 0.8767 | 0.1227 | 893.1 |
| 0.03 | 26.947 | 0.8773 | 0.1223 | 957.9 |
| 0.1  | 26.974 | 0.8778 | 0.1221 | 1008.8 |
| 0.3  | **26.992** | **0.8780** | 0.1222 | 1040.1 |

26.992 (recon_w=0.3) is the new measured forward ceiling for d244, slightly above rescue ① 26.974. The vanilla 27.194 ceiling is still out of reach without depth reduction.

## Ceiling reference points

- vanilla HiSplat (no codec): 27.194 PSNR (job 344145)
- rescue ② depth `[2,2,2]`: 27.162 PSNR @ ~11 MB (job 344590)
- d244 anchor near-lossless (forward eval): 26.916 PSNR @ ~1.2 MB (job 344152)
- d244 warmstart with recon_w=0.3: 26.992 PSNR @ ~1.0 MB (job 346336)

## Pending

- λ=0.001 retrain in progress with recon_w=0.01 + LR_AUX=1e-3 safety net (recon_w=0 exploded twice at this λ).

## Next actions

1. Pick up λ=0.001 result, place it on the headline curve to complete the 6-point sweep.
2. Optional: extend recon_w sweep with one more point (recon_w=1.0) to see where the lever turns over.
3. Update [findings.md](findings.md), [gotchas.md](gotchas.md) with the warmstart/recon_w/λ=0.001 findings (done as part of this update).
4. Plot the final figures and write the report. Headline figure: warmstart curve (solid) vs fresh curve (dotted) vs vanilla ceiling (horizontal) plus rescue ② isolated point. Secondary figure: recon_w sweep along PSNR/KB axes.


# Gotchas

## C1. Rate Term Requires Full Codec Mode

If `LAMBDA_RD>0`, use `CODEC_TRAIN_MODE=full` or `entropy`. Other modes do not return `estimated_bits`, so the rate term is silently absent. The training slurm script has a guard for this.

## C2. Warmup Bypass PSNR Is Blind To Compression

With warmup bypass, rendered features are the original features. Use separate forward/compress eval to measure actual RD behavior.

## C3. Eval KB Is Authoritative For RD Plots

Training `estimated_kb` and eval `estimated_kb` are not directly comparable because training uses batch/random crops. Use eval values for figures.

## C4. `VAL_INTERVAL` Must Fit The Epoch

Keep `VAL_INTERVAL <= 33017` for the current RealEstate10K setup. Higher values can fail before training starts.

## C5. Save All Sweep Checkpoints

Use `SAVE_TOP_K=-1` for sweep runs. `SAVE_TOP_K=1` may leave only the latest/global-step checkpoint and break offline checkpoint selection.

## C6. W&B Timeout

Cluster jobs can die on W&B service startup. Slurm defaults now use `WANDB_MODE=disabled`. Override explicitly only when needed.

## C7. `build_indexes(scales)` Boundary Fragility

Real entropy coding recomputes `h_s(z_hat)` during decompression. Tiny cuDNN differences can flip one CDF index. Keep scale snapping symmetric in `compress()` and `decompress()`.

## C8. Compression Patch Deployment

Before running real-bytes eval on another checkout or cluster copy, verify:

```bash
grep -n "_build_stable_indexes" src/model/encoder/compression/stage_hyperprior.py
```

## C9. Debug Verbosity

Codec roundtrip diagnostics are disabled by default. Use `MSH_CODEC_DEBUG=1` only for targeted real-bytes debugging, because full eval produces tens of thousands of diagnostic lines.

## C10. λ=0.001 Fragile Zone with recon_w=0

In `forward + CODEC_RECON_WEIGHT=0 + LAMBDA_RD=0.001` setup the codec explodes mid-training (`Stage 2 codec x_hat magnitude exploded in forward/full mode`). Reproduced twice (jobs 345702 step 41k, 345978 step 54k cumulative). λ=0 and λ≥0.003 in the same setup all complete 70k. The aux quantile is suspected: at λ=0.001 the rate pressure is just enough to drift z, but not enough to settle the codec at a new operating point.

Safety net for any forward sweep job: set `CODEC_RECON_WEIGHT≥0.01`. This holds aux at the anchor value (23,615) across all four ablation runs and prevents explosion. The retrain of λ=0.001 uses recon_w=0.01 + LR_AUX=1e-3.


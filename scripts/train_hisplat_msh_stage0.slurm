#!/usr/bin/env bash
#SBATCH -J hisplat-msh-stage0
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v11
#SBATCH -t 6-00:00:00
#SBATCH -o slurm-hisplat-msh-stage0-%A.out
#SBATCH -e slurm-hisplat-msh-stage0-%A.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export STAGE_SCHEDULE="${STAGE_SCHEDULE:-stage0_only}"
export PHASE="${PHASE:-finetune}"
export MAX_STEPS="${MAX_STEPS:-30000}"
export CODEC_TRAIN_MODE="${CODEC_TRAIN_MODE:-full}"
export CODEC_WARMUP_BYPASS="${CODEC_WARMUP_BYPASS:-false}"
export LAMBDA_RD="${LAMBDA_RD:-0.0}"
export LAMBDA_RD_PER_STAGE="${LAMBDA_RD_PER_STAGE:-[${LAMBDA_RD},0,0]}"
export OUTPUT_NAME="${OUTPUT_NAME:-hisplat_msh_prog_stage0_$(date +%Y%m%d_%H%M%S)}"
export WANDB_NAME="${WANDB_NAME:-hisplat-msh-prog-stage0}"

exec bash "${SCRIPT_DIR}/train_hisplat_msh.slurm"

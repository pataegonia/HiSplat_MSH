#!/usr/bin/env bash
#SBATCH -J hisplat-msh-joint
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v11
#SBATCH -t 6-00:00:00
#SBATCH -o slurm-hisplat-msh-joint-%A.out
#SBATCH -e slurm-hisplat-msh-joint-%A.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export STAGE_SCHEDULE="${STAGE_SCHEDULE:-joint}"
export PHASE="${PHASE:-cooldown}"
export CODEC_TRAIN_MODE="${CODEC_TRAIN_MODE:-full}"
export CODEC_WARMUP_BYPASS="${CODEC_WARMUP_BYPASS:-false}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-__AUTO_FROM_PREV__}"
export LAMBDA_RD="${LAMBDA_RD:-0.0}"
export LAMBDA_RD_STAGE0="${LAMBDA_RD_STAGE0:-${LAMBDA_RD}}"
export LAMBDA_RD_STAGE1="${LAMBDA_RD_STAGE1:-${LAMBDA_RD}}"
export LAMBDA_RD_STAGE2="${LAMBDA_RD_STAGE2:-${LAMBDA_RD}}"
export LAMBDA_RD_PER_STAGE="${LAMBDA_RD_PER_STAGE:-[${LAMBDA_RD_STAGE0},${LAMBDA_RD_STAGE1},${LAMBDA_RD_STAGE2}]}"
export OUTPUT_NAME="${OUTPUT_NAME:-hisplat_msh_prog_joint_$(date +%Y%m%d_%H%M%S)}"
export WANDB_NAME="${WANDB_NAME:-hisplat-msh-prog-joint}"

exec bash "${SCRIPT_DIR}/train_hisplat_msh.slurm"

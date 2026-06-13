#!/usr/bin/env bash
#SBATCH -J hisplat-msh
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v11
#SBATCH -t 6-00:00:00
#SBATCH -o slurm-hisplat-msh-%A.out
#SBATCH -e slurm-hisplat-msh-%A.err

set -euo pipefail

# ============================================================
# HiSplat + per-stage MSH training
#
# PHASE=1 or warmup:
#   Load pretrained HiSplat weights, optimize MSH codec params only.
#   By default, codec output is bypassed from rendering and used only for recon loss.
#
# PHASE=2 or finetune:
#   Load Phase 1 checkpoint weights, optimize HiSplat + MSH with split LR.
#
# PHASE=3 or cooldown:
#   Load finetune checkpoint weights, continue with reduced LRs.
#
# Evaluation should use:
#   mode=test test.compress=true model.encoder.use_gauss_feature_compression=true
# ============================================================

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_DIR="${REPO_DIR:-${SUBMIT_DIR}}"

# ---- User settings ----
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hisplat}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-checkpoints/hisplat_re10k.ckpt}"
PHASE="${PHASE:-1}" # 1|warmup, 2|finetune, 3|cooldown
STAGE_SCHEDULE="${STAGE_SCHEDULE:-joint}" # joint|stage0_only|stage1_only|stage2_only
MAX_STEPS="${MAX_STEPS:-50000}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-2}"
VAL_INTERVAL="${VAL_INTERVAL:-1000}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
CKPT_INTERVAL="${CKPT_INTERVAL:-5000}"
SAVE_TOP_K="${SAVE_TOP_K:-1}"
SH_DEGREE="${SH_DEGREE:-4}"
LAMBDA_RD="${LAMBDA_RD:-0.0}"
LR_CODEC="${LR_CODEC:-1.0e-4}"
LR_AUX="${LR_AUX:-1.0e-5}"
LR_GENERATOR="${LR_GENERATOR:-2.0e-5}"
LR_CODEC_COOLDOWN="${LR_CODEC_COOLDOWN:-2.0e-5}"
LR_GENERATOR_COOLDOWN="${LR_GENERATOR_COOLDOWN:-5.0e-6}"
CODEC_LR_WARMUP_STEPS="${CODEC_LR_WARMUP_STEPS:-200}"
CODEC_LR_WARMUP_START_FACTOR="${CODEC_LR_WARMUP_START_FACTOR:-0.1}"
CODEC_RECON_WEIGHT="${CODEC_RECON_WEIGHT:-1.0}"
CODEC_WARMUP_BYPASS="${CODEC_WARMUP_BYPASS:-true}"
TRUNCATION_AUG_PROBS="${TRUNCATION_AUG_PROBS:-null}"
LAMBDA_RD_PER_STAGE="${LAMBDA_RD_PER_STAGE:-null}"
CODEC_INPUT="${CODEC_INPUT:-proj}" # proj|stack
CODEC_STAGE_N="${CODEC_STAGE_N:-[192,192,192]}"
CODEC_STAGE_M="${CODEC_STAGE_M:-[320,320,320]}"
CODEC_STAGE_DEPTH="${CODEC_STAGE_DEPTH:-[2,4,4]}"
CODEC_TRAIN_MODE="${CODEC_TRAIN_MODE:-transform}" # transform|quantize|quantize_hp|full
WANDB_MODE="${WANDB_MODE:-disabled}"
OUTPUT_NAME="${OUTPUT_NAME:-hisplat_msh_${CODEC_TRAIN_MODE}_phase${PHASE}_lmd${LAMBDA_RD}_$(date +%Y%m%d_%H%M%S)}"

if [[ "${LAMBDA_RD}" =~ ^-?[0-9]+$ ]]; then
    LAMBDA_RD="${LAMBDA_RD}.0"
fi

case "${CODEC_TRAIN_MODE}" in
    transform|quantize|quantize_hp|full|entropy)
        ;;
    *)
        echo "ERROR: CODEC_TRAIN_MODE must be transform, quantize, quantize_hp, or full."
        exit 1
        ;;
esac

# Guard C1: rate term is silently dropped unless estimated_bits exist.
# Only full/entropy return estimated_bits, so LAMBDA_RD>0 with any other
# mode trains a rate-on run that has no rate loss at all.
if awk -v x="${LAMBDA_RD}" 'BEGIN { exit !(x + 0 > 0) }'; then
    case "${CODEC_TRAIN_MODE}" in
        full|entropy)
            ;;
        *)
            echo "ERROR: LAMBDA_RD=${LAMBDA_RD} > 0 requires CODEC_TRAIN_MODE=full (got ${CODEC_TRAIN_MODE}); the rate term is silently dropped otherwise."
            exit 1
            ;;
    esac
fi
if [[ "${LAMBDA_RD_PER_STAGE}" != "null" && "${LAMBDA_RD_PER_STAGE}" != "None" ]]; then
    case "${CODEC_TRAIN_MODE}" in
        full|entropy)
            ;;
        *)
            echo "ERROR: LAMBDA_RD_PER_STAGE=${LAMBDA_RD_PER_STAGE} requires CODEC_TRAIN_MODE=full (got ${CODEC_TRAIN_MODE})."
            exit 1
            ;;
    esac
fi

case "${CODEC_INPUT}" in
    proj|stack)
        ;;
    *)
        echo "ERROR: CODEC_INPUT must be proj or stack."
        exit 1
        ;;
esac

# ---- Auto-resolve previous phase checkpoint ----
if [[ "${PRETRAINED_CKPT}" == "__AUTO_FROM_PHASE1__" || "${PRETRAINED_CKPT}" == "__AUTO_FROM_PREV__" ]]; then
    PREV_OUTPUT_DIR="${PREV_OUTPUT_DIR:-${PHASE1_OUTPUT_DIR:-}}"
    if [[ -z "${PREV_OUTPUT_DIR}" ]]; then
        echo "ERROR: PREV_OUTPUT_DIR or PHASE1_OUTPUT_DIR must be set for ${PRETRAINED_CKPT}."
        exit 1
    fi
    if [[ "${PREV_OUTPUT_DIR}" = /* ]]; then
        CKPT_DIR="${PREV_OUTPUT_DIR}/checkpoints"
    else
        CKPT_DIR="${REPO_DIR}/outputs/${PREV_OUTPUT_DIR}/checkpoints"
    fi
    PRETRAINED_CKPT="$(ls -1 "${CKPT_DIR}"/*.ckpt 2>/dev/null \
        | sed -E 's/.*step[_=]([0-9]+).ckpt/\1 &/' \
        | sort -n \
        | tail -1 \
        | awk '{print $2}')"
    if [[ -z "${PRETRAINED_CKPT}" ]]; then
        echo "ERROR: No checkpoint found in ${CKPT_DIR}"
        exit 1
    fi
fi

# ---- Phase mapping ----
case "${PHASE}" in
    1|warmup)
        TRAIN_PHASE="warmup"
        ;;
    2|finetune)
        TRAIN_PHASE="finetune"
        ;;
    3|cooldown)
        TRAIN_PHASE="cooldown"
        ;;
    *)
        echo "ERROR: PHASE must be 1/warmup, 2/finetune, or 3/cooldown."
        exit 1
        ;;
esac
WANDB_NAME="${WANDB_NAME:-hisplat_msh_${TRAIN_PHASE}_lmd${LAMBDA_RD}}"

# ---- Conda ----
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Conda not found."
    exit 1
fi
conda activate "${CONDA_ENV_NAME}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_DIR}"

echo "============================================"
echo "HiSplat + per-stage MSH training"
echo "Repo: ${REPO_DIR}"
echo "Phase: ${TRAIN_PHASE}"
echo "Stage schedule: ${STAGE_SCHEDULE}"
echo "Lambda RD: ${LAMBDA_RD}"
echo "Pretrained ckpt: ${PRETRAINED_CKPT}"
echo "Max steps: ${MAX_STEPS}"
echo "Batch size/GPU: ${BATCH_SIZE_PER_GPU}"
echo "Val interval/check epoch: ${VAL_INTERVAL} / ${CHECK_VAL_EVERY_N_EPOCH}"
echo "Checkpoint interval/top-k: ${CKPT_INTERVAL} / ${SAVE_TOP_K}"
echo "SH degree: ${SH_DEGREE}"
echo "Codec stage N: ${CODEC_STAGE_N}"
echo "Codec stage M: ${CODEC_STAGE_M}"
echo "Codec stage depth: ${CODEC_STAGE_DEPTH}"
echo "Codec train mode: ${CODEC_TRAIN_MODE}"
echo "Codec input: ${CODEC_INPUT}"
echo "LR codec/aux/generator: ${LR_CODEC} / ${LR_AUX} / ${LR_GENERATOR}"
echo "Codec LR warmup: ${CODEC_LR_WARMUP_STEPS} steps from factor ${CODEC_LR_WARMUP_START_FACTOR}"
echo "Codec recon weight: ${CODEC_RECON_WEIGHT}"
echo "Codec warmup bypass: ${CODEC_WARMUP_BYPASS}"
echo "Truncation aug probs: ${TRUNCATION_AUG_PROBS}"
echo "Lambda RD per stage: ${LAMBDA_RD_PER_STAGE}"
echo "Output: ${OUTPUT_NAME}"
echo "W&B mode: ${WANDB_MODE}"
echo "W&B name: ${WANDB_NAME}"
echo "Extra Hydra args: ${EXTRA_HYDRA_ARGS:-<none>}"
echo "============================================"

read -ra EXTRA_ARGS_ARR <<< "${EXTRA_HYDRA_ARGS:-}"

python -m src.main \
    "+experiment=re10k" \
    "mode=train" \
    "device=1" \
    "dataset.roots=[${DATASET_ROOT}]" \
    "data_loader.train.batch_size=${BATCH_SIZE_PER_GPU}" \
    "data_loader.train.num_workers=${SLURM_CPUS_PER_GPU:-8}" \
    "output_dir=${OUTPUT_NAME}" \
    "trainer.max_steps=${MAX_STEPS}" \
    "trainer.val_check_interval=${VAL_INTERVAL}" \
    "trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}" \
    "checkpointing.load=${PRETRAINED_CKPT}" \
    "checkpointing.resume=false" \
    "checkpointing.every_n_train_steps=${CKPT_INTERVAL}" \
    "checkpointing.save_top_k=${SAVE_TOP_K}" \
    "model.encoder.gaussian_adapter.sh_degree=${SH_DEGREE}" \
    "model.encoder.use_gauss_feature_compression=true" \
    "model.encoder.codec_stage_N=${CODEC_STAGE_N}" \
    "model.encoder.codec_stage_M=${CODEC_STAGE_M}" \
    "model.encoder.codec_stage_depth=${CODEC_STAGE_DEPTH}" \
    "model.encoder.codec_train_mode=${CODEC_TRAIN_MODE}" \
    "model.encoder.codec_input=${CODEC_INPUT}" \
    "train.phase=${TRAIN_PHASE}" \
    "train.stage_schedule=${STAGE_SCHEDULE}" \
    "train.lambda_rd=${LAMBDA_RD}" \
    "train.lr_codec=${LR_CODEC}" \
    "train.lr_aux=${LR_AUX}" \
    "train.lr_generator=${LR_GENERATOR}" \
    "train.lr_codec_cooldown=${LR_CODEC_COOLDOWN}" \
    "train.lr_generator_cooldown=${LR_GENERATOR_COOLDOWN}" \
    "train.codec_lr_warmup_steps=${CODEC_LR_WARMUP_STEPS}" \
    "train.codec_lr_warmup_start_factor=${CODEC_LR_WARMUP_START_FACTOR}" \
    "train.codec_recon_weight=${CODEC_RECON_WEIGHT}" \
    "train.codec_warmup_bypass=${CODEC_WARMUP_BYPASS}" \
    "train.truncation_aug_probs=${TRUNCATION_AUG_PROBS}" \
    "train.lambda_rd_per_stage=${LAMBDA_RD_PER_STAGE}" \
    "test.compress=false" \
    "wandb.mode=${WANDB_MODE}" \
    "wandb.name=${WANDB_NAME}" \
    "${EXTRA_ARGS_ARR[@]}"

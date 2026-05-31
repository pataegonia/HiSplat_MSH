#!/usr/bin/env bash
#SBATCH -J hisplat-msh-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 1-00:00:00
#SBATCH -o slurm-hisplat-msh-eval-%A.out
#SBATCH -e slurm-hisplat-msh-eval-%A.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_DIR="${REPO_DIR:-${SUBMIT_DIR}}"

# ---- User settings ----
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hisplat}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
CKPT="${CKPT:-outputs/msh_proj_full_ratewarm_l1e-5_15k/checkpoints/epoch_0-step_5000.ckpt}"
OUTPUT_NAME="${OUTPUT_NAME:-eval_ratewarm_l1e-5_s5000_compress}"
WANDB_NAME="${WANDB_NAME:-${OUTPUT_NAME}}"
SH_DEGREE="${SH_DEGREE:-4}"
CODEC_INPUT="${CODEC_INPUT:-proj}"
CODEC_STAGE_N="${CODEC_STAGE_N:-[192,192,192]}"
CODEC_STAGE_M="${CODEC_STAGE_M:-[320,320,320]}"
CODEC_STAGE_DEPTH="${CODEC_STAGE_DEPTH:-[2,4,4]}"
CODEC_TRAIN_MODE="${CODEC_TRAIN_MODE:-full}"
TEST_COMPRESS="${TEST_COMPRESS:-true}"

case "${CODEC_TRAIN_MODE}" in
    full|entropy)
        ;;
    *)
        echo "ERROR: MSH eval needs CODEC_TRAIN_MODE=full (or entropy). Current: ${CODEC_TRAIN_MODE}"
        exit 1
        ;;
esac

if [[ "${CKPT}" = /* ]]; then
    CKPT_PATH="${CKPT}"
else
    CKPT_PATH="${REPO_DIR}/${CKPT}"
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT_PATH}"
    exit 1
fi

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
echo "HiSplat + per-stage MSH eval"
echo "Repo: ${REPO_DIR}"
echo "Dataset root: ${DATASET_ROOT}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Output: ${OUTPUT_NAME}"
echo "W&B name: ${WANDB_NAME}"
echo "SH degree: ${SH_DEGREE}"
echo "Codec stage N: ${CODEC_STAGE_N}"
echo "Codec stage M: ${CODEC_STAGE_M}"
echo "Codec stage depth: ${CODEC_STAGE_DEPTH}"
echo "Codec train mode: ${CODEC_TRAIN_MODE}"
echo "Codec input: ${CODEC_INPUT}"
echo "Test compress: ${TEST_COMPRESS}"
echo "Extra Hydra args: ${EXTRA_HYDRA_ARGS:-<none>}"
echo "============================================"

read -ra EXTRA_ARGS_ARR <<< "${EXTRA_HYDRA_ARGS:-}"

python -m src.main \
    "+experiment=re10k" \
    "mode=test" \
    "device=1" \
    "dataset.roots=[${DATASET_ROOT}]" \
    "data_loader.test.num_workers=${SLURM_CPUS_PER_GPU:-8}" \
    "output_dir=${OUTPUT_NAME}" \
    "checkpointing.load=${CKPT_PATH}" \
    "model.encoder.gaussian_adapter.sh_degree=${SH_DEGREE}" \
    "model.encoder.use_gauss_feature_compression=true" \
    "model.encoder.codec_stage_N=${CODEC_STAGE_N}" \
    "model.encoder.codec_stage_M=${CODEC_STAGE_M}" \
    "model.encoder.codec_stage_depth=${CODEC_STAGE_DEPTH}" \
    "model.encoder.codec_train_mode=${CODEC_TRAIN_MODE}" \
    "model.encoder.codec_input=${CODEC_INPUT}" \
    "test.compress=${TEST_COMPRESS}" \
    "test.compute_scores=true" \
    "wandb.mode=offline" \
    "wandb.name=${WANDB_NAME}" \
    "${EXTRA_ARGS_ARR[@]}"

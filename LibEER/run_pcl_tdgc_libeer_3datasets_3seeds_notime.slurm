#!/bin/bash
#SBATCH --job-name=pcl_3ds
#SBATCH --partition=gpu
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err

set -euo pipefail

DATASET_KEYS=(seed seediv deap)
SEEDS=(2024 2025 2026)

DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
DATASET_KEY="${DATASET_KEYS[$DATASET_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
MODEL="PCL_TDGCN"
EPOCHS=1000
LR=0.001

mkdir -p logs

module purge
module load conda/23.3_with_py3.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate libeer

cd /share/home/luzhixuan/projects/LibEER/LibEER

module list
which python
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"

case "${DATASET_KEY}" in
  seed)
    DATASET="seed_de_lds"
    SETTING="seed_sub_independent_train_val_test_setting"
    DATASET_PATH="/share/home/luzhixuan/datasets/SEED"
    BATCH_SIZE=48
    EXTRA_ARGS=(-sessions 1)
    ;;
  seediv)
    DATASET="seediv_de_lds"
    SETTING="seediv_sub_independent_train_val_test_setting"
    DATASET_PATH="/share/home/luzhixuan/datasets/SEED_IV"
    BATCH_SIZE=48
    EXTRA_ARGS=(-sessions 1)
    ;;
  deap)
    DATASET="deap"
    SETTING="deap_sub_independent_train_val_test_setting"
    DATASET_PATH="/share/home/luzhixuan/datasets/DEAP/data_preprocessed_python"
    BATCH_SIZE=2048
    EXTRA_ARGS=(-feature_type de_lds -time_window 1 -sample_length 1 -stride 1 -bounds 5 5 -label_used valence)
    ;;
  *)
    echo "Unknown dataset key: ${DATASET_KEY}" >&2
    exit 1
    ;;
esac

RUN_ROOT="/share/home/luzhixuan/runs/pcl_tdgc_libeer_3datasets_3seeds/${DATASET_KEY}/${MODEL}/seed_${SEED}"
mkdir -p "${RUN_ROOT}/result" "${RUN_ROOT}/log"

echo "Running ${MODEL} dataset=${DATASET_KEY} seed=${SEED} epochs=${EPOCHS} batch_size=${BATCH_SIZE}"
echo "Output root: ${RUN_ROOT}"

python PCL_TDGCN_train.py \
  -device cuda \
  -metrics acc macro-f1 \
  -metric_choose macro-f1 \
  -model "${MODEL}" \
  -seed "${SEED}" \
  -batch_size "${BATCH_SIZE}" \
  -epochs "${EPOCHS}" \
  -lr "${LR}" \
  -num_workers 0 \
  -setting "${SETTING}" \
  -dataset "${DATASET}" \
  -dataset_path "${DATASET_PATH}" \
  -output_dir "${RUN_ROOT}/result" \
  -log_dir "${RUN_ROOT}/log" \
  "${EXTRA_ARGS[@]}"

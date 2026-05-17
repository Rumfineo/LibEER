#!/bin/bash
#SBATCH --job-name=pcl_libeer
#SBATCH --partition=gpu
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err

set -euo pipefail

SEEDS=(2024 2025)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
RUN_ROOT="/share/home/luzhixuan/runs/pcl_tdgc_libeer_seed_ind_notime/seed_${SEED}"

mkdir -p logs "${RUN_ROOT}/result" "${RUN_ROOT}/log"

module purge
module load conda/23.3_with_py3.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate libeer

cd /share/home/luzhixuan/projects/LibEER/LibEER

module list
which python
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"

echo "Running PCL_TDGCN LibEER benchmark seed=${SEED}"

python PCL_TDGCN_train.py \
  -device cuda \
  -metrics acc macro-f1 \
  -metric_choose macro-f1 \
  -model PCL_TDGCN \
  -seed "${SEED}" \
  -batch_size 48 \
  -epochs 1000 \
  -lr 0.001 \
  -num_workers 0 \
  -setting seed_sub_independent_train_val_test_setting \
  -dataset seed_de_lds \
  -dataset_path /share/home/luzhixuan/datasets/SEED \
  -sessions 1 \
  -output_dir "${RUN_ROOT}/result" \
  -log_dir "${RUN_ROOT}/log"

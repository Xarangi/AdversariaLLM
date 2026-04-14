#!/bin/bash
#SBATCH --job-name=sr-backfill-qwen3-llama3
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn-b[001-005],cn-e00[2-3]
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/mila/s/sarangis/AdversariaLLM/slurm-sr-backfill-qwen3-llama3-%A_%a.out
#SBATCH --error=/home/mila/s/sarangis/AdversariaLLM/slurm-sr-backfill-qwen3-llama3-%A_%a.err

set -euo pipefail

cd /home/mila/s/sarangis/AdversariaLLM
export HF_HOME=/network/scratch/s/sarangis/hf-cache
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PYTHONUNBUFFERED=1

combos=(
  "2026-04-12 autodan"
  "2026-04-12 crescendo"
  "2026-04-12 gcg"
  "2026-04-12 pair"
  "2026-04-12 pgd"
)

combo="${combos[$SLURM_ARRAY_TASK_ID]}"
read -r RUN_DATE ATTACK <<< "$combo"

CACHE_PATH="outputs/${RUN_DATE}/scoring/cache_strong_reject_qwen3_llama3_${ATTACK}.json"

mkdir -p "outputs/${RUN_DATE}/scoring"

echo "[backfill] date=${RUN_DATE} attack=${ATTACK} cache=${CACHE_PATH}"
.venv/bin/python scripts/score_missing_strong_reject.py \
  --date "${RUN_DATE}" \
  --attack "${ATTACK}" \
  --judge strong_reject \
  --cache-path "${CACHE_PATH}" \
  --batch-size 128 \
  --passes 3 \
  --save-every-batches 2

#!/bin/bash
#SBATCH --job-name=sr-backfill-qwen3-llama3
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:4
#SBATCH --exclude=cn-b[001-005],cn-e00[2-3]
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/mila/s/sarangis/AdversariaLLM/slurm-sr-backfill-qwen3-llama3-%j.out
#SBATCH --error=/home/mila/s/sarangis/AdversariaLLM/slurm-sr-backfill-qwen3-llama3-%j.err

set -euo pipefail

cd /home/mila/s/sarangis/AdversariaLLM
export HF_HOME=/network/scratch/s/sarangis/hf-cache
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PYTHONUNBUFFERED=1

RUN_DATE="${RUN_DATE:-2026-04-13}"
NUM_GPUS="${NUM_GPUS:-4}"
ATTACKS=(autodan crescendo gcg pair pgd)

if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer, got: $NUM_GPUS" >&2
  exit 1
fi

AVAILABLE_GPUS="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
if [[ -n "$AVAILABLE_GPUS" ]] && [[ "$AVAILABLE_GPUS" -gt 0 ]] && [[ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]]; then
  echo "[backfill] reducing NUM_GPUS from ${NUM_GPUS} to available ${AVAILABLE_GPUS}" >&2
  NUM_GPUS="$AVAILABLE_GPUS"
fi

mkdir -p "outputs/${RUN_DATE}/scoring"

for ((start=0; start<${#ATTACKS[@]}; start+=NUM_GPUS)); do
  pids=()
  labels=()
  for ((slot=0; slot<NUM_GPUS; slot++)); do
    idx=$((start + slot))
    if (( idx >= ${#ATTACKS[@]} )); then
      break
    fi

    attack="${ATTACKS[$idx]}"
    cache_path="outputs/${RUN_DATE}/scoring/cache_strong_reject_qwen3_llama3_${attack}.json"

    echo "[backfill] launching date=${RUN_DATE} attack=${attack} gpu=${slot} cache=${cache_path}"
    CUDA_VISIBLE_DEVICES="${slot}" .venv/bin/python scripts/score_missing_strong_reject.py \
      --date "${RUN_DATE}" \
      --attack "${attack}" \
      --judge strong_reject \
      --cache-path "${cache_path}" \
      --batch-size 128 \
      --passes 3 \
      --save-every-batches 2 &

    pids+=("$!")
    labels+=("${attack}@gpu${slot}")
  done

  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[backfill] worker failed: ${labels[$i]}" >&2
      exit 1
    fi
  done
done

echo "[backfill] completed all attacks for ${RUN_DATE}"

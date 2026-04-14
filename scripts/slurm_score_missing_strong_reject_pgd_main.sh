#!/bin/bash
#SBATCH --job-name=pgd-llama3-sr
#SBATCH --partition=main
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/mila/s/sarangis/AdversariaLLM/slurm-pgd-strongreject-%j.out
#SBATCH --error=/home/mila/s/sarangis/AdversariaLLM/slurm-pgd-strongreject-%j.err

set -euo pipefail

cd /home/mila/s/sarangis/AdversariaLLM
export HF_HOME=/network/scratch/s/sarangis/hf-cache
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

.venv/bin/python scripts/score_missing_strong_reject.py \
  --date 2026-04-13 \
  --attack pgd \
  --judge strong_reject \
  --cache-path outputs/2026-04-13/scoring/cache_strong_reject_pgd.json \
  --batch-size 128 \
  --passes 3 \
  --save-every-batches 2

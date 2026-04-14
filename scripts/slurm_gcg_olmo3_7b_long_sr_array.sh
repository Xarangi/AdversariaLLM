#!/bin/bash
#SBATCH --job-name=gcg-olmo3-7b-sr
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-299
#SBATCH --output=/home/mila/s/sarangis/AdversariaLLM/slurm-gcg-olmo3-7b-sr-%A_%a.out
#SBATCH --error=/home/mila/s/sarangis/AdversariaLLM/slurm-gcg-olmo3-7b-sr-%A_%a.err

set -euo pipefail

cd /home/mila/s/sarangis/AdversariaLLM
export HF_HOME=/network/scratch/s/sarangis/hf-cache
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PYTHONUNBUFFERED=1

MONGOD_BIN=/home/mila/s/sarangis/AdversariaLLM/local/mongodb/bin/mongod
MONGO_PORT=$((27017 + SLURM_ARRAY_TASK_ID))
MONGO_DBPATH="/tmp/${USER}/adversariallm-mongo/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
MONGO_LOGPATH="${MONGO_DBPATH}/mongod.log"

if [ ! -x "$MONGOD_BIN" ]; then
  echo "mongod binary not found at $MONGOD_BIN" >&2
  exit 1
fi

mkdir -p "$MONGO_DBPATH"
"$MONGOD_BIN" --dbpath "$MONGO_DBPATH" --bind_ip 127.0.0.1 --port "$MONGO_PORT" --logpath "$MONGO_LOGPATH" --fork

cleanup() {
  "$MONGOD_BIN" --dbpath "$MONGO_DBPATH" --shutdown >/dev/null 2>&1 || true
}
trap cleanup EXIT

export MONGODB_URI="mongodb://127.0.0.1:${MONGO_PORT}"
export MONGODB_DB="adversariallm"

.venv/bin/python run_attacks.py \
  root_dir=/home/mila/s/sarangis/AdversariaLLM \
  model=olmo3-7b-instruct-local \
  dataset=adv_behaviors \
  datasets.adv_behaviors.idx=${SLURM_ARRAY_TASK_ID} \
  attack=gcg \
  classifiers='[strong_reject]'

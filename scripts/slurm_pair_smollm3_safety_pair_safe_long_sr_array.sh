#!/bin/bash
#SBATCH --job-name=pair-smollm3-safety-pair-safe
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn-a[001-011],cn-b[001-005],cn-c[002-014,016-019,021-034,036-040],cn-e00[2-3]
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-99
#SBATCH --output=/home/mila/s/sarangis/AdversariaLLM/slurm-pair-smollm3-safety-pair-safe-%A_%a.out
#SBATCH --error=/home/mila/s/sarangis/AdversariaLLM/slurm-pair-smollm3-safety-pair-safe-%A_%a.err

set -euo pipefail

cd /home/mila/s/sarangis/AdversariaLLM
export HF_HOME=/network/scratch/s/sarangis/hf-cache
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PYTHONUNBUFFERED=1

MONGOD_BIN=/home/mila/s/sarangis/AdversariaLLM/local/mongodb/bin/mongod
MONGO_PORT=$((31017 + SLURM_ARRAY_TASK_ID))
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
  model=tvergara/smollm3-safety-pair-safe \
  dataset=adv_behaviors \
  datasets.adv_behaviors.idx=${SLURM_ARRAY_TASK_ID} \
  attack=pair \
  classifiers='[strong_reject]'

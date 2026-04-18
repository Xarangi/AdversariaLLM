#!/bin/bash
#SBATCH --job-name=pgd-olmo3-7b-sr
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn-a[001-011],cn-b[001-005],cn-c[002-014,016-019,021-034,036-040],cn-e00[2-3]
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-99
#SBATCH --output=/home/mila/s/sarangis/AdversariaLLM/slurm-pgd-olmo3-7b-sr-%A_%a.out
#SBATCH --error=/home/mila/s/sarangis/AdversariaLLM/slurm-pgd-olmo3-7b-sr-%A_%a.err

set -euo pipefail

cd /home/mila/s/sarangis/AdversariaLLM
export HF_HOME=/network/scratch/s/sarangis/hf-cache
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PYTHONUNBUFFERED=1

SCRATCH_RUN_BASE=/network/scratch/s/sarangis/AdversariaLLM
SAVE_DIR="${SCRATCH_RUN_BASE}/outputs"
EMBED_DIR="${SCRATCH_RUN_BASE}/embeddings/pgd/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${SAVE_DIR}" "${EMBED_DIR}"

MONGOD_BIN=/home/mila/s/sarangis/AdversariaLLM/local/mongodb/bin/mongod
MONGO_PORT=$((33017 + SLURM_ARRAY_TASK_ID))
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
  save_dir="${SAVE_DIR}" \
  embed_dir="${EMBED_DIR}" \
  model=olmo3-7b-instruct-local \
  dataset=adv_behaviors \
  datasets.adv_behaviors.idx=${SLURM_ARRAY_TASK_ID} \
  attack=pgd \
  classifiers='[strong_reject]'

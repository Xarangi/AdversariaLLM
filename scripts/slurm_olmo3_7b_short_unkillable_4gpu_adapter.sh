#!/bin/bash
#SBATCH --job-name=olmo3-7b-short4-adapter
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/home/mila/s/sarangis/AdversariaLLM/slurm-%x-%j.out
#SBATCH --error=/home/mila/s/sarangis/AdversariaLLM/slurm-%x-%j.err

set -euo pipefail

cd /home/mila/s/sarangis/AdversariaLLM
export HF_HOME=/network/scratch/s/sarangis/hf-cache
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PYTHONUNBUFFERED=1

ATTACK="${ATTACK:-pgd}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-99}"
NUM_GPUS="${NUM_GPUS:-4}"

case "${ATTACK}" in
  pgd|crescendo)
    ;;
  *)
    echo "ATTACK must be one of: pgd, crescendo (got ${ATTACK})" >&2
    exit 1
    ;;
esac

if ! [[ "${START_IDX}" =~ ^[0-9]+$ ]] || ! [[ "${END_IDX}" =~ ^[0-9]+$ ]]; then
  echo "START_IDX and END_IDX must be non-negative integers." >&2
  exit 1
fi

if (( START_IDX > END_IDX )); then
  echo "START_IDX (${START_IDX}) cannot be greater than END_IDX (${END_IDX})." >&2
  exit 1
fi

if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer (got ${NUM_GPUS})." >&2
  exit 1
fi

if (( NUM_GPUS > 4 )); then
  echo "NUM_GPUS cannot exceed 4 for this adapter (got ${NUM_GPUS})." >&2
  exit 1
fi

AVAILABLE_GPUS="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
if [[ -n "${AVAILABLE_GPUS}" ]] && [[ "${AVAILABLE_GPUS}" =~ ^[0-9]+$ ]] && (( AVAILABLE_GPUS > 0 )) && (( NUM_GPUS > AVAILABLE_GPUS )); then
  echo "Reducing NUM_GPUS from ${NUM_GPUS} to available ${AVAILABLE_GPUS}." >&2
  NUM_GPUS="${AVAILABLE_GPUS}"
fi

MONGOD_BIN=/home/mila/s/sarangis/AdversariaLLM/local/mongodb/bin/mongod
if [[ ! -x "${MONGOD_BIN}" ]]; then
  echo "mongod binary not found at ${MONGOD_BIN}" >&2
  exit 1
fi

SCRATCH_RUN_BASE=/network/scratch/s/sarangis/AdversariaLLM
SAVE_DIR="${SCRATCH_RUN_BASE}/outputs"
EMBED_BASE="${SCRATCH_RUN_BASE}/embeddings/${ATTACK}-olmo3-7b-short4"
mkdir -p "${SAVE_DIR}" "${EMBED_BASE}"

PORT_BASE=$((40000 + (SLURM_JOB_ID % 1000) * 10))

run_one() {
  local idx="$1"
  local gpu_slot="$2"
  local mongo_port=$((PORT_BASE + gpu_slot))
  local mongo_dbpath="/tmp/${USER}/adversariallm-mongo/${SLURM_JOB_ID}_${ATTACK}_${idx}"
  local mongo_logpath="${mongo_dbpath}/mongod.log"
  local embed_dir="${EMBED_BASE}/${SLURM_JOB_ID}_${idx}"

  mkdir -p "${mongo_dbpath}" "${embed_dir}"

  "${MONGOD_BIN}" --dbpath "${mongo_dbpath}" --bind_ip 127.0.0.1 --port "${mongo_port}" --logpath "${mongo_logpath}" --fork

  local rc=0
  MONGODB_URI="mongodb://127.0.0.1:${mongo_port}" \
  MONGODB_DB="adversariallm" \
  CUDA_VISIBLE_DEVICES="${gpu_slot}" \
  .venv/bin/python run_attacks.py \
    root_dir=/home/mila/s/sarangis/AdversariaLLM \
    save_dir="${SAVE_DIR}" \
    embed_dir="${embed_dir}" \
    model=olmo3-7b-instruct-local \
    dataset=adv_behaviors \
    datasets.adv_behaviors.idx="${idx}" \
    attack="${ATTACK}" \
    classifiers='[strong_reject]' || rc=$?

  "${MONGOD_BIN}" --dbpath "${mongo_dbpath}" --shutdown >/dev/null 2>&1 || true
  return "${rc}"
}

echo "Launching ATTACK=${ATTACK} on indices ${START_IDX}-${END_IDX} with ${NUM_GPUS} workers."

for ((base=START_IDX; base<=END_IDX; base+=NUM_GPUS)); do
  pids=()
  labels=()

  for ((slot=0; slot<NUM_GPUS; slot++)); do
    idx=$((base + slot))
    if (( idx > END_IDX )); then
      break
    fi

    echo "[launch] attack=${ATTACK} idx=${idx} gpu=${slot}"
    run_one "${idx}" "${slot}" &
    pids+=("$!")
    labels+=("idx=${idx}@gpu=${slot}")
  done

  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[error] worker failed: ${labels[$i]}" >&2
      exit 1
    fi
  done
done

echo "Completed ATTACK=${ATTACK} for indices ${START_IDX}-${END_IDX}."

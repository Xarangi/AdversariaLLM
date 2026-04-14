# Submission Guide

This guide records how the current Slurm submissions in this workspace are wired, what the jobs use at runtime, and which changes were made to make them stable.

## What gets submitted

The repository currently uses one Slurm array script per attack/model combination.

- [scripts/slurm_gcg_olmo3_7b_long_sr_array.sh](scripts/slurm_gcg_olmo3_7b_long_sr_array.sh) submits GCG runs for OLMo-3-7B.
- [scripts/slurm_autodan_olmo3_7b_long_sr_array.sh](scripts/slurm_autodan_olmo3_7b_long_sr_array.sh) submits AutoDAN runs for OLMo-3-7B.
- [scripts/slurm_pair_olmo3_7b_long_sr_array.sh](scripts/slurm_pair_olmo3_7b_long_sr_array.sh) submits PAIR runs for OLMo-3-7B.
- [scripts/slurm_gcg_smollm3_safety_pair_safe_long_sr_array.sh](scripts/slurm_gcg_smollm3_safety_pair_safe_long_sr_array.sh) submits GCG runs for the cached `tvergara/smollm3-safety-pair-safe` model.
- [scripts/slurm_gcg_smollm3_safety_pair_unsafe_long_sr_array.sh](scripts/slurm_gcg_smollm3_safety_pair_unsafe_long_sr_array.sh) submits GCG runs for the cached `tvergara/smollm3-safety-pair-unsafe` model.

## Required Slurm fields

Each array script uses the same basic scheduler settings.

- `--partition=long` because these are long-running attack sweeps.
- `--cpus-per-task=2` because the run is GPU-bound and only needs modest CPU support.
- `--mem=32G` to hold the model, optimizer state, and logging overhead.
- `--time=7-00:00:00` to give the full sweep enough wall time.
- `--array=0-299` so each array element maps to one HarmBench behavior index.
- `--output` and `--error` point to per-task log files in the repository root so stdout and stderr stay separated by array index.

The model selection is done in the `run_attacks.py` call itself, not in the Slurm header.

## Runtime environment

The scripts now standardize the runtime environment before launching Python.

- `HF_HOME=/network/scratch/s/sarangis/hf-cache` keeps Hugging Face downloads off the home filesystem.
- `HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub` and `TRANSFORMERS_CACHE=${HF_HOME}/hub` point the hub and transformers caches at scratch.
- `PYTHONUNBUFFERED=1` makes task logs stream immediately instead of buffering.

For the OLMo-3 runs, the scripts also request GPU resources in a way that avoids the known V100 compatibility problem.

- GCG uses `--gres=gpu:l40s:1`.
- AutoDAN and PAIR use `--gres=gpu:1` with `--exclude=cn-b[001-005],cn-e00[2-3]` so they can land on l-nodes and other non-V100 GPUs but avoid the unsupported V100 nodes.

## MongoDB setup for judging

The judge pipeline expects MongoDB metadata for duplicate filtering and scoring bookkeeping, so each task starts its own local MongoDB instance.

What the scripts do:

- Use the portable binary at [local/mongodb/bin/mongod](local/mongodb/bin/mongod).
- Create a private dbpath under `/tmp/${USER}/adversariallm-mongo/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}`.
- Bind MongoDB to `127.0.0.1` only.
- Assign a unique port per array element.
- Export `MONGODB_URI=mongodb://127.0.0.1:<port>` and `MONGODB_DB=adversariallm`.
- Register a shell trap so the local server is shut down on exit.

This is the important difference from the earlier failed runs: the jobs no longer depend on an external MongoDB service being present in the environment.

## Model registry changes

The model registry lives in [conf/models/models.yaml](conf/models/models.yaml).

Two new local entries were added there for the cached SmolLM3 pair models:

- `tvergara/smollm3-safety-pair-safe` points at `/network/scratch/s/sarangis/hf-cache/hub/models--tvergara--smollm3-safety-pair-safe/snapshots/b545184189f8e3aebb6806b9653c7e284357faa8`.
- `tvergara/smollm3-safety-pair-unsafe` points at `/network/scratch/s/sarangis/hf-cache/hub/models--tvergara--smollm3-safety-pair-unsafe/snapshots/b52fcb11c956ba0c1922b0dcb16f2cae81d278c5`.

Both entries use the same schema as the existing model registry:

- `id`
- `tokenizer_id`
- `short_name`
- `developer_name`
- `compile`
- `dtype`
- `chat_template`
- `trust_remote_code`

## Which attacks use which models

The current OLMo-3 jobs are wired as follows:

- GCG on `olmo3-7b-instruct-local` with `attack=gcg`.
- AutoDAN on `olmo3-7b-instruct-local` with `attack=autodan`.
- PAIR on `olmo3-7b-instruct-local` with `attack=pair`.
- All three use `dataset=adv_behaviors` and `classifiers='[strong_reject]'`.
- Each array index maps directly to `datasets.adv_behaviors.idx=${SLURM_ARRAY_TASK_ID}`.

The new SmolLM3 GCG jobs are wired as follows:

- GCG safe uses `model=tvergara/smollm3-safety-pair-safe`.
- GCG unsafe uses `model=tvergara/smollm3-safety-pair-unsafe`.
- Both use the same `adv_behaviors` dataset and `strong_reject` classifier.

## StrongReject judging details

The `strong_reject` classifier is not a simple standalone model load. It resolves to a LoRA adapter on top of Gemma.

- The adapter files are cached under `/network/scratch/s/sarangis/hf-cache/hub/models--qylu4156--strongreject-15k-v1`.
- The adapter config points at `google/gemma-2b` as its base model.
- The local workspace already has a cache symlink for the Gemma base so the judge can load offline.

That is why the per-task MongoDB setup and the HF cache path matter: attack output generation and judge scoring both depend on them.

## How to submit

Submit the array scripts with `sbatch` from the repository root.

Examples:

```bash
sbatch scripts/slurm_gcg_olmo3_7b_long_sr_array.sh
sbatch scripts/slurm_autodan_olmo3_7b_long_sr_array.sh
sbatch scripts/slurm_pair_olmo3_7b_long_sr_array.sh
sbatch scripts/slurm_gcg_smollm3_safety_pair_safe_long_sr_array.sh
sbatch scripts/slurm_gcg_smollm3_safety_pair_unsafe_long_sr_array.sh
```

Each submission launches 300 array elements, one per behavior index.

## How to inspect progress

Check scheduler state with `squeue` and job metadata with `scontrol show job`.

- Current log files are named `slurm-<attack>-<model>-<jobid>_<array>.out` and `.err`.
- The array output directory is the repository root, so you can tail logs directly in place.
- If a task is slow or stalled, inspect the matching `.out` file first and then the `.err` file.

Useful commands:

```bash
squeue -h -j <jobid> -o '%i %t %j %M %R'
scontrol show job <jobid>_0
tail -f slurm-gcg-olmo3-7b-sr-<jobid>_0.out
```

## Changes already made in this workspace

- Added local MongoDB startup and shutdown logic to the OLMo-3 array scripts.
- Standardized HF cache usage on scratch.
- Installed a portable MongoDB binary under [local/mongodb](local/mongodb).
- Preloaded `lmsys/vicuna-13b-v1.5` into scratch cache.
- Preloaded `tvergara/smollm3-safety-pair-safe` into scratch cache.
- Preloaded `tvergara/smollm3-safety-pair-unsafe` into scratch cache.
- Cached the `qylu4156/strongreject-15k-v1` adapter files.
- Added an offline Gemma cache link so StrongReject can resolve its base model.
- Narrowed PAIR and AutoDAN away from V100 nodes while still allowing l-nodes and other non-V100 GPUs.

## Practical note

The SmolLM3 pair request in the chat repeated the safe model name twice. The repository now has both safe and unsafe model entries available, so the next submission step can target either one explicitly without further registry changes.
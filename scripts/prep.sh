#!/usr/bin/env bash
# Offline stages for one steering target, after generate.sh has written
# <run_dir>/raw/<ds>.jsonl.
#
# Usage: scripts/prep.sh <run_dir> <target_model> <patch_layers> <device>
#   e.g. scripts/prep.sh runs/qwen3_8b Qwen/Qwen3-8B 18-35 cuda:0
#   DS=aqua scripts/prep.sh ...      # run a subset of datasets
#
# Stages:
#   1. segment          sophia.build_segmented (discourse-marker steps, is_correct)
#   2. embed            sophia.embed_steps (Qwen3-4B-Base mean-pool features)
#   3. full-data K=5    sophia.cluster_full (z-score, KMeans seed 42, n_init 10)
#   4. residual cache   sophia.extract_states on the target, patch-layer range
#   5. full-data vecs   sophia.full_vectors (reference for stability checks)
#   6. cross-fit build  sophia.crossfit_build (5 folds, n_init 50)
set -euo pipefail
cd "$(dirname "$0")/.."
run="$1"; model="$2"; layers="$3"; dev="$4"
DS="${DS:-gsm8k,aqua,logiqa,math}"
PY="${PYTHON:-python}"
# On hosts with more cores than the OpenBLAS build supports (64), K-means
# aborts with "double free or corruption" unless the thread count is capped.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}" \
       MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
log() { echo "[$(date '+%F %T')] [$run] $*"; }

for ds in ${DS//,/ }; do
  [ -f "$run/raw/$ds.jsonl" ] || { echo "missing $run/raw/$ds.jsonl (run scripts/generate.sh first)"; exit 1; }
done

log "1/6 segment"
$PY -m sophia.build_segmented --input-dir "$run/raw" --output-dir "$run/segmented"

log "2/6 embed (Qwen3-4B-Base) on $dev"
$PY -m sophia.embed_steps --datasets "$DS" --input-dir "$run/segmented" --out-root "$run/work" --device "$dev"

log "3/6 full-data clustering"
$PY -m sophia.cluster_full --segmented-dir "$run/segmented" --embed-root "$run/work" \
  --out-dir "$run/clustered" --artifact-dir "$run/cluster_artifacts" --datasets "$DS"

log "4/6 residual cache ($model, layers $layers) on $dev"
$PY -m sophia.extract_states --model "$model" --layers "$layers" --clustered-dir "$run/clustered" \
  --out-dir "$run/step_states" --datasets "$DS" --device "$dev"

log "5/6 full-data steering vectors"
for ds in ${DS//,/ }; do
  $PY -m sophia.full_vectors --cache-dir "$run/step_states" --clustered-dir "$run/clustered" \
    --out "$run/full_vectors/steering_vectors_$ds.pt" --datasets "$ds"
done

log "6/6 cross-fit build"
$PY -m sophia.crossfit_build --datasets "$DS" --clustered-dir "$run/clustered" \
  --embed-root "$run/work" --cache-dir "$run/step_states" --full-vectors "$run/full_vectors" \
  --out-root "$run/crossfit" --n-init 50
log "PREP DONE"

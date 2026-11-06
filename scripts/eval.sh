#!/usr/bin/env bash
# Held-out SOPHIA evaluation for one steering target (five conditions).
#
# Usage: scripts/eval.sh <run_dir> <target_model> <patch_layers> <thinking_device> <base_device> <ds> [<ds> ...]
#   e.g. scripts/eval.sh runs/qwen3_8b Qwen/Qwen3-8B 18-35 cuda:0 cuda:1 gsm8k aqua logiqa math
#   STAYER_RULE=next_same scripts/eval.sh ...   # robustness variant (default prev_same)
#
# The steering target and the Qwen3-4B-Base re-clusterer can share one GPU or
# be split across two. On shared GPUs a crash (usually OOM) is retried after
# a pause, up to MAX_ATTEMPTS times; the eval resumes from rows already written.
set -uo pipefail
cd "$(dirname "$0")/.."
run="$1"; model="$2"; layers="$3"; dev_t="$4"; dev_b="$5"; shift 5
rule="${STAYER_RULE:-prev_same}"
PY="${PYTHON:-python}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for ds in "$@"; do
  [ -f "$run/crossfit/$ds/build_summary.json" ] || { echo "missing $run/crossfit/$ds (run scripts/prep.sh first)"; exit 1; }
  ok=0
  for attempt in $(seq 1 "${MAX_ATTEMPTS:-30}"); do
    echo "[$(date '+%F %T')] start $run/$ds ($rule) on $dev_t/$dev_b (attempt $attempt)"
    if $PY -u -m sophia.crossfit_eval --dataset "$ds" --stayer-rule "$rule" \
         --work-dir "$run/crossfit" --clustered-dir "$run/clustered" \
         --target-model "$model" --patch-layers "$layers" \
         --device-thinking "$dev_t" --device-base "$dev_b"; then
      ok=1
      echo "[$(date '+%F %T')] done $run/$ds"
      break
    fi
    echo "[$(date '+%F %T')] FAILED $run/$ds attempt $attempt; retrying in 120s"
    sleep 120
  done
  [ "$ok" = 1 ] || { echo "giving up on $run/$ds"; exit 1; }
done
echo "[$(date '+%F %T')] EVAL QUEUE FINISHED"

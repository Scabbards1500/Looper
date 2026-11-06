#!/usr/bin/env bash
# Generate 500 chain-of-thought traces per dataset for one steering target.
#
# Usage: scripts/generate.sh <gpu> <model> <run_dir> [datasets] [gpu_mem_util]
#   e.g. scripts/generate.sh 0 Qwen/Qwen3-8B runs/qwen3_8b
#        scripts/generate.sh 1 Qwen/Qwen3-8B runs/qwen3_8b logiqa,aqua 0.6
#
# vLLM usually lives in its own environment: point VLLM_PYTHON at its python.
# Datasets already written to <run_dir>/raw are skipped, so the four datasets
# can be split across GPUs by running this script once per GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
gpu="$1"; model="$2"; run="$3"; ds="${4:-gsm8k,aqua,logiqa,math}"; util="${5:-0.85}"
CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_DISABLE_PROGRESS_BARS=1 \
  "${VLLM_PYTHON:-python}" -m sophia.gen_traces \
  --model "$model" --datasets "$ds" --gpu-mem-util "$util" --out-dir "$run/raw"

#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
asr_dir="$(cd "${script_dir}/.." && pwd)"
cd "${asr_dir}"

checkpoint_path="$1"
gpu_id="${2:-0}"
work_root="${3:-./paper_benchmark_runs}"

if [[ ! -f "${checkpoint_path}" ]]; then
  echo "Checkpoint not found: ${checkpoint_path}"
  exit 1
fi

mkdir -p "${work_root}"
exp_dir="${work_root}/exp_from_checkpoint"
mkdir -p "${exp_dir}"

target_ckpt="${exp_dir}/epoch-1.pt"
ln -sfn "$(realpath "${checkpoint_path}")" "${target_ckpt}"

CUDA_VISIBLE_DEVICES="${gpu_id}" \
DECODE_DIR="${work_root}/decode" \
"${script_dir}/decode_paper_benchmark.sh" 1 1 "${exp_dir}" "${gpu_id}"

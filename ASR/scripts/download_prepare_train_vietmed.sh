#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
asr_dir="$(cd "${script_dir}/.." && pwd)"
repo_dir="$(cd "${asr_dir}/.." && pwd)"
cd "${repo_dir}"

vietmed_output_dir="${VIETMED_OUTPUT_DIR:-${asr_dir}/download}"
vietmed_cache_dir="${VIETMED_CACHE_DIR:-}"
prepare_stage="${PREPARE_STAGE:-1}"
prepare_stop_stage="${PREPARE_STOP_STAGE:-3}"

download_args=(
  --output-dir "${vietmed_output_dir}"
)

if [[ -n "${vietmed_cache_dir}" ]]; then
  download_args+=(--cache-dir "${vietmed_cache_dir}")
fi

python3 "${asr_dir}/local/download_vietmed_hf.py" "${download_args[@]}"

(
  cd "${asr_dir}"
  STAGE="${prepare_stage}" STOP_STAGE="${prepare_stop_stage}" ./scripts/prepare_vietmed.sh
  ./scripts/train_vietmed.sh
)

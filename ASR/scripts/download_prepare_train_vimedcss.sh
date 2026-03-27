#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
asr_dir="$(cd "${script_dir}/.." && pwd)"
repo_dir="$(cd "${asr_dir}/.." && pwd)"
cd "${repo_dir}"

vimedcss_output_dir="${VIMEDCSS_OUTPUT_DIR:-${asr_dir}/download_vimedcss}"
vimedcss_cache_dir="${VIMEDCSS_CACHE_DIR:-}"
prepare_stage="${PREPARE_STAGE:-1}"
prepare_stop_stage="${PREPARE_STOP_STAGE:-3}"

download_args=(
  --output-dir "${vimedcss_output_dir}"
)

if [[ -n "${vimedcss_cache_dir}" ]]; then
  download_args+=(--cache-dir "${vimedcss_cache_dir}")
fi

python3 "${asr_dir}/local/download_vimedcss_hf.py" "${download_args[@]}"

(
  cd "${asr_dir}"
  STAGE="${prepare_stage}" STOP_STAGE="${prepare_stop_stage}" bash ./scripts/prepare_vimedcss.sh
  bash ./scripts/train_vimedcss.sh
)

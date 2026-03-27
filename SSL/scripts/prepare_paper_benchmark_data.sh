#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
benchmark_root="${BENCHMARK_ROOT:-${repo_root}/SSL/download/paper_benchmark}"
manifest_dir="${PAPER_MANIFEST_DIR:-${repo_root}/SSL/data/paper_manifests}"
fbank_dir="${PAPER_FBANK_DIR:-${repo_root}/SSL/data/fbank}"

gigaspeech2_src="${GIGASPEECH2_SRC:-}"
commonvoice_src="${COMMONVOICE_SRC:-}"

if [[ -z "${gigaspeech2_src}" || -z "${commonvoice_src}" ]]; then
  echo "Please set GIGASPEECH2_SRC and COMMONVOICE_SRC before running this script."
  exit 1
fi

python3 "${repo_root}/SSL/local/prepare_paper_test_sets.py" \
  gigaspeech2 \
  --src-dir "${gigaspeech2_src}" \
  --output-dir "${benchmark_root}/gigaspeech2"

python3 "${repo_root}/SSL/local/prepare_paper_test_sets.py" \
  commonvoice \
  --src-dir "${commonvoice_src}" \
  --output-dir "${benchmark_root}/commonvoice"

python3 "${repo_root}/SSL/local/prepare_paper_test_sets.py" \
  fleurs \
  --output-dir "${benchmark_root}/fleurs"

python3 "${repo_root}/ASR/local/prepare_manifest.py" \
  --corpus-dir "${benchmark_root}" \
  --output-dir "${manifest_dir}" \
  --language vietnamese \
  --dataset-parts "gigaspeech2 commonvoice fleurs"

python3 "${repo_root}/ASR/local/compute_fbank.py" \
  --manifest-dir "${manifest_dir}" \
  --output-dir "${fbank_dir}" \
  --dataset "gigaspeech2 commonvoice fleurs"

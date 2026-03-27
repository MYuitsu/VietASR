#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${4:-0}"

epoch="$1"
avg="$2"
exp_dir="$3"

bpe_model="${BPE_MODEL:-data/lang_bpe_2000/bpe.model}"
decode_dir="${DECODE_DIR:-${exp_dir}/paper-benchmark}"

gigaspeech2_cuts="${GIGASPEECH2_CUTS:-}"
commonvoice_cuts="${COMMONVOICE_CUTS:-}"
fleurs_cuts="${FLEURS_CUTS:-}"

if [[ -z "${gigaspeech2_cuts}" || -z "${commonvoice_cuts}" || -z "${fleurs_cuts}" ]]; then
  echo "Please set GIGASPEECH2_CUTS, COMMONVOICE_CUTS, and FLEURS_CUTS."
  exit 1
fi

run_decode() {
  local test_name="$1"
  local cuts_path="$2"

  python ./zipformer/decode.py \
    --epoch "${epoch}" \
    --avg "${avg}" \
    --exp-dir "${exp_dir}" \
    --decode-dir "${decode_dir}" \
    --max-duration 1000 \
    --bpe-model "${bpe_model}" \
    --decoding-method modified_beam_search \
    --beam-size 4 \
    --use-averaged-model 0 \
    --cuts-name "${test_name}" \
    --cuts-path "${cuts_path}"
}

run_decode "gigaspeech2" "${gigaspeech2_cuts}"
run_decode "commonvoice" "${commonvoice_cuts}"
run_decode "fleurs" "${fleurs_cuts}"

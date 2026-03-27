#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

export PYTHONPATH="${repo_root}:${repo_root}/ASR/zipformer:${repo_root}/icefall:${PYTHONPATH:-}"
export VIETASR_REPO_ROOT="${repo_root}"
export VIETASR_CHECKPOINT="${VIETASR_CHECKPOINT:-${repo_root}/hf_models/viet_iter3_pseudo_label/exp/epoch-12.pt}"
export VIETASR_TOKENS="${VIETASR_TOKENS:-${repo_root}/hf_models/viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/tokens.txt}"

decode_dir="${repo_root}/ASR/paper_benchmark_runs/hf_viet_iter3_decode"
res_dir="${decode_dir}/modified_beam_search"
timestamp="$(date +%Y%m%d-%H%M%S)"
detail_dir="${res_dir}/fleurs_detail_${timestamp}"
mkdir -p "${detail_dir}"

python3 ASR/zipformer/decode.py \
  --epoch 12 \
  --avg 1 \
  --exp-dir ASR/paper_benchmark_runs/hf_viet_iter3_exp \
  --decode-dir ASR/paper_benchmark_runs/hf_viet_iter3_decode \
  --max-duration 1000 \
  --bpe-model "${repo_root}/hf_models/viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model" \
  --decoding-method modified_beam_search \
  --beam-size 4 \
  --use-averaged-model 0 \
  --cuts-name fleurs \
  --cuts-path SSL/data/fbank/vietASR_cuts_fleurs.jsonl.gz \
  2>&1 | tee "${detail_dir}/decode.log"

recogs_path="${res_dir}/recogs-fleurs-beam_size_4-epoch-12-avg-1-modified_beam_search-beam-size-4.txt"
errs_path="${res_dir}/errs-fleurs-beam_size_4-epoch-12-avg-1-modified_beam_search-beam-size-4.txt"
summary_path="${res_dir}/wer-summary-fleurs-beam_size_4-epoch-12-avg-1-modified_beam_search-beam-size-4.txt"

python3 ASR/local/export_decode_details.py \
  --cuts-path SSL/data/fbank/vietASR_cuts_fleurs.jsonl.gz \
  --recogs-path "${recogs_path}" \
  --output-tsv "${detail_dir}/fleurs_decode_details.tsv" \
  --output-jsonl "${detail_dir}/fleurs_decode_details.jsonl"

cp "${summary_path}" "${detail_dir}/wer-summary-fleurs.txt"
cp "${errs_path}" "${detail_dir}/errs-fleurs.txt"
cp "${recogs_path}" "${detail_dir}/recogs-fleurs.txt"

echo "Detail report written to ${detail_dir}"

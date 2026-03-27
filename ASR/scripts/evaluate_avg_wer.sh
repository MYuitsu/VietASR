#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
asr_dir="$(cd "${script_dir}/.." && pwd)"
repo_dir="$(cd "${asr_dir}/.." && pwd)"
cd "${repo_dir}"

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <version-name|exp-dir> <epoch> <avg-values> [gpu-id]"
  echo 'Example:'
  echo '  $0 vietmed_v1 12 "1 3 5 7" 0'
  exit 1
fi

version_or_exp_dir="$1"
epoch="$2"
avg_values_str="$3"
gpu_id="${4:-0}"

registry_path="${CHECKPOINT_REGISTRY:-${repo_dir}/ASR/checkpoint_versions.json}"
bpe_model="${BPE_MODEL:-data/lang_bpe_2000/bpe.model}"
gigaspeech2_cuts="${GIGASPEECH2_CUTS:-}"
commonvoice_cuts="${COMMONVOICE_CUTS:-}"
fleurs_cuts="${FLEURS_CUTS:-}"

if [[ -z "${gigaspeech2_cuts}" || -z "${commonvoice_cuts}" || -z "${fleurs_cuts}" ]]; then
  echo "Please set GIGASPEECH2_CUTS, COMMONVOICE_CUTS, and FLEURS_CUTS."
  exit 1
fi

if [ -d "$version_or_exp_dir" ]; then
  exp_dir="$(realpath "$version_or_exp_dir")"
  version_name="$(basename "$exp_dir")"
else
  exp_dir="$(python3 "${repo_dir}/ASR/local/manage_checkpoint_versions.py" --registry "$registry_path" resolve --name "$version_or_exp_dir")"
  version_name="$version_or_exp_dir"
fi

if [ ! -d "$exp_dir" ]; then
  echo "Experiment directory not found: $exp_dir"
  exit 1
fi

decode_root="${DECODE_ROOT:-${exp_dir}/avg_wer_eval/${version_name}/epoch-${epoch}}"
mkdir -p "$decode_root"

for avg in $avg_values_str; do
  echo "Running benchmark for version=${version_name}, epoch=${epoch}, avg=${avg}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  BPE_MODEL="${bpe_model}" \
  DECODE_DIR="${decode_root}" \
  GIGASPEECH2_CUTS="${gigaspeech2_cuts}" \
  COMMONVOICE_CUTS="${commonvoice_cuts}" \
  FLEURS_CUTS="${fleurs_cuts}" \
  "${repo_dir}/ASR/scripts/decode_paper_benchmark.sh" "${epoch}" "${avg}" "${exp_dir}" "${gpu_id}"
done

output_tsv="${decode_root}/avg-wer-results.tsv"
python3 "${repo_dir}/ASR/local/collect_avg_wer_results.py" \
  --decode-dir "${decode_root}" \
  --epoch "${epoch}" \
  --avg-values ${avg_values_str} \
  --gigaspeech2-cuts "${gigaspeech2_cuts}" \
  --commonvoice-cuts "${commonvoice_cuts}" \
  --fleurs-cuts "${fleurs_cuts}" \
  --output-tsv "${output_tsv}"

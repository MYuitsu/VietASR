#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH="$PWD/../icefall${PYTHONPATH:+:$PYTHONPATH}"

nj="${NJ:-15}"
stage="${STAGE:-1}"
stop_stage="${STOP_STAGE:-3}"
dl_dir="${DL_DIR:-$PWD/download}"
manifest_dir="${MANIFEST_DIR:-data_vietmed/manifests}"
fbank_dir="${FBANK_DIR:-data_vietmed/fbank}"
lang_dir="${LANG_DIR:-data_vietmed/lang_bpe_1000}"
dataset_parts="${DATASET_PARTS:-dev test VietMed}"

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p "$manifest_dir" "$fbank_dir" "$lang_dir"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare VietMed manifests"
  python3 local/prepare_manifest.py \
    --num-jobs "$nj" \
    --corpus-dir "$dl_dir" \
    --output-dir "$manifest_dir" \
    --language vietnamese \
    --dataset-parts "$dataset_parts"
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Compute VietMed fbank"
  python3 local/compute_fbank.py \
    --manifest-dir "$manifest_dir" \
    --output-dir "$fbank_dir" \
    --dataset "$dataset_parts" \
    --use-executor false
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Train VietMed BPE"
  transcript_path="$lang_dir/transcript_words.txt"
  : > "$transcript_path"
  find -L "$dl_dir/VietMed" -name "*.trans.txt" -print0 | while IFS= read -r -d '' f; do
    cut -d " " -f 2- "$f" >> "$transcript_path"
  done

  python3 local/train_bpe_model.py \
    --lang-dir "$lang_dir" \
    --vocab-size 1000 \
    --transcript "$transcript_path"
fi

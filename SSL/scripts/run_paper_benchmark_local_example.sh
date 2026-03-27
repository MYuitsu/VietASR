#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH="/home/nguyenthaiduy277/VietASR/icefall:${PYTHONPATH:-}"

# Fill these two paths with your local datasets if available.
export GIGASPEECH2_SRC="${GIGASPEECH2_SRC:-/path/to/GigaSpeech2}"
export COMMONVOICE_SRC="${COMMONVOICE_SRC:-/path/to/common_voice_17_0_vi}"

# Real checkpoint found on this machine.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/nguyenthaiduy277/VietASR/ASR/viet_iter3_pseudo_label/exp/epoch-12.pt}"
GPU_ID="${GPU_ID:-0}"

cd /home/nguyenthaiduy277/VietASR-1/SSL

./scripts/prepare_paper_benchmark_data.sh

export GIGASPEECH2_CUTS="../data/fbank/vietASR_cuts_gigaspeech2.jsonl.gz"
export COMMONVOICE_CUTS="../data/fbank/vietASR_cuts_commonvoice.jsonl.gz"
export FLEURS_CUTS="../data/fbank/vietASR_cuts_fleurs.jsonl.gz"

./scripts/run_paper_benchmark_from_checkpoint.sh "${CHECKPOINT_PATH}" "${GPU_ID}"

#!/bin/bash
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
asr_dir="$(cd "${script_dir}/.." && pwd)"
cd "${asr_dir}"

export CUDA_VISIBLE_DEVICES=$3
export PYTHONPATH="$PWD/../icefall${PYTHONPATH:+:$PYTHONPATH}"

python3 ./zipformer/decode.py \
    --epoch $1 \
    --avg $2 \
    --exp-dir zipformer/exp \
    --max-duration 1000 \
    --bpe-model data/lang_bpe_2000/bpe.model \
    --decoding-method greedy_search \
    --manifest-dir data/fbank \
    --use-averaged-model 1 \
    --cuts-name test # specify the cut to decode

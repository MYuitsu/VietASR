#! /usr/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
asr_dir="$(cd "${script_dir}/.." && pwd)"
cd "${asr_dir}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$PWD/../icefall${PYTHONPATH:+:$PYTHONPATH}"

python3 zipformer/train.py \
    --world-size 4 \
    --num-epochs 300 \
    --start-epoch 1 \
    --use-fp16 1 \
    --train-cuts 50h \
    --manifest-dir data_vietmed/fbank \
    --train-dataset-parts "VietMed" \
    --dev-dataset-part "dev" \
    --test-dataset-part "test" \
    --bpe-model data_vietmed/lang_bpe_1000/bpe.model \
    --max-duration 1000 \
    --enable-musan 0 \
    --exp-dir zipformer/exp-vietmed \
    --enable-spec-aug 1 \
    --seed 1332 \
    --master-port 12356

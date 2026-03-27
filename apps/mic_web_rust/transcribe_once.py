#!/usr/bin/env python3

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

import k2
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence


def add_repo_paths(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "ASR/zipformer"))
    sys.path.insert(0, str(repo_root / "icefall"))


def read_sound_file(filename: str, expected_sample_rate: int) -> torch.Tensor:
    wave, sample_rate = torchaudio.load(filename)
    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"expected sample rate {expected_sample_rate}, got {sample_rate}"
        )
    return wave[0].contiguous()


def token_ids_to_words(token_ids: List[int], token_table: k2.SymbolTable) -> str:
    text = ""
    for idx in token_ids:
        text += token_table[idx]
    return text.replace("▁", " ").strip()


def compute_fbank(wave: torch.Tensor, sample_rate: int, num_bins: int) -> torch.Tensor:
    return kaldi.fbank(
        wave.unsqueeze(0),
        num_mel_bins=num_bins,
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=sample_rate,
        snip_edges=False,
        high_freq=-400.0,
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--sound-file", type=Path, required=True)
    parser.add_argument("--method", type=str, default="modified_beam_search")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    add_repo_paths(args.repo_root)

    from beam_search import greedy_search_batch, modified_beam_search
    from export import num_tokens
    from train import get_model, get_params, get_parser as get_train_parser

    params = get_params()
    parser_defaults = vars(get_train_parser().parse_args([]))
    params.update(parser_defaults)
    params.update(
        {
            "checkpoint": str(args.checkpoint),
            "tokens": str(args.tokens),
            "method": args.method,
            "beam_size": args.beam_size,
            "sample_rate": args.sample_rate,
            "sound_files": [str(args.sound_file)],
        }
    )

    token_table = k2.SymbolTable.from_file(str(args.tokens))
    params.blank_id = token_table["<blk>"]
    params.unk_id = token_table["<unk>"]
    params.vocab_size = num_tokens(token_table) + 1

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    model = get_model(params)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()

    wave = read_sound_file(str(args.sound_file), args.sample_rate)
    features = [compute_fbank(wave, args.sample_rate, params.feature_dim).to(device)]
    feature_lengths = torch.tensor([features[0].size(0)], device=device)
    features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))

    encoder_out, encoder_out_lens = model.forward_encoder(features, feature_lengths)

    if args.method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=args.beam_size,
        )
    elif args.method == "greedy_search":
        hyp_tokens = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    transcript = token_ids_to_words(hyp_tokens[0], token_table)
    print(json.dumps({"transcript": transcript}, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

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


def normalize_wave(wave: torch.Tensor) -> torch.Tensor:
    peak = wave.abs().max().item()
    if peak <= 0:
        return wave
    target = 0.85
    scale = min(target / peak, 32.0)
    return (wave * scale).clamp(-1.0, 1.0)


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


def segment_wave(
    wave: torch.Tensor,
    sample_rate: int,
    frame_ms: int = 30,
    min_speech_ms: int = 220,
    min_silence_ms: int = 180,
) -> List[torch.Tensor]:
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    num_frames = math.ceil(wave.numel() / frame_len)
    rms_values = []
    for i in range(num_frames):
        start = i * frame_len
        end = min((i + 1) * frame_len, wave.numel())
        chunk = wave[start:end]
        rms_values.append(float(torch.sqrt(torch.mean(chunk * chunk) + 1e-12)))

    if not rms_values:
        return [wave]

    max_rms = max(rms_values)
    threshold = max(max_rms * 0.08, 0.0015)
    min_speech_frames = max(1, round(min_speech_ms / frame_ms))
    min_silence_frames = max(1, round(min_silence_ms / frame_ms))

    segments: List[Tuple[int, int]] = []
    speech_start = None
    silence_run = 0

    for idx, rms in enumerate(rms_values):
        is_speech = rms >= threshold
        if is_speech:
            if speech_start is None:
                speech_start = idx
            silence_run = 0
            continue

        if speech_start is not None:
            silence_run += 1
            if silence_run >= min_silence_frames:
                speech_end = idx - silence_run + 1
                if speech_end - speech_start >= min_speech_frames:
                    segments.append((speech_start, speech_end))
                speech_start = None
                silence_run = 0

    if speech_start is not None:
        speech_end = len(rms_values)
        if speech_end - speech_start >= min_speech_frames:
            segments.append((speech_start, speech_end))

    if not segments:
        return [wave]

    merged: List[Tuple[int, int]] = []
    for start, end in segments:
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        gap = start - prev_end
        if gap <= min_silence_frames:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    out = []
    pad = int(sample_rate * 0.08)
    for start_f, end_f in merged:
        start = max(0, start_f * frame_len - pad)
        end = min(wave.numel(), end_f * frame_len + pad)
        out.append(wave[start:end].contiguous())
    return out


def strip_fillers(text: str) -> str:
    fillers = {
        "ỪM",
        "Ừ",
        "Ờ",
        "À",
        "ÀM",
        "UM",
        "UH",
    }
    words = text.strip().split()
    while words and words[0] in fillers:
        words.pop(0)
    return " ".join(words)


def decode_wave(
    wave: torch.Tensor,
    sample_rate: int,
    params,
    model,
    token_table,
    method: str,
    beam_size: int,
    device: torch.device,
) -> str:
    features = [compute_fbank(wave, sample_rate, params.feature_dim).to(device)]
    feature_lengths = torch.tensor([features[0].size(0)], device=device)
    features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))

    from beam_search import greedy_search_batch, modified_beam_search

    encoder_out, encoder_out_lens = model.forward_encoder(features, feature_lengths)

    if method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=beam_size,
        )
    elif method == "greedy_search":
        hyp_tokens = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    return token_ids_to_words(hyp_tokens[0], token_table)


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

    wave = normalize_wave(read_sound_file(str(args.sound_file), args.sample_rate))
    segments = segment_wave(wave, args.sample_rate)
    parts = []
    for segment in segments:
        if segment.numel() < int(args.sample_rate * 0.18):
            continue
        text = decode_wave(
            segment,
            args.sample_rate,
            params,
            model,
            token_table,
            args.method,
            args.beam_size,
            device,
        )
        text = strip_fillers(text)
        if text:
            parts.append(text)

    transcript = " ".join(parts).strip()
    if not transcript:
        transcript = strip_fillers(
            decode_wave(
                wave,
                args.sample_rate,
                params,
                model,
                token_table,
                "greedy_search",
                args.beam_size,
                device,
            )
        ).strip()
    print(json.dumps({"transcript": transcript}, ensure_ascii=False))


if __name__ == "__main__":
    main()

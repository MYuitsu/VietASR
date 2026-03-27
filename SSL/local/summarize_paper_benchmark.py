#!/usr/bin/env python3

import argparse
from pathlib import Path

from lhotse import load_manifest_lazy


def read_best_wer(summary_path: Path) -> float:
    with open(summary_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        raise ValueError(f"No WER entries found in {summary_path}")

    _, wer = lines[1].split()
    return float(wer)


def count_ref_words(cuts_path: Path) -> int:
    cuts = load_manifest_lazy(cuts_path)
    total = 0
    for cut in cuts:
        for supervision in cut.supervisions:
            total += len(supervision.text.strip().split())
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Summarize the VietASR paper benchmark with weighted average WER."
    )
    parser.add_argument("--gigaspeech2-cuts", type=Path, required=True)
    parser.add_argument("--gigaspeech2-summary", type=Path, required=True)
    parser.add_argument("--commonvoice-cuts", type=Path, required=True)
    parser.add_argument("--commonvoice-summary", type=Path, required=True)
    parser.add_argument("--fleurs-cuts", type=Path, required=True)
    parser.add_argument("--fleurs-summary", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        ("GigaSpeech 2", args.gigaspeech2_cuts, args.gigaspeech2_summary),
        ("Common Voice", args.commonvoice_cuts, args.commonvoice_summary),
        ("FLEURS", args.fleurs_cuts, args.fleurs_summary),
    ]

    weighted_errors = 0.0
    total_words = 0

    print("Dataset\tWords\tBest WER")
    for name, cuts_path, summary_path in rows:
        words = count_ref_words(cuts_path)
        wer = read_best_wer(summary_path)
        weighted_errors += words * wer
        total_words += words
        print(f"{name}\t{words}\t{wer:.2f}")

    avg = weighted_errors / total_words if total_words > 0 else 0.0
    print(f"Avg\t{total_words}\t{avg:.2f}")


if __name__ == "__main__":
    main()

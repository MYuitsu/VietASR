#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from lhotse import load_manifest_lazy


DATASET_NAMES = ("gigaspeech2", "commonvoice", "fleurs")


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


def find_summary(res_dir: Path, dataset_name: str, epoch: int, avg: int) -> Path:
    pattern = f"wer-summary-{dataset_name}-*-epoch-{epoch}-avg-{avg}-*.txt"
    matches = sorted(res_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No summary file found for dataset={dataset_name}, epoch={epoch}, avg={avg} "
            f"under {res_dir} with pattern {pattern}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple summary files found for dataset={dataset_name}, epoch={epoch}, avg={avg}: "
            + ", ".join(str(item) for item in matches)
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect weighted average WER for a sweep of avg values."
    )
    parser.add_argument("--decode-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--avg-values", type=int, nargs="+", required=True)
    parser.add_argument("--gigaspeech2-cuts", type=Path, required=True)
    parser.add_argument("--commonvoice-cuts", type=Path, required=True)
    parser.add_argument("--fleurs-cuts", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    args = parser.parse_args()

    res_dir = args.decode_dir / "modified_beam_search"
    cut_paths: Dict[str, Path] = {
        "gigaspeech2": args.gigaspeech2_cuts,
        "commonvoice": args.commonvoice_cuts,
        "fleurs": args.fleurs_cuts,
    }
    word_counts = {name: count_ref_words(path) for name, path in cut_paths.items()}

    rows: List[Dict[str, object]] = []
    for avg in args.avg_values:
        weighted_errors = 0.0
        total_words = 0
        row: Dict[str, object] = {"epoch": args.epoch, "avg": avg}
        for dataset_name in DATASET_NAMES:
            summary_path = find_summary(res_dir, dataset_name, args.epoch, avg)
            wer = read_best_wer(summary_path)
            words = word_counts[dataset_name]
            row[f"{dataset_name}_wer"] = wer
            row[f"{dataset_name}_summary"] = str(summary_path)
            weighted_errors += words * wer
            total_words += words
        row["weighted_avg_wer"] = weighted_errors / total_words if total_words else 0.0
        rows.append(row)

    rows.sort(key=lambda item: item["weighted_avg_wer"])

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "avg",
        "weighted_avg_wer",
        "gigaspeech2_wer",
        "commonvoice_wer",
        "fleurs_wer",
        "gigaspeech2_summary",
        "commonvoice_summary",
        "fleurs_summary",
    ]
    with open(args.output_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print("epoch\tavg\tweighted_avg_wer\tgigaspeech2_wer\tcommonvoice_wer\tfleurs_wer")
    for row in rows:
        print(
            f"{row['epoch']}\t{row['avg']}\t{row['weighted_avg_wer']:.4f}\t"
            f"{row['gigaspeech2_wer']:.2f}\t{row['commonvoice_wer']:.2f}\t{row['fleurs_wer']:.2f}"
        )
    print(f"\nSaved TSV: {args.output_tsv}")


if __name__ == "__main__":
    main()

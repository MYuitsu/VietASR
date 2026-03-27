#!/usr/bin/env python3

import argparse
import ast
import csv
import json
from pathlib import Path

from lhotse import load_manifest_lazy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuts-path", type=Path, required=True)
    parser.add_argument("--recogs-path", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def parse_recogs(path: Path):
    entries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            utt_id, payload = line.split(":\t", maxsplit=1)
            kind, values = payload.split("=", maxsplit=1)
            values = ast.literal_eval(values)
            text = " ".join(values)
            entries.setdefault(utt_id, {})[kind] = text

    return entries


def main():
    args = parse_args()

    recogs = parse_recogs(args.recogs_path)

    rows = []
    for cut in load_manifest_lazy(args.cuts_path):
        utt_id = cut.id
        recog = recogs.get(utt_id, {})
        audio_path = Path(cut.recording.sources[0].source)

        rows.append(
            {
                "utt_id": utt_id,
                "file_name": audio_path.name,
                "audio_path": str(audio_path),
                "test_transcript": cut.supervisions[0].text,
                "vietasr_transcript": recog.get("hyp", ""),
                "ref_from_recogs": recog.get("ref", ""),
            }
        )

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "utt_id",
                "file_name",
                "audio_path",
                "test_transcript",
                "vietasr_transcript",
                "ref_from_recogs",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


AUDIO_COLUMN_CANDIDATES = (
    "audio",
    "audio_path",
    "path",
    "wav",
    "wav_path",
    "file",
    "file_path",
    "recording",
    "recording_path",
)
TEXT_COLUMN_CANDIDATES = (
    "text",
    "sentence",
    "transcript",
    "transcription",
    "normalized_text",
    "utterance",
    "content",
)
SPLIT_COLUMN_CANDIDATES = ("split", "subset", "partition", "set")
ID_COLUMN_CANDIDATES = ("utt_id", "utterance_id", "id", "uid", "segment_id")


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def normalize_text(text: str) -> str:
    text = " ".join(text.strip().split())
    return re.sub(r"\s+", " ", text)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_metadata_file(src_dir: Path) -> Path:
    candidates = []
    for pattern in ("*.csv", "*.tsv", "*.jsonl", "*.json"):
        candidates.extend(sorted(src_dir.glob(pattern)))
        candidates.extend(sorted(src_dir.glob(f"*/*{pattern.lstrip('*')}")))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find metadata file under {src_dir}. "
            "Please pass --metadata explicitly."
        )
    if len(candidates) > 1:
        logging.info("Found multiple metadata files, using the first one: %s", candidates[0])
    return candidates[0]


def sniff_delimiter(path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        return dialect.delimiter
    except csv.Error:
        return ","


def iter_jsonl(path: Path) -> Iterator[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {exc}") from exc
            if not isinstance(obj, dict):
                continue
            yield {str(k): "" if v is None else str(v) for k, v in obj.items()}


def iter_json(path: Path) -> Iterator[Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            data = data["data"]
        else:
            raise ValueError(
                f"JSON metadata {path} must be a list of objects or a dict with a 'data' list."
            )
    if not isinstance(data, list):
        raise ValueError(f"JSON metadata {path} must be a list of objects.")
    for item in data:
        if isinstance(item, dict):
            yield {str(k): "" if v is None else str(v) for k, v in item.items()}


def iter_table(path: Path) -> Iterator[Dict[str, str]]:
    delimiter = sniff_delimiter(path)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            yield {str(k): "" if v is None else str(v) for k, v in row.items()}


def load_rows(path: Path) -> Iterator[Dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return iter_jsonl(path)
    if suffix == ".json":
        return iter_json(path)
    return iter_table(path)


def pick_column(fieldnames: Iterable[str], candidates: Tuple[str, ...], user_value: Optional[str]) -> Optional[str]:
    names = list(fieldnames)
    lowered = {name.lower(): name for name in names if name}
    if user_value:
        if user_value in names:
            return user_value
        lookup = lowered.get(user_value.lower())
        if lookup:
            return lookup
        raise KeyError(f"Column {user_value!r} not found. Available columns: {names}")
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def resolve_audio_path(
    raw_value: str,
    metadata_path: Path,
    src_dir: Path,
    path_prefix: Optional[Path],
) -> Path:
    raw_value = raw_value.strip()
    if not raw_value:
        raise FileNotFoundError("Empty audio path in metadata row")

    candidates: List[Path] = []
    value_path = Path(raw_value)
    if value_path.is_absolute():
        candidates.append(value_path)
    else:
        candidates.append(metadata_path.parent / value_path)
        candidates.append(src_dir / value_path)
        if path_prefix is not None:
            candidates.append(path_prefix / value_path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve audio path {raw_value!r}. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def convert_to_wav(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def place_audio(src: Path, dst: Path, copy_audio: bool) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if src.suffix.lower() == ".wav":
        if copy_audio:
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)
        return
    convert_to_wav(src, dst)


def sanitize_utt_id(raw_id: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_.-]+", "_", raw_id.strip())
    normalized = normalized.strip("._-")
    return normalized or "utt"


def write_transcripts(rows_by_split: Dict[str, List[Tuple[str, str]]], output_dir: Path) -> None:
    for split, rows in rows_by_split.items():
        split_dir = output_dir / split
        ensure_dir(split_dir)
        transcript_path = split_dir / f"{split}.trans.txt"
        with open(transcript_path, "w", encoding="utf-8") as f:
            for utt_id, text in rows:
                f.write(f"{utt_id} {text}\n")
        logging.info("Wrote %s rows to %s", len(rows), transcript_path)


def normalize_dataset(args: argparse.Namespace) -> None:
    src_dir = Path(args.src_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    metadata_path = Path(args.metadata).resolve() if args.metadata else find_metadata_file(src_dir)
    path_prefix = Path(args.path_prefix).resolve() if args.path_prefix else None

    ensure_dir(output_dir)

    rows = list(load_rows(metadata_path))
    if not rows:
        raise ValueError(f"No usable rows found in metadata file: {metadata_path}")

    fieldnames = rows[0].keys()
    audio_column = pick_column(fieldnames, AUDIO_COLUMN_CANDIDATES, args.audio_column)
    text_column = pick_column(fieldnames, TEXT_COLUMN_CANDIDATES, args.text_column)
    split_column = pick_column(fieldnames, SPLIT_COLUMN_CANDIDATES, args.split_column)
    id_column = pick_column(fieldnames, ID_COLUMN_CANDIDATES, args.id_column)

    if audio_column is None:
        raise KeyError(f"Could not infer audio column from metadata columns: {list(fieldnames)}")
    if text_column is None:
        raise KeyError(f"Could not infer text column from metadata columns: {list(fieldnames)}")

    logging.info("Using metadata file: %s", metadata_path)
    logging.info("audio column: %s", audio_column)
    logging.info("text column: %s", text_column)
    logging.info("split column: %s", split_column if split_column else "<default>")
    logging.info("id column: %s", id_column if id_column else "<from audio stem>")

    rows_by_split: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    used_ids: Dict[str, int] = defaultdict(int)
    num_processed = 0

    for index, row in enumerate(rows):
        raw_audio = row.get(audio_column, "").strip()
        raw_text = row.get(text_column, "").strip()
        if not raw_audio or not raw_text:
            continue

        split = (
            normalize_text(row.get(split_column, "")) if split_column else args.default_split
        ) or args.default_split
        split = re.sub(r"[^0-9A-Za-z_.-]+", "_", split)

        src_audio = resolve_audio_path(raw_audio, metadata_path, src_dir, path_prefix)
        base_id = row.get(id_column, "").strip() if id_column else src_audio.stem
        if not base_id:
            base_id = src_audio.stem or f"vietmed_{index:08d}"
        utt_id = sanitize_utt_id(base_id)
        used_ids[utt_id] += 1
        if used_ids[utt_id] > 1:
            utt_id = f"{utt_id}_{used_ids[utt_id]:06d}"

        dst_audio = output_dir / split / f"{utt_id}.wav"
        place_audio(src_audio, dst_audio, copy_audio=args.copy_audio)
        rows_by_split[split].append((utt_id, normalize_text(raw_text)))
        num_processed += 1

    if num_processed == 0:
        raise ValueError("No valid samples were processed from metadata")

    write_transcripts(rows_by_split, output_dir)
    logging.info("Normalized %s samples into %s", num_processed, output_dir)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize raw VietMed data into VietASR supervised format."
    )
    parser.add_argument("--src-dir", type=str, required=True, help="Path to raw VietMed root.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="download/VietMed",
        help="Output directory in VietASR supervised format.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Metadata file (.csv/.tsv/.jsonl/.json). If omitted, the script will try to auto-detect one.",
    )
    parser.add_argument("--audio-column", type=str, default=None, help="Audio path column name.")
    parser.add_argument("--text-column", type=str, default=None, help="Transcript column name.")
    parser.add_argument("--split-column", type=str, default=None, help="Dataset split column name.")
    parser.add_argument("--id-column", type=str, default=None, help="Utterance id column name.")
    parser.add_argument(
        "--default-split",
        type=str,
        default="train",
        help="Split name to use when metadata has no split column.",
    )
    parser.add_argument(
        "--path-prefix",
        type=str,
        default=None,
        help="Optional prefix prepended when audio paths in metadata are relative to another base dir.",
    )
    parser.add_argument(
        "--copy-audio",
        type=str2bool,
        default=False,
        help="If true, copy wav files instead of symlinking them.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    normalize_dataset(get_args())

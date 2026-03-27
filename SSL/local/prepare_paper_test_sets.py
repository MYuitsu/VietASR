#!/usr/bin/env python3

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from datasets import load_dataset
import pandas as pd
import soundfile as sf


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def convert_to_wav(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if src.suffix.lower() == ".wav":
        if not dst.exists():
            shutil.copy2(src, dst)
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


def write_audio_array(array, sampling_rate: int, dst: Path) -> None:
    ensure_dir(dst.parent)
    sf.write(dst, array, sampling_rate)


def write_transcripts(rows: Iterable[tuple[str, str]], transcript_path: Path) -> None:
    ensure_dir(transcript_path.parent)
    with open(transcript_path, "w", encoding="utf-8") as f:
        for utt_id, text in rows:
            f.write(f"{utt_id} {normalize_text(text)}\n")


def prepare_gigaspeech2(args: argparse.Namespace) -> None:
    if args.from_hf:
        out_audio_dir = Path(args.output_dir) / "audio"
        transcript_path = out_audio_dir / "gigaspeech2.trans.txt"
        dataset = load_dataset(
            "speechcolab/gigaspeech2",
            args.lang,
            split=args.split,
            trust_remote_code=True,
            token=True,
        )
        rows = []
        for idx, example in enumerate(dataset):
            text = _pick_fleurs_text(example)
            if text is None:
                text = example.get("text") or example.get("sentence")
            if text is None:
                continue

            audio = example.get("audio")
            if not isinstance(audio, dict):
                continue

            src_audio = Path(audio["path"])
            utt_id = src_audio.stem if src_audio.suffix else f"gigaspeech2_{idx:06d}"
            dst_audio = out_audio_dir / f"{utt_id}.wav"
            if src_audio.is_file():
                convert_to_wav(src_audio, dst_audio)
            else:
                write_audio_array(audio["array"], audio["sampling_rate"], dst_audio)
            rows.append((utt_id, text))

        write_transcripts(rows, transcript_path)
        return

    src_root = Path(args.src_dir)
    lang_root = src_root / "data" / args.lang
    tsv_path = lang_root / f"{args.split}.tsv"
    extracted_dir = lang_root / args.split
    out_audio_dir = Path(args.output_dir) / "audio"
    transcript_path = out_audio_dir / "gigaspeech2.trans.txt"

    if not tsv_path.is_file():
        raise FileNotFoundError(f"Missing transcript file: {tsv_path}")

    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for segment_id, text in reader:
            src_audio = extracted_dir / f"{segment_id}.wav"
            if not src_audio.is_file():
                matches = list(lang_root.rglob(f"{segment_id}.wav"))
                if not matches:
                    raise FileNotFoundError(
                        "Missing audio for segment_id "
                        f"{segment_id!r}. GigaSpeech2 official metadata and "
                        "Lhotse recipe require transcript IDs to match wav file "
                        "names directly; refusing to guess a transcript-audio "
                        "mapping from archive order."
                    )
                src_audio = matches[0]

            dst_audio = out_audio_dir / f"{segment_id}.wav"
            convert_to_wav(src_audio, dst_audio)
            rows.append((segment_id, text))

    write_transcripts(rows, transcript_path)


def prepare_commonvoice(args: argparse.Namespace) -> None:
    if args.from_hf:
        out_audio_dir = Path(args.output_dir) / "audio"
        transcript_path = out_audio_dir / "commonvoice.trans.txt"
        dataset = load_dataset(
            args.dataset_name,
            args.lang,
            split=args.split,
            trust_remote_code=True,
            token=True,
        )
        rows = []
        for idx, example in enumerate(dataset):
            text = example.get("sentence") or example.get("text")
            if text is None:
                continue

            audio = example.get("audio")
            if not isinstance(audio, dict):
                continue

            src_audio = Path(audio["path"])
            utt_id = src_audio.stem if src_audio.suffix else f"commonvoice_{idx:06d}"
            dst_audio = out_audio_dir / f"{utt_id}.wav"
            if src_audio.is_file():
                convert_to_wav(src_audio, dst_audio)
            else:
                write_audio_array(audio["array"], audio["sampling_rate"], dst_audio)
            rows.append((utt_id, text))

        write_transcripts(rows, transcript_path)
        return

    src_root = Path(args.src_dir)
    tsv_path = src_root / f"{args.split}.tsv"
    clips_dir = src_root / "clips"
    out_audio_dir = Path(args.output_dir) / "audio"
    transcript_path = out_audio_dir / "commonvoice.trans.txt"

    if not tsv_path.is_file():
        raise FileNotFoundError(f"Missing transcript file: {tsv_path}")
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"Missing clips dir: {clips_dir}")

    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rel_audio = row.get("path")
            text = row.get("sentence") or row.get("text")
            if not rel_audio or not text:
                continue

            src_audio = clips_dir / rel_audio
            utt_id = Path(rel_audio).stem
            dst_audio = out_audio_dir / f"{utt_id}.wav"
            convert_to_wav(src_audio, dst_audio)
            rows.append((utt_id, text))

    write_transcripts(rows, transcript_path)


def _pick_fleurs_text(example: dict) -> Optional[str]:
    for key in ("transcription", "raw_transcription", "sentence", "text"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def prepare_fleurs(args: argparse.Namespace) -> None:
    out_audio_dir = Path(args.output_dir) / "audio"
    transcript_path = out_audio_dir / "fleurs.trans.txt"

    try:
        dataset = load_dataset(
            "google/fleurs",
            args.config,
            split=args.split,
            trust_remote_code=True,
        )
        rows = []

        for idx, example in enumerate(dataset):
            audio = example["audio"]
            text = _pick_fleurs_text(example)
            if text is None:
                continue

            src_audio = Path(audio["path"])
            utt_id = src_audio.stem if src_audio.suffix else f"fleurs_{idx:06d}"
            dst_audio = out_audio_dir / f"{utt_id}.wav"
            if src_audio.is_file():
                convert_to_wav(src_audio, dst_audio)
            else:
                write_audio_array(audio["array"], audio["sampling_rate"], dst_audio)
            rows.append((utt_id, text))

        write_transcripts(rows, transcript_path)
        return
    except RuntimeError as ex:
        if "Dataset scripts are no longer supported" not in str(ex):
            raise

    cache_dir = Path(args.output_dir) / "_hf_fleurs_cache"
    ensure_dir(cache_dir)

    subprocess.run(
        [
            "hf",
            "download",
            "google/fleurs",
            "--repo-type",
            "dataset",
            "--include",
            f"{args.config}/fleurs-{args.split}-*",
            "--local-dir",
            str(cache_dir),
        ],
        check=True,
    )

    parquet_files = sorted(
        (cache_dir / args.config).glob(f"fleurs-{args.split}-*.parquet")
    )
    if not parquet_files:
        raise FileNotFoundError(
            f"Missing FLEURS parquet files in {(cache_dir / args.config)}"
        )

    frames = [pd.read_parquet(parquet_file) for parquet_file in parquet_files]
    df = pd.concat(frames, ignore_index=True)
    rows = []
    for idx, row in df.iterrows():
        text = (
            row.get("transcription")
            or row.get("raw_transcription")
            or row.get("sentence")
            or row.get("text")
        )
        audio = row.get("audio")
        if not text or not isinstance(audio, dict):
            continue

        audio_path = audio.get("path")
        if not audio_path:
            continue

        utt_id = str(row.get("id", f"fleurs_{idx:06d}"))
        src_audio = Path(audio_path)
        dst_audio = out_audio_dir / f"{utt_id}.wav"
        convert_to_wav(src_audio, dst_audio)
        rows.append((utt_id, text))

    write_transcripts(rows, transcript_path)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the three public test sets used in the VietASR paper."
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    giga = subparsers.add_parser("gigaspeech2")
    giga.add_argument("--src-dir", type=str, default="")
    giga.add_argument("--lang", type=str, default="vi")
    giga.add_argument("--split", type=str, default="test")
    giga.add_argument("--output-dir", type=str, required=True)
    giga.add_argument("--from-hf", action="store_true")

    cv = subparsers.add_parser("commonvoice")
    cv.add_argument("--src-dir", type=str, default="")
    cv.add_argument("--split", type=str, default="test")
    cv.add_argument("--output-dir", type=str, required=True)
    cv.add_argument("--lang", type=str, default="vi")
    cv.add_argument("--from-hf", action="store_true")
    cv.add_argument(
        "--dataset-name",
        type=str,
        default="fsicoli/common_voice_17_0",
    )

    fleurs = subparsers.add_parser("fleurs")
    fleurs.add_argument("--config", type=str, default="vi_vn")
    fleurs.add_argument("--split", type=str, default="test")
    fleurs.add_argument("--output-dir", type=str, required=True)

    args = parser.parse_args()

    if args.dataset == "gigaspeech2":
        prepare_gigaspeech2(args)
    elif args.dataset == "commonvoice":
        prepare_commonvoice(args)
    elif args.dataset == "fleurs":
        prepare_fleurs(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import io
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import soundfile as sf
from datasets import Audio, Dataset, load_dataset
from scipy.signal import resample_poly


DEFAULT_DATASET = "leduckhai/VietMed"
DEFAULT_SPLITS = ("train", "dev", "test")
DEFAULT_LOCAL_CACHE_ROOT = Path("/home/nguyenthaiduy277/.cache/huggingface/datasets/leduckhai___viet_med")
DEFAULT_OUTPUT_SPLIT_MAP = {
    "train": "VietMed",
    "dev": "dev",
    "test": "test",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def sanitize_utt_id(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip())
    value = value.strip("._-")
    return value or "utt"


def write_split(
    dataset_name: str,
    split_name: str,
    output_root: Path,
    cache_dir: Path | None,
    local_cache_root: Path | None,
    overwrite: bool,
    output_split_name: str,
) -> int:
    ds = load_split_dataset(
        dataset_name=dataset_name,
        split_name=split_name,
        cache_dir=cache_dir,
        local_cache_root=local_cache_root,
    )

    split_dir = output_root / output_split_name
    ensure_dir(split_dir)
    transcript_path = split_dir / f"{output_split_name}.trans.txt"

    transcript_rows: List[Tuple[str, str]] = []
    used_ids: Dict[str, int] = {}

    for index, item in enumerate(ds):
        audio = item["audio"]
        text = item.get("text")
        if not text:
            continue

        raw_id = (
            item.get("id")
            or item.get("utt_id")
            or item.get("audio_id")
            or Path(audio.get("path", "")).stem
            or f"{output_split_name}_{index:08d}"
        )
        utt_id = sanitize_utt_id(str(raw_id))
        used_ids[utt_id] = used_ids.get(utt_id, 0) + 1
        if used_ids[utt_id] > 1:
            utt_id = f"{utt_id}_{used_ids[utt_id]:06d}"

        wav_path = split_dir / f"{utt_id}.wav"
        if overwrite or not wav_path.is_file():
            write_audio_to_wav(audio, wav_path)

        transcript_rows.append((utt_id, normalize_text(str(text))))

    with open(transcript_path, "w", encoding="utf-8") as f:
        for utt_id, text in transcript_rows:
            f.write(f"{utt_id} {text}\n")

    logging.info(
        "Saved %s utterances from split=%s to %s",
        len(transcript_rows),
        split_name,
        split_dir,
    )
    return len(transcript_rows)


def write_audio_to_wav(audio: Dict[str, object], wav_path: Path) -> None:
    ensure_dir(wav_path.parent)
    audio_bytes = audio.get("bytes")
    audio_path = audio.get("path")

    if audio_bytes:
        array, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        array, sampling_rate = ensure_16khz(array, sampling_rate)
        sf.write(wav_path, array, sampling_rate)
        return

    if audio_path:
        array, sampling_rate = sf.read(str(audio_path), dtype="float32")
        array, sampling_rate = ensure_16khz(array, sampling_rate)
        sf.write(wav_path, array, sampling_rate)
        return

    raise ValueError("Audio field does not contain bytes or a path")


def ensure_16khz(array, sampling_rate: int):
    target_sr = 16000
    if sampling_rate == target_sr:
        return array, sampling_rate
    if array.ndim == 1:
        array = resample_poly(array, target_sr, sampling_rate)
    else:
        array = resample_poly(array, target_sr, sampling_rate, axis=0)
    return array.astype("float32"), target_sr


def load_split_dataset(
    dataset_name: str,
    split_name: str,
    cache_dir: Path | None,
    local_cache_root: Path | None,
) -> Dataset:
    try:
        logging.info("Loading %s split=%s from Hugging Face datasets", dataset_name, split_name)
        ds = load_dataset(
            dataset_name,
            split=split_name,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        return ds.cast_column("audio", Audio(decode=False))
    except Exception as exc:
        logging.warning("Falling back to local cached arrow for split=%s due to: %s", split_name, exc)
        if local_cache_root is None:
            raise
        arrow_path = find_local_arrow(local_cache_root, split_name)
        logging.info("Loading split=%s from local cache file %s", split_name, arrow_path)
        ds = Dataset.from_file(str(arrow_path))
        return ds.cast_column("audio", Audio(decode=False))


def find_local_arrow(local_cache_root: Path, split_name: str) -> Path:
    pattern = f"**/*-{split_name}.arrow"
    matches = sorted(local_cache_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find a cached arrow file for split={split_name} under {local_cache_root}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the labeled VietMed splits from Hugging Face and convert them to VietASR supervised format."
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument(
        "--splits",
        type=str,
        default=" ".join(DEFAULT_SPLITS),
        help="Space-separated dataset splits to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ASR/download"),
        help="Target ASR download root in VietASR supervised format.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("ASR/.cache/huggingface/datasets"),
        help="Hugging Face datasets cache dir. Defaults to a writable repo-local path.",
    )
    parser.add_argument(
        "--local-cache-root",
        type=Path,
        default=DEFAULT_LOCAL_CACHE_ROOT,
        help="Fallback Hugging Face datasets cache root to read local arrow files offline.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing wav/transcript files.",
    )
    args = parser.parse_args()
    args.dataset_name_for_loader = args.dataset

    total = 0
    for split_name in args.splits.split():
        output_split_name = DEFAULT_OUTPUT_SPLIT_MAP.get(split_name, split_name)
        total += write_split(
            dataset_name=args.dataset,
            split_name=split_name,
            output_root=args.output_dir,
            cache_dir=args.cache_dir,
            local_cache_root=args.local_cache_root,
            overwrite=args.overwrite,
            output_split_name=output_split_name,
        )
    logging.info("Finished downloading VietMed. Total utterances: %s", total)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    main()

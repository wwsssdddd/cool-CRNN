"""Подготовка train/validation TSV из папки, generic TSV или MJSynth list."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from .config import CHARS
from .dataset import SUPPORTED_EXTENSIONS


def _label_from_path(path: Path, source_format: str) -> str:
    stem = path.stem
    if source_format == "mjsynth":
        parts = stem.split("_")
        if len(parts) < 2:
            raise ValueError(f"MJSynth filename не содержит слово: {path.name}")
        return parts[1].lower()
    return stem.lower()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def collect_samples(
    root: Path,
    annotations: Path | None,
    source_format: str,
) -> list[tuple[Path, str]]:
    if annotations is None:
        paths = sorted(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        return [(path, _label_from_path(path, source_format)) for path in paths]

    samples: list[tuple[Path, str]] = []
    for line_number, raw_line in enumerate(annotations.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if source_format == "tsv":
            parts = raw_line.split("\t", maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Строка {line_number}: требуется path<TAB>label")
            raw_path, label = parts
            label = label.strip().lower()
        else:
            raw_path = raw_line.split()[0]
            label = ""
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = root / raw_path.lstrip("./")
        if source_format != "tsv":
            label = _label_from_path(path, source_format)
        samples.append((path, label))
    return samples


def validate_samples(
    samples: list[tuple[Path, str]],
    *,
    skip_invalid: bool,
) -> tuple[list[tuple[Path, str]], int]:
    valid: list[tuple[Path, str]] = []
    skipped = 0
    allowed = set(CHARS)
    for path, label in samples:
        reason = None
        if not path.is_file():
            reason = "файл не найден"
        elif not label:
            reason = "пустая метка"
        elif set(label) - allowed:
            reason = f"символы вне алфавита: {sorted(set(label) - allowed)}"
        if reason:
            if not skip_invalid:
                raise ValueError(f"{path}: {reason}")
            skipped += 1
            continue
        valid.append((path, label))
    return valid, skipped


def write_manifest(path: Path, samples: list[tuple[Path, str]], root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for image, label in samples:
            stream.write(f"{_relative(image, root)}\t{label}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Подготовить manifests для CRNN")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--format", choices=["filename", "mjsynth", "tsv"], default="filename")
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--val-output", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-invalid", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not 0 <= args.val_fraction < 1:
        raise ValueError("val-fraction должен быть в диапазоне [0, 1)")
    samples = collect_samples(root, args.annotations, args.format)
    samples, skipped = validate_samples(samples, skip_invalid=args.skip_invalid)
    random.Random(args.seed).shuffle(samples)

    val_count = 0
    if args.val_output and args.val_fraction > 0:
        val_count = max(1, round(len(samples) * args.val_fraction))
    validation = samples[:val_count]
    training = samples[val_count:]
    if not training:
        raise ValueError("После разбиения train manifest пуст")
    write_manifest(args.train_output, training, root)
    if args.val_output:
        write_manifest(args.val_output, validation, root)
    print(f"train={len(training)} validation={len(validation)} skipped={skipped}")


if __name__ == "__main__":
    main()

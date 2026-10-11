"""Подготовка train/validation TSV из папки, generic TSV или MJSynth list."""

from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
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

    if source_format == "icdar_xml":
        samples: list[tuple[Path, str]] = []
        xml_root = ET.parse(annotations).getroot()
        for index, element in enumerate(xml_root.findall(".//image"), 1):
            raw_path = element.get("file")
            raw_label = element.get("tag")
            if raw_path is None or raw_label is None:
                raise ValueError(
                    f"XML-элемент image #{index} должен содержать атрибуты file и tag"
                )

            relative_path = Path(raw_path)
            candidates = [root / relative_path]
            # В поставке ICDAR XML использует word/1/1.jpg, а изображения
            # находятся в выбранном root как 1/1.jpg.
            if relative_path.parts and relative_path.parts[0].lower() == "word":
                candidates.append(root / Path(*relative_path.parts[1:]))
            path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
            samples.append((path, raw_label.strip().lower()))
        if not samples:
            raise ValueError(f"В ICDAR XML нет элементов image: {annotations}")
        return samples

    samples = []
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


def write_mjsynth_manifest_streaming(
    root: Path,
    annotations: Path,
    output: Path,
    *,
    skip_invalid: bool,
    skip_file_check: bool,
) -> tuple[int, int]:
    """Преобразует официальный MJSynth annotation-файл без загрузки его в RAM."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    allowed = set(CHARS)
    written = 0
    skipped = 0

    try:
        with annotations.open("r", encoding="utf-8") as source, temporary.open(
            "w", encoding="utf-8"
        ) as destination:
            for line_number, raw_line in enumerate(source, 1):
                if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                    continue
                raw_path = raw_line.split(maxsplit=1)[0]
                relative_path = Path(raw_path.removeprefix("./"))
                path = root / relative_path
                label = _label_from_path(relative_path, "mjsynth")

                reason = None
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    reason = "путь выходит за пределы root"
                elif not skip_file_check and not path.is_file():
                    reason = "файл не найден"
                elif not label:
                    reason = "пустая метка"
                elif set(label) - allowed:
                    reason = f"символы вне алфавита: {sorted(set(label) - allowed)}"

                if reason:
                    if not skip_invalid:
                        raise ValueError(
                            f"Строка {line_number}, {path}: {reason}"
                        )
                    skipped += 1
                    continue
                destination.write(f"{relative_path.as_posix()}\t{label}\n")
                written += 1
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return written, skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Подготовить manifests для CRNN")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument(
        "--format",
        choices=["filename", "mjsynth", "tsv", "icdar_xml"],
        default="filename",
    )
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--val-output", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Потоковый режим для одного официального MJSynth annotation-файла",
    )
    parser.add_argument(
        "--skip-file-check",
        action="store_true",
        help="Не делать stat каждого изображения (только вместе с --stream)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if args.format in {"tsv", "icdar_xml"} and args.annotations is None:
        raise ValueError(f"Для формата {args.format} необходимо передать --annotations")
    if args.annotations is not None and not args.annotations.is_file():
        raise FileNotFoundError(args.annotations)
    if not 0 <= args.val_fraction < 1:
        raise ValueError("val-fraction должен быть в диапазоне [0, 1)")
    if args.skip_file_check and not args.stream:
        raise ValueError("--skip-file-check разрешён только вместе с --stream")
    if args.stream:
        if args.format != "mjsynth" or args.annotations is None:
            raise ValueError("--stream поддерживает только MJSynth с --annotations")
        if args.val_output is not None or args.val_fraction != 0:
            raise ValueError(
                "В --stream режиме используйте отдельные annotation-файлы и "
                "--val-fraction 0 без --val-output"
            )
        written, skipped = write_mjsynth_manifest_streaming(
            root,
            args.annotations,
            args.train_output,
            skip_invalid=args.skip_invalid,
            skip_file_check=args.skip_file_check,
        )
        print(f"written={written} skipped={skipped}")
        return
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

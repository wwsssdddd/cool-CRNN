"""Оценка IIIT5K/SVT/IC03/IC13 в lexicon-free и lexicon-based режимах."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from tqdm.auto import tqdm

from .checkpoint import load_model
from .config import CHARS, DEVICE
from .dataset import read_image
from .lexicon import LexiconDecoder, greedy_decode_log_probs, load_lexicon
from .metrics import char_error_rate, word_accuracy


@dataclass(frozen=True)
class BenchmarkRecord:
    image: Path
    label: str
    lexicons: dict[str, str | list[str]]


@dataclass
class InferenceRecord:
    source: BenchmarkRecord
    log_probs: torch.Tensor
    greedy: str
    inference_ms: float


def _resolve(path: str, root: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else root / value


def load_benchmark_manifest(path: str | Path, data_root: str | Path) -> list[BenchmarkRecord]:
    """Читает JSONL или TSV: image<TAB>label<TAB>optional_lexicon_path."""
    manifest = Path(path)
    root = Path(data_root).expanduser()
    records: list[BenchmarkRecord] = []
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if manifest.suffix.lower() == ".jsonl":
            payload = json.loads(raw_line)
            image = _resolve(str(payload["image"]), root)
            label = str(payload["label"]).strip().lower()
            lexicons = dict(payload.get("lexicons", {}))
        else:
            parts = raw_line.split("\t")
            if len(parts) not in {2, 3}:
                raise ValueError(f"Строка {line_number}: нужно 2 или 3 TSV-поля")
            image = _resolve(parts[0], root)
            label = parts[1].strip().lower()
            lexicons = {"default": parts[2]} if len(parts) == 3 else {}
        if not image.is_file():
            raise FileNotFoundError(f"Строка {line_number}: нет изображения {image}")
        unknown = set(label) - set(CHARS)
        if not label or unknown:
            raise ValueError(f"Строка {line_number}: некорректная метка {label!r}")
        records.append(BenchmarkRecord(image, label, lexicons))
    if not records:
        raise ValueError("Benchmark manifest пуст")
    return records


def run_inference(
    records: list[BenchmarkRecord],
    checkpoint: str | Path,
    device: str,
) -> tuple[list[InferenceRecord], int]:
    model = load_model(checkpoint, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    output: list[InferenceRecord] = []
    for record in tqdm(records, desc="inference"):
        image = read_image(record.image)
        width = torch.tensor([image.shape[-1]], dtype=torch.long)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(image.unsqueeze(0).to(device), width)
            valid_steps = int(model.output_lengths(width)[0])
            log_probs = logits[:valid_steps, 0].float().log_softmax(1).cpu()
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - started) * 1000
        output.append(InferenceRecord(record, log_probs, greedy_decode_log_probs(log_probs), elapsed))
    return output, parameter_count


def _summary(predictions: list[str], targets: list[str], times: list[float]) -> dict[str, float]:
    return {
        "word_accuracy": word_accuracy(predictions, targets),
        "cer": char_error_rate(predictions, targets),
        "mean_time_ms": mean(times) if times else 0.0,
    }


def _decoder_for_record(
    record: BenchmarkRecord,
    *,
    root: Path,
    global_decoder: LexiconDecoder | None,
    lexicon_key: str,
    cache: dict[str, LexiconDecoder],
) -> LexiconDecoder:
    if global_decoder is not None:
        return global_decoder
    if lexicon_key not in record.lexicons:
        raise KeyError(f"Для {record.image} нет лексикона {lexicon_key!r}")
    source = record.lexicons[lexicon_key]
    if isinstance(source, list):
        key = json.dumps(source, ensure_ascii=False)
        if key not in cache:
            cache[key] = LexiconDecoder(source)
        return cache[key]
    lexicon_path = _resolve(source, root)
    key = str(lexicon_path.resolve())
    if key not in cache:
        cache[key] = LexiconDecoder(load_lexicon(lexicon_path))
    return cache[key]


def evaluate(
    inference: list[InferenceRecord],
    *,
    data_root: str | Path,
    global_lexicon: str | Path | None,
    lexicon_key: str,
    deltas: list[int],
    exhaustive: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, list[dict[str, Any]]]]:
    targets = [item.source.label for item in inference]
    summaries: dict[str, dict[str, float]] = {}
    predictions_by_mode: dict[str, list[dict[str, Any]]] = {}

    greedy_predictions = [item.greedy for item in inference]
    summaries["none"] = _summary(
        greedy_predictions,
        targets,
        [item.inference_ms for item in inference],
    )
    predictions_by_mode["none"] = [
        {
            "image": str(item.source.image),
            "target": item.source.label,
            "prediction": item.greedy,
            "inference_ms": item.inference_ms,
        }
        for item in inference
    ]

    if not global_lexicon and not any(item.source.lexicons for item in inference):
        return summaries, predictions_by_mode

    root = Path(data_root).expanduser()
    global_decoder = LexiconDecoder(load_lexicon(global_lexicon)) if global_lexicon else None
    cache: dict[str, LexiconDecoder] = {}
    modes: list[tuple[str, int | None]] = [(f"delta_{delta}", delta) for delta in deltas]
    if exhaustive:
        modes.append(("lexicon_exhaustive", None))

    for mode_name, delta in modes:
        mode_predictions: list[str] = []
        lexicon_times: list[float] = []
        rows: list[dict[str, Any]] = []
        for item in tqdm(inference, desc=mode_name):
            decoder = _decoder_for_record(
                item.source,
                root=root,
                global_decoder=global_decoder,
                lexicon_key=lexicon_key,
                cache=cache,
            )
            decoded = decoder.decode(item.log_probs, delta=delta)
            mode_predictions.append(decoded.text)
            lexicon_times.append(decoded.search_time_ms)
            rows.append(
                {
                    "image": str(item.source.image),
                    "target": item.source.label,
                    "greedy": decoded.greedy_text,
                    "prediction": decoded.text,
                    "candidate_count": decoded.candidate_count,
                    "lexicon_time_ms": decoded.search_time_ms,
                }
            )
        summaries[mode_name] = _summary(mode_predictions, targets, lexicon_times)
        summaries[mode_name]["mean_candidates"] = mean(
            row["candidate_count"] for row in rows
        )
        predictions_by_mode[mode_name] = rows
    return summaries, predictions_by_mode


def plot_delta_results(summaries: dict[str, dict[str, float]], destination: Path) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(
        (
            (int(name.removeprefix("delta_")), values)
            for name, values in summaries.items()
            if name.startswith("delta_")
        ),
        key=lambda item: item[0],
    )
    if not rows:
        return
    deltas = [row[0] for row in rows]
    accuracies = [row[1]["word_accuracy"] * 100 for row in rows]
    times = [row[1]["mean_time_ms"] for row in rows]
    figure, accuracy_axis = plt.subplots(figsize=(8, 5))
    time_axis = accuracy_axis.twinx()
    accuracy_axis.plot(deltas, accuracies, marker="o", color="tab:blue")
    time_axis.bar(deltas, times, alpha=0.3, color="tab:red")
    accuracy_axis.set(xlabel="delta", ylabel="word accuracy, %", title="Lexicon search")
    time_axis.set_ylabel("mean search time, ms")
    accuracy_axis.grid(alpha=0.25)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CRNN benchmark evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--global-lexicon")
    parser.add_argument("--lexicon-key", default="default")
    parser.add_argument("--deltas", nargs="*", type=int, default=[])
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = load_benchmark_manifest(args.manifest, args.data_root)
    inference, parameter_count = run_inference(records, args.checkpoint, args.device)
    summaries, predictions = evaluate(
        inference,
        data_root=args.data_root,
        global_lexicon=args.global_lexicon,
        lexicon_key=args.lexicon_key,
        deltas=args.deltas,
        exhaustive=args.exhaustive,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "manifest": str(Path(args.manifest).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "samples": len(records),
        "parameter_count": parameter_count,
        "model_size_mb_fp32": parameter_count * 4 / 1_000_000,
        "modes": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for mode, rows in predictions.items():
        with (output_dir / f"predictions_{mode}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    plot_delta_results(summaries, output_dir / "figure4_delta.png")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

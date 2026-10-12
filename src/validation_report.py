"""Строит наглядный отчёт с хорошими и плохими validation-примерами."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .checkpoint import load_model
from .config import BLANK_IDX, DEVICE
from .dataset import CaptchaDataset, ctc_collate_fn, read_image
from .metrics import char_error_rate, levenshtein_distance, word_accuracy
from .train import compute_ctc_loss, decode_predictions, save_validation_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CRNN validation examples report")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--fixed-width", type=int, default=100)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--good-count", type=int, default=3)
    parser.add_argument("--bad-count", type=int, default=3)
    return parser


def _processed_uint8(path: Path, fixed_width: int) -> np.ndarray:
    tensor = read_image(path, fixed_width=fixed_width)
    return np.clip(tensor.squeeze(0).numpy() * 128.0 + 128.0, 0, 255).astype(np.uint8)


def _select_indices(
    predictions: list[str],
    targets: list[str],
    good_count: int,
    bad_count: int,
) -> tuple[list[int], list[int]]:
    correct = [index for index, pair in enumerate(zip(predictions, targets)) if pair[0] == pair[1]]
    correct.sort(key=lambda index: (-len(targets[index]), index))

    incorrect = [index for index, pair in enumerate(zip(predictions, targets)) if pair[0] != pair[1]]
    incorrect.sort(
        key=lambda index: (
            levenshtein_distance(predictions[index], targets[index]) / max(1, len(targets[index])),
            levenshtein_distance(predictions[index], targets[index]),
            len(targets[index]),
        ),
        reverse=True,
    )
    return correct[:good_count], incorrect[:bad_count]


def _render_report(
    destination: Path,
    paths: list[Path],
    predictions: list[str],
    targets: list[str],
    selected: list[tuple[str, int]],
    *,
    fixed_width: int,
    validation_loss: float,
    validation_cer: float,
    validation_word_accuracy: float,
    iteration: int,
) -> list[dict[str, object]]:
    rows = len(selected)
    figure, axes = plt.subplots(rows, 3, figsize=(15, max(3.0 * rows, 6.0)))
    axes = np.atleast_2d(axes)
    report_records: list[dict[str, object]] = []

    for row, (category, index) in enumerate(selected):
        path = paths[index]
        original = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise ValueError(f"Не удалось прочитать выбранное изображение: {path}")
        processed = _processed_uint8(path, fixed_width)
        prediction = predictions[index]
        target = targets[index]
        distance = levenshtein_distance(prediction, target)
        sample_cer = distance / max(1, len(target))

        axes[row, 0].imshow(original, cmap="gray", vmin=0, vmax=255, aspect="auto")
        axes[row, 0].set_title(f"{category}: original")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(processed, cmap="gray", vmin=0, vmax=255, aspect="auto")
        axes[row, 1].set_title(f"processed 32×{fixed_width}")
        axes[row, 1].axis("off")

        axes[row, 2].axis("off")
        axes[row, 2].text(
            0.0,
            0.72,
            f"prediction: {prediction}\ntarget:       {target}\n"
            f"edit distance: {distance}\nsample CER: {sample_cer:.2%}\n"
            f"exact: {prediction == target}",
            family="monospace",
            fontsize=12,
            va="top",
        )

        report_records.append(
            {
                "category": category,
                "index": index,
                "image": str(path),
                "prediction": prediction,
                "target": target,
                "edit_distance": distance,
                "sample_cer": sample_cer,
                "correct": prediction == target,
            }
        )

    figure.suptitle(
        f"CRNN validation — iteration {iteration:,}\n"
        f"loss={validation_loss:.4f}  CER={validation_cer:.2%}  "
        f"word accuracy={validation_word_accuracy:.2%}",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return report_records


def main() -> None:
    args = build_parser().parse_args()
    dataset = CaptchaDataset(
        data_path=args.data_root,
        manifest_path=args.manifest,
        split=False,
        fixed_width=args.fixed_width,
    )
    count = min(args.sample_count, len(dataset))
    subset = Subset(dataset, range(count))
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ctc_collate_fn,
    )
    model = load_model(args.checkpoint, args.device)
    criterion = nn.CTCLoss(blank=BLANK_IDX, reduction="sum", zero_infinity=True)
    predictions: list[str] = []
    targets: list[str] = []
    loss_sum = 0.0

    with torch.inference_mode():
        for batch in tqdm(loader, desc="validation report"):
            logits = model(batch.images.to(args.device), batch.input_widths)
            input_lengths = model.output_lengths(batch.input_widths)
            loss = compute_ctc_loss(
                logits,
                batch.targets,
                criterion,
                input_lengths=input_lengths,
                target_lengths=batch.target_lengths,
            )
            loss_sum += float(loss.cpu())
            predictions.extend(decode_predictions(logits, input_lengths))
            targets.extend(batch.texts)

    validation_loss = loss_sum / count
    validation_cer = char_error_rate(predictions, targets)
    validation_word_accuracy = word_accuracy(predictions, targets)
    paths = [Path(path) for path in dataset.paths[:count]]
    good, bad = _select_indices(
        predictions,
        targets,
        args.good_count,
        args.bad_count,
    )
    selected = [("GOOD", index) for index in good] + [("BAD", index) for index in bad]
    report_records = _render_report(
        args.output,
        paths,
        predictions,
        targets,
        selected,
        fixed_width=args.fixed_width,
        validation_loss=validation_loss,
        validation_cer=validation_cer,
        validation_word_accuracy=validation_word_accuracy,
        iteration=args.iteration,
    )
    prediction_log = save_validation_predictions(
        args.output.parent,
        args.iteration,
        predictions,
        targets,
        [str(path) for path in paths],
        validation_loss=validation_loss,
        validation_cer=validation_cer,
        validation_word_accuracy=validation_word_accuracy,
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "iteration": args.iteration,
        "sample_count": count,
        "validation_loss": validation_loss,
        "validation_cer": validation_cer,
        "validation_word_accuracy": validation_word_accuracy,
        "prediction_log": str(prediction_log),
        "examples": report_records,
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

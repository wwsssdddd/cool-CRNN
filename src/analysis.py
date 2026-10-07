"""Поиск и визуализация капч, на которых модель ошибается."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import BATCH_SIZE, BLANK_IDX, DEVICE, MODELS_PATH
from .checkpoint import load_model
from .dataset import CaptchaDataset, ctc_collate_fn
from .train import eval_epoch


def plot_problem_examples(
    images: list[np.ndarray],
    targets: list[str],
    predictions: list[str],
) -> None:
    """Рисует близкую к квадратной сетку ошибочно распознанных примеров."""
    if not (len(images) == len(targets) == len(predictions)):
        raise ValueError("images, targets и predictions должны иметь одинаковую длину")
    if not images:
        print("Ошибочных примеров нет.")
        return

    columns = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 2.4 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, image, target, prediction in zip(axes.flat, images, targets, predictions):
        axis.imshow(image, cmap="gray")
        axis.set_title(f"target: {target}\nprediction: {prediction}")
        axis.axis("off")
    figure.tight_layout()
    output = Path(MODELS_PATH) / "problem_examples.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    print(f"Визуализация ошибок сохранена: {output}")
    plt.show()


def searcher_for_problem_examples() -> None:
    """Находит ошибки модели на test split и визуализирует исходные изображения."""
    model_path = Path(MODELS_PATH) / "model"
    model = load_model(model_path)
    dataset = CaptchaDataset()
    dataset.set_state("test")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=ctc_collate_fn,
    )
    criterion = nn.CTCLoss(blank=BLANK_IDX, reduction="sum", zero_infinity=True)
    _, predictions, targets = eval_epoch(model, loader, criterion)

    images: list[np.ndarray] = []
    wrong_targets: list[str] = []
    wrong_predictions: list[str] = []
    for path, target, prediction in zip(dataset.paths, targets, predictions):
        if prediction == target:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Пропущено нечитаемое изображение: {path}")
            continue
        images.append(image)
        wrong_targets.append(target)
        wrong_predictions.append(prediction)

    plot_problem_examples(images, wrong_targets, wrong_predictions)


if __name__ == "__main__":
    searcher_for_problem_examples()

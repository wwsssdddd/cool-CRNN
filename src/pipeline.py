"""Запуск обучения и отрисовка кривых обучения."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from .config import (
    BATCH_SIZE,
    BLANK_IDX,
    DEVICE,
    EPOCHS,
    LR,
    MODELS_PATH,
    RESUME_PATH,
    SEED,
    TRAIN_MANIFEST_PATH,
    TRAIN_IMAGE_WIDTH,
    VAL_MANIFEST_PATH,
)
from .dataset import CaptchaDataset
from .model import RCNN
from .train import seed_everything, train


def plot_history(history: dict[str, list]) -> None:
    """Рисует и сохраняет кривые CTC-loss и CER."""
    output_dir = Path(MODELS_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    figure = plt.figure(figsize=(12, 5))
    axis_loss = figure.add_subplot(121)
    axis_loss.plot(epochs, history["train_loss"], label="train")
    axis_loss.plot(epochs, history["test_loss"], label="test")
    axis_loss.set(title="CTC Loss", xlabel="epoch", ylabel="loss")
    axis_loss.grid(alpha=0.25)
    axis_loss.legend()

    axis_cer = figure.add_subplot(122)
    axis_cer.plot(epochs, history["train_cer"], label="train")
    axis_cer.plot(epochs, history["test_cer"], label="test")
    axis_cer.set(title="Character Error Rate", xlabel="epoch", ylabel="CER")
    axis_cer.grid(alpha=0.25)
    axis_cer.legend()

    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=160, bbox_inches="tight")
    plt.show()


def save_model(model: nn.Module, path: str | Path) -> None:
    """Сохраняет только веса модели, чтобы файл не зависел от pickle-класса."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)


def training_pipeline() -> None:
    """Создаёт данные и модель, обучает, сохраняет лучший checkpoint и графики."""
    seed_everything(SEED)
    if bool(TRAIN_MANIFEST_PATH) != bool(VAL_MANIFEST_PATH):
        raise ValueError(
            "CRNN_TRAIN_MANIFEST и CRNN_VAL_MANIFEST должны быть заданы вместе"
        )
    if TRAIN_MANIFEST_PATH and VAL_MANIFEST_PATH:
        train_dataset = CaptchaDataset(
            manifest_path=TRAIN_MANIFEST_PATH,
            split=False,
            fixed_width=TRAIN_IMAGE_WIDTH,
        )
        test_dataset = CaptchaDataset(manifest_path=VAL_MANIFEST_PATH, split=False)
    else:
        train_dataset = CaptchaDataset(fixed_width=TRAIN_IMAGE_WIDTH)
        test_dataset = CaptchaDataset()
        test_dataset.set_state("test")

    output_dir = Path(MODELS_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "device": DEVICE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "seed": SEED,
        "train_manifest": TRAIN_MANIFEST_PATH,
        "validation_manifest": VAL_MANIFEST_PATH,
        "train_image_width": TRAIN_IMAGE_WIDTH,
        "train_samples": len(train_dataset),
        "validation_samples": len(test_dataset),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"device={DEVICE} | train={len(train_dataset)} | test={len(test_dataset)} | "
        f"epochs={EPOCHS} | batch_size={BATCH_SIZE}"
    )
    model = RCNN().to(DEVICE)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=LR, rho=0.9, eps=1e-6)
    criterion = nn.CTCLoss(blank=BLANK_IDX, reduction="sum", zero_infinity=True)
    history, best_model = train(
        train_dataset,
        test_dataset,
        model,
        optimizer,
        criterion,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        checkpoint_dir=MODELS_PATH,
        resume_path=RESUME_PATH,
    )
    model_path = Path(MODELS_PATH) / "model"
    save_model(best_model, model_path)
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Лучшая модель сохранена: {model_path}")
    plot_history(history)


if __name__ == "__main__":
    training_pipeline()

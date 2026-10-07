"""Обучение CRNN с CTC-таргетами и изображениями переменной длины."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import DEVICE, NUM_WORKERS
from .dataset import CTCBatch, convert_label_to_string, ctc_collate_fn
from .metrics import char_error_rate, word_accuracy


def seed_everything(seed: int) -> None:
    """Фиксирует генераторы Python, NumPy и PyTorch для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_ctc_loss(
    logits: torch.Tensor,
    targets: torch.LongTensor,
    criterion: nn.CTCLoss,
    input_lengths: torch.LongTensor | None = None,
    target_lengths: torch.LongTensor | None = None,
) -> torch.Tensor:
    """Считает CTC-loss для конкатенированных целей произвольной длины."""
    if logits.ndim != 3:
        raise ValueError(f"logits должен иметь форму (T, B, C), получено {tuple(logits.shape)}")
    time_steps, batch_size, _ = logits.shape

    if targets.ndim == 2:
        # Совместимость с простыми фиксированными тестовыми батчами.
        inferred_target_length = targets.shape[1]
        target_lengths = torch.full((batch_size,), inferred_target_length, dtype=torch.long)
        targets = targets.reshape(-1)
    elif targets.ndim != 1:
        raise ValueError("targets должен быть 1D (конкатенированные цели) или 2D")

    if input_lengths is None:
        input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long)
    if target_lengths is None:
        raise ValueError("Для одномерных targets необходимо передать target_lengths")

    input_lengths = input_lengths.detach().to(device="cpu", dtype=torch.long)
    target_lengths = target_lengths.detach().to(device="cpu", dtype=torch.long)
    if input_lengths.numel() != batch_size or target_lengths.numel() != batch_size:
        raise ValueError("input_lengths и target_lengths должны иметь по одному значению на объект")
    if int(target_lengths.sum()) != targets.numel():
        raise ValueError("Сумма target_lengths не совпадает с числом меток в targets")
    if torch.any(input_lengths > time_steps):
        raise ValueError("input_lengths превышает число временных шагов logits")
    if torch.any(target_lengths > input_lengths):
        raise ValueError("Цель длиннее доступной входной последовательности CTC")

    cpu_targets = targets.detach().to(device="cpu", dtype=torch.long)
    offset = 0
    for index, target_length in enumerate(target_lengths.tolist()):
        target = cpu_targets[offset : offset + target_length]
        repeats = int((target[1:] == target[:-1]).sum()) if target_length > 1 else 0
        if target_length + repeats > int(input_lengths[index]):
            raise ValueError(
                "Недостаточно временных шагов CTC для цели с повторяющимися символами"
            )
        offset += target_length

    log_probs = logits.float().log_softmax(dim=2)
    # CTCLoss не во всех версиях PyTorch реализован для MPS. Перенос на CPU
    # остаётся частью autograd-графа, поэтому градиент вернётся на MPS.
    if log_probs.device.type == "mps":
        log_probs = log_probs.cpu()
    targets = targets.detach().to(device=log_probs.device, dtype=torch.long).contiguous()
    return criterion(log_probs, targets, input_lengths, target_lengths)


def decode_predictions(
    logits: torch.Tensor,
    input_lengths: torch.LongTensor | None = None,
) -> list[str]:
    """Декодирует только реальные временные шаги каждого padded-объекта."""
    labels = logits.detach().argmax(dim=2).transpose(0, 1).cpu()
    if input_lengths is None:
        input_lengths = torch.full((labels.shape[0],), labels.shape[1], dtype=torch.long)
    lengths = input_lengths.detach().cpu().tolist()
    return [
        convert_label_to_string(row[:length], is_prediction=True)
        for row, length in zip(labels, lengths)
    ]


def _run_batch(
    model: nn.Module,
    batch: CTCBatch,
    criterion: nn.CTCLoss,
) -> tuple[torch.Tensor, list[str]]:
    images = batch.images.to(DEVICE)
    logits = model(images, batch.input_widths)
    input_lengths = model.output_lengths(batch.input_widths)
    loss = compute_ctc_loss(
        logits,
        batch.targets,
        criterion,
        input_lengths=input_lengths,
        target_lengths=batch.target_lengths,
    )
    return loss, decode_predictions(logits, input_lengths)


def fit_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, list[str], list[str]]:
    """Выполняет одну обучающую эпоху."""
    model.train()
    total_loss = 0.0
    total_examples = 0
    predicted_words: list[str] = []
    target_words: list[str] = []

    for batch in tqdm(train_loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        loss, predictions = _run_batch(model, batch, criterion)
        loss.backward()
        optimizer.step()

        batch_size = batch.images.shape[0]
        total_loss += float(loss.detach().cpu())
        total_examples += batch_size
        predicted_words.extend(predictions)
        target_words.extend(batch.texts)

    if total_examples == 0:
        raise ValueError("Обучающий DataLoader пуст")
    return total_loss / total_examples, predicted_words, target_words


def eval_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.CTCLoss,
) -> tuple[float, list[str], list[str]]:
    """Выполняет одну эпоху оценки без вычисления градиентов."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predicted_words: list[str] = []
    target_words: list[str] = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="valid", leave=False):
            loss, predictions = _run_batch(model, batch, criterion)
            batch_size = batch.images.shape[0]
            total_loss += float(loss.cpu())
            total_examples += batch_size
            predicted_words.extend(predictions)
            target_words.extend(batch.texts)

    if total_examples == 0:
        raise ValueError("Валидационный DataLoader пуст")
    return total_loss / total_examples, predicted_words, target_words


def _print_examples(predictions: list[str], targets: list[str], count: int = 3) -> None:
    pairs = list(zip(predictions, targets))[:count]
    if pairs:
        print("  examples:", ", ".join(f"{prediction!r}/{target!r}" for prediction, target in pairs))


def _load_checkpoint(path: str | Path) -> dict:
    """Загружает только созданный этим проектом обучающий checkpoint."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch без аргумента weights_only
        return torch.load(path, map_location="cpu")


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _save_checkpoint_atomic(state: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(destination)


def train(
    train_dataset: Dataset,
    test_dataset: Dataset,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    epochs: int,
    batch_size: int,
    checkpoint_dir: str | Path | None = None,
    resume_path: str | Path | None = None,
) -> tuple[dict[str, list], nn.Module]:
    """Обучает модель и возвращает историю и снимок с минимальным test CER."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs и batch_size должны быть положительными")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.startswith("cuda"),
        "collate_fn": ctc_collate_fn,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    history: dict[str, list] = {
        "train_loss": [],
        "test_loss": [],
        "train_cer": [],
        "test_cer": [],
        "train_word_acc": [],
        "test_word_acc": [],
    }
    best_cer = float("inf")
    best_model = copy.deepcopy(model).cpu()
    start_epoch = 1

    if resume_path:
        checkpoint = _load_checkpoint(resume_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer_state(optimizer, DEVICE)
        loaded_history = checkpoint.get("history", {})
        for key in history:
            history[key] = list(loaded_history.get(key, []))
        best_cer = float(checkpoint.get("best_cer", float("inf")))
        if "best_model_state_dict" in checkpoint:
            best_model.load_state_dict(checkpoint["best_model_state_dict"])
        else:
            best_model = copy.deepcopy(model).cpu()
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Обучение продолжено с эпохи {start_epoch}: {resume_path}")

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_predictions, train_targets = fit_epoch(
            model, train_loader, criterion, optimizer
        )
        test_loss, test_predictions, test_targets = eval_epoch(model, test_loader, criterion)
        values = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_cer": char_error_rate(train_predictions, train_targets),
            "test_cer": char_error_rate(test_predictions, test_targets),
            "train_word_acc": word_accuracy(train_predictions, train_targets),
            "test_word_acc": word_accuracy(test_predictions, test_targets),
        }
        for key, value in values.items():
            history[key].append(value)

        improved = values["test_cer"] < best_cer
        if improved:
            best_cer = values["test_cer"]
            best_model = copy.deepcopy(model).cpu()

        if checkpoint_dir is not None:
            output_dir = Path(checkpoint_dir)
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_model_state_dict": best_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_cer": best_cer,
                "history": history,
            }
            _save_checkpoint_atomic(checkpoint_state, output_dir / "last.pt")
            if improved:
                torch.save(best_model.state_dict(), output_dir / "best.pt")

        print(
            f"epoch {epoch:03d}/{epochs:03d} | "
            f"loss {train_loss:.4f}/{test_loss:.4f} | "
            f"CER {values['train_cer']:.4f}/{values['test_cer']:.4f} | "
            f"word acc {values['train_word_acc']:.4f}/{values['test_word_acc']:.4f}"
        )
        _print_examples(test_predictions, test_targets)

    return history, best_model

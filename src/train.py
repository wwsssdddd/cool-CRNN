"""Обучение CRNN с CTC-таргетами и изображениями переменной длины."""

from __future__ import annotations

import copy
import gzip
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
from tqdm.auto import tqdm

from .config import DEVICE, NUM_WORKERS, ORIGINAL_COMMIT
from .dataset import CTCBatch, convert_label_to_string, ctc_collate_fn
from .metrics import char_error_rate, levenshtein_distance, word_accuracy


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
    *,
    decode: bool = True,
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
    predictions = decode_predictions(logits, input_lengths) if decode else []
    return loss, predictions


def normalize_batch_loss(loss: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Делит CTC sum-loss на batch, как gradParams:div(nFrame) в оригинале."""
    if batch_size < 1:
        raise ValueError("batch_size должен быть положительным")
    return loss / batch_size


def fit_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    grad_clip_norm: float = 0.0,
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
        batch_size = batch.images.shape[0]
        normalized_loss = normalize_batch_loss(loss, batch_size)
        if not torch.isfinite(normalized_loss):
            raise FloatingPointError(f"Нефинитный CTC loss: {normalized_loss.item()}")
        normalized_loss.backward()
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

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
    pairs = list(zip(predictions, targets))
    correct = [(prediction, target) for prediction, target in pairs if prediction == target][:count]
    incorrect = [(prediction, target) for prediction, target in pairs if prediction != target][:count]
    if correct:
        print(
            "  correct examples:",
            ", ".join(f"{prediction!r}/{target!r}" for prediction, target in correct),
        )
    if incorrect:
        print(
            "  error examples:",
            ", ".join(f"{prediction!r}/{target!r}" for prediction, target in incorrect),
        )


def save_validation_predictions(
    output_dir: str | Path,
    iteration: int,
    predictions: list[str],
    targets: list[str],
    sample_paths: list[str | None],
    *,
    validation_loss: float,
    validation_cer: float,
    validation_word_accuracy: float,
) -> Path:
    """Атомарно сохраняет все ответы одной валидации в сжатый JSONL."""
    if not (len(predictions) == len(targets) == len(sample_paths)):
        raise ValueError("predictions, targets и sample_paths должны иметь одинаковую длину")

    log_dir = Path(output_dir) / "validation_predictions"
    log_dir.mkdir(parents=True, exist_ok=True)
    destination = log_dir / f"iteration_{iteration:07d}.jsonl.gz"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    summary = {
        "record_type": "summary",
        "iteration": iteration,
        "sample_count": len(predictions),
        "validation_loss": validation_loss,
        "validation_cer": validation_cer,
        "validation_word_accuracy": validation_word_accuracy,
    }

    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
            for index, (prediction, target, sample_path) in enumerate(
                zip(predictions, targets, sample_paths)
            ):
                distance = levenshtein_distance(prediction, target)
                record = {
                    "record_type": "sample",
                    "iteration": iteration,
                    "index": index,
                    "image": sample_path,
                    "target": target,
                    "prediction": prediction,
                    "correct": prediction == target,
                    "edit_distance": distance,
                    "sample_cer": distance / max(1, len(target)),
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _validation_sample_paths(dataset: Dataset, count: int) -> list[str | None]:
    """Возвращает пути первых validation-объектов, если Dataset их предоставляет."""
    paths = getattr(dataset, "paths", None)
    if paths is None or len(paths) < count:
        return [None] * count
    return [str(paths[index]) for index in range(count)]


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


def _save_history_atomic(history: dict[str, list], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(destination)


def train(
    train_dataset: Dataset,
    test_dataset: Dataset,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    batch_size: int,
    max_iterations: int,
    display_interval: int,
    validation_interval: int,
    validation_samples: int,
    val_batch_size: int,
    snapshot_interval: int,
    checkpoint_dir: str | Path | None = None,
    resume_path: str | Path | None = None,
    grad_clip_norm: float = 0.0,
) -> tuple[dict[str, list], nn.Module]:
    """Итерационное обучение по схеме официального ``training.lua``.

    Оригинал выбирает каждый minibatch случайно с возвращением, делит CTC
    sum-loss и градиенты на размер batch, валидируется через фиксированное
    число итераций и сохраняет периодические snapshots. Дополнительные CER и
    best-checkpoints не влияют на оптимизацию.
    """
    positive_values = {
        "batch_size": batch_size,
        "max_iterations": max_iterations,
        "display_interval": display_interval,
        "validation_interval": validation_interval,
        "validation_samples": validation_samples,
        "val_batch_size": val_batch_size,
        "snapshot_interval": snapshot_interval,
    }
    invalid = [name for name, value in positive_values.items() if value < 1]
    if invalid:
        raise ValueError(f"Параметры должны быть положительными: {', '.join(invalid)}")
    if grad_clip_norm < 0:
        raise ValueError("grad_clip_norm не может быть отрицательным")

    common_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.startswith("cuda"),
        "collate_fn": ctc_collate_fn,
    }
    history: dict[str, list] = {
        "iteration": [],
        "train_loss": [],
        "test_loss": [],
        "test_cer": [],
        "test_word_acc": [],
        "learning_rate": [],
    }
    best_cer = float("inf")
    best_word_acc = float("-inf")
    best_model = copy.deepcopy(model).cpu()
    start_iteration = 0

    if resume_path:
        checkpoint = _load_checkpoint(resume_path)
        if checkpoint.get("format_version") != 2:
            raise ValueError(
                "Checkpoint создан старой несовместимой архитектурой. "
                "Для исправленной реализации начните новый output directory."
            )
        if checkpoint.get("original_commit") != ORIGINAL_COMMIT:
            raise ValueError("Checkpoint относится к другой версии original-compatible модели")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer_state(optimizer, DEVICE)
        loaded_history = checkpoint.get("history", {})
        for key in history:
            history[key] = list(loaded_history.get(key, []))
        best_cer = float(checkpoint.get("best_cer", float("inf")))
        best_word_acc = float(checkpoint.get("best_word_acc", float("-inf")))
        if "best_model_state_dict" in checkpoint:
            best_model.load_state_dict(checkpoint["best_model_state_dict"])
        else:
            best_model = copy.deepcopy(model).cpu()
        start_iteration = int(checkpoint["iteration"])
        print(f"Обучение продолжено с итерации {start_iteration}: {resume_path}")

    validation_count = min(validation_samples, len(test_dataset))
    if validation_count < 1:
        raise ValueError("Валидационный Dataset пуст")
    validation_dataset = Subset(test_dataset, range(validation_count))
    validation_paths = _validation_sample_paths(test_dataset, validation_count)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.startswith("cuda"),
        collate_fn=ctc_collate_fn,
    )
    output_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None

    def validate_and_checkpoint(iteration: int, train_loss: float | None) -> None:
        nonlocal best_cer, best_word_acc, best_model
        val_loss, predictions, targets = eval_epoch(model, validation_loader, criterion)
        val_cer = char_error_rate(predictions, targets)
        val_word_acc = word_accuracy(predictions, targets)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history["iteration"].append(iteration)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(val_loss)
        history["test_cer"].append(val_cer)
        history["test_word_acc"].append(val_word_acc)
        history["learning_rate"].append(learning_rate)

        improved_cer = val_cer < best_cer
        improved_word_acc = val_word_acc > best_word_acc
        if improved_cer:
            best_cer = val_cer
            best_model = copy.deepcopy(model).cpu()
        if improved_word_acc:
            best_word_acc = val_word_acc

        if output_dir is not None:
            try:
                prediction_log = save_validation_predictions(
                    output_dir,
                    iteration,
                    predictions,
                    targets,
                    validation_paths,
                    validation_loss=val_loss,
                    validation_cer=val_cer,
                    validation_word_accuracy=val_word_acc,
                )
            except OSError as error:
                warnings.warn(
                    f"Не удалось сохранить validation-предсказания: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                prediction_log = None

            checkpoint_state = {
                "format_version": 2,
                "original_commit": ORIGINAL_COMMIT,
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "best_model_state_dict": best_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_cer": best_cer,
                "best_word_acc": best_word_acc,
                "history": history,
            }
            _save_checkpoint_atomic(checkpoint_state, output_dir / "last.pt")
            _save_history_atomic(history, output_dir / "history.json")
            if improved_cer:
                _save_checkpoint_atomic(best_model.state_dict(), output_dir / "best_cer.pt")
                _save_checkpoint_atomic(best_model.state_dict(), output_dir / "best.pt")
            if improved_word_acc:
                _save_checkpoint_atomic(model.state_dict(), output_dir / "best_word_accuracy.pt")

        train_text = "n/a" if train_loss is None else f"{train_loss:.4f}"
        print(
            f"iteration {iteration:07d}/{max_iterations:07d} | "
            f"loss train/val {train_text}/{val_loss:.4f} | "
            f"val CER {val_cer:.4f} | val word acc {val_word_acc:.4f} | "
            f"lr {learning_rate:g}"
        )
        if output_dir is not None and prediction_log is not None:
            print(f"  validation answers: {prediction_log}")
        _print_examples(predictions, targets)

    if start_iteration >= max_iterations:
        print(f"Достигнуто max_iterations={max_iterations}; новые шаги не требуются")
        return history, best_model

    # Как в оригинале: validation до первого обновления модели.
    if start_iteration == 0 and not history["iteration"]:
        validate_and_checkpoint(0, None)

    remaining_iterations = max_iterations - start_iteration
    sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=remaining_iterations * batch_size,
    )
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        drop_last=True,
        **common_loader_kwargs,
    )

    interval_loss_sum = 0.0
    interval_examples = 0
    display_loss_sum = 0.0
    display_examples = 0

    for offset, batch in enumerate(train_loader, 1):
        iteration = start_iteration + offset
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _run_batch(model, batch, criterion, decode=False)
        actual_batch_size = batch.images.shape[0]
        normalized_loss = normalize_batch_loss(loss, actual_batch_size)
        if not torch.isfinite(normalized_loss):
            raise FloatingPointError(
                f"Нефинитный CTC loss на итерации {iteration}: {normalized_loss.item()}"
            )
        normalized_loss.backward()
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        raw_loss = float(loss.detach().cpu())
        interval_loss_sum += raw_loss
        interval_examples += actual_batch_size
        display_loss_sum += raw_loss
        display_examples += actual_batch_size

        if iteration % display_interval == 0:
            print(
                f"iteration {iteration:07d}/{max_iterations:07d} | "
                f"train loss {display_loss_sum / display_examples:.4f}"
            )
            display_loss_sum = 0.0
            display_examples = 0

        if iteration % validation_interval == 0 or iteration == max_iterations:
            mean_train_loss = interval_loss_sum / interval_examples
            validate_and_checkpoint(iteration, mean_train_loss)
            interval_loss_sum = 0.0
            interval_examples = 0

        if output_dir is not None and iteration % snapshot_interval == 0:
            _save_checkpoint_atomic(
                model.state_dict(),
                output_dir / f"snapshot_{iteration:07d}.pt",
            )

    return history, best_model

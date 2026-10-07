"""Точная CTC-вероятность строки по forward algorithm из статьи."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from .config import BLANK_IDX


def _to_numpy(log_probs: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(log_probs, torch.Tensor):
        values = log_probs.detach().to(device="cpu", dtype=torch.float64).numpy()
    else:
        values = np.asarray(log_probs, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Ожидается матрица (T, C), получено {values.shape}")
    return values


def ctc_log_probability(
    log_probs: torch.Tensor | np.ndarray,
    labels: Sequence[int],
    blank: int = BLANK_IDX,
) -> float:
    """Возвращает ``log p(labels | log_probs)`` без потери точности на длинных T."""
    matrix = _to_numpy(log_probs)
    time_steps, classes = matrix.shape
    target = [int(label) for label in labels]
    if not 0 <= blank < classes:
        raise ValueError(f"blank={blank} вне диапазона классов 0..{classes - 1}")
    if any(label == blank or not 0 <= label < classes for label in target):
        raise ValueError("Таргет содержит blank или индекс вне диапазона классов")
    if time_steps == 0:
        return 0.0 if not target else -math.inf

    required_steps = len(target) + sum(a == b for a, b in zip(target, target[1:]))
    if required_steps > time_steps:
        return -math.inf

    extended = [blank]
    for label in target:
        extended.extend((label, blank))

    previous = np.full(len(extended), -np.inf, dtype=np.float64)
    previous[0] = matrix[0, blank]
    if len(extended) > 1:
        previous[1] = matrix[0, extended[1]]

    for time_index in range(1, time_steps):
        current = np.full_like(previous, -np.inf)
        for state, label in enumerate(extended):
            predecessors = [previous[state]]
            if state > 0:
                predecessors.append(previous[state - 1])
            if state > 1 and label != blank and label != extended[state - 2]:
                predecessors.append(previous[state - 2])
            current[state] = matrix[time_index, label] + np.logaddexp.reduce(predecessors)
        previous = current

    if len(extended) == 1:
        return float(previous[0])
    return float(np.logaddexp(previous[-1], previous[-2]))


def text_log_probability(
    log_probs: torch.Tensor | np.ndarray,
    text: str,
    char_to_label: dict[str, int],
    blank: int = BLANK_IDX,
) -> float:
    """Кодирует строку и вычисляет её точную CTC log-вероятность."""
    try:
        labels = [char_to_label[character] for character in text]
    except KeyError as error:
        raise ValueError(f"Символ {error.args[0]!r} отсутствует в алфавите") from error
    return ctc_log_probability(log_probs, labels, blank)


def ctc_probability(
    log_probs: torch.Tensor | np.ndarray,
    labels: Sequence[int],
    blank: int = BLANK_IDX,
) -> float:
    """Вероятность в обычном масштабе; для длинных строк может округлиться до нуля."""
    return math.exp(ctc_log_probability(log_probs, labels, blank))

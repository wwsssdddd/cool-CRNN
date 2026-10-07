"""Lexicon-free и lexicon-based CTC transcription из раздела 2.3 статьи."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .bktree import BKTree
from .config import BLANK_IDX, CHARS
from .ctc_probability import text_log_probability
from .dataset import CHAR_2_LABEL, convert_label_to_string


def load_lexicon(path: str | Path) -> list[str]:
    """Читает по одному слову на строку, нормализует и удаляет дубликаты."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Словарь не найден: {source}")
    return normalize_lexicon(source.read_text(encoding="utf-8").splitlines())


def normalize_lexicon(words: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    allowed = set(CHARS)
    for raw_word in words:
        word = raw_word.strip().lower()
        if not word or word in seen:
            continue
        unknown = set(word) - allowed
        if unknown:
            raise ValueError(f"Слово {word!r} содержит символы вне алфавита: {sorted(unknown)}")
        result.append(word)
        seen.add(word)
    if not result:
        raise ValueError("Лексикон пуст")
    return result


def greedy_decode_log_probs(
    log_probs: torch.Tensor | np.ndarray,
    blank: int = BLANK_IDX,
) -> str:
    values = (
        log_probs.detach().cpu().numpy()
        if isinstance(log_probs, torch.Tensor)
        else np.asarray(log_probs)
    )
    if values.ndim != 2:
        raise ValueError("log_probs должен иметь форму (T, C)")
    labels = values.argmax(axis=1).tolist()
    if blank != BLANK_IDX:
        result: list[int] = []
        previous: int | None = None
        for label in labels:
            if label != previous and label != blank:
                result.append(label)
            previous = label
        return "".join(CHARS[label] for label in result)
    return convert_label_to_string(labels, is_prediction=True)


@dataclass(frozen=True)
class LexiconResult:
    text: str
    greedy_text: str
    log_probability: float
    candidate_count: int
    search_time_ms: float


class LexiconDecoder:
    """Ранжирует словарные кандидаты по полной CTC-вероятности Eq. (1)."""

    def __init__(self, words: Iterable[str]) -> None:
        self.words = normalize_lexicon(words)
        self.tree = BKTree(self.words)

    def decode(
        self,
        log_probs: torch.Tensor | np.ndarray,
        *,
        delta: int | None = None,
    ) -> LexiconResult:
        """``delta=None`` делает полный перебор, число включает BK-tree поиск."""
        started = time.perf_counter()
        greedy = greedy_decode_log_probs(log_probs)
        if delta is None:
            candidates = self.words
        else:
            candidates = [word for _, word in self.tree.query(greedy, delta)]

        # При пустом N_delta статья фактически сохраняет lexicon-free ответ;
        # это также делает delta=0 эквивалентным greedy режиму.
        if not candidates:
            try:
                score = text_log_probability(log_probs, greedy, CHAR_2_LABEL)
            except ValueError:
                score = -math.inf
            return LexiconResult(
                greedy,
                greedy,
                score,
                0,
                (time.perf_counter() - started) * 1000,
            )

        best_word = candidates[0]
        best_score = -math.inf
        for candidate in candidates:
            score = text_log_probability(log_probs, candidate, CHAR_2_LABEL)
            if score > best_score:
                best_word = candidate
                best_score = score
        return LexiconResult(
            best_word,
            greedy,
            best_score,
            len(candidates),
            (time.perf_counter() - started) * 1000,
        )

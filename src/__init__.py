"""Публичный интерфейс пакета CRNN с ленивой загрузкой тяжёлых модулей."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "training_pipeline",
    "searcher_for_problem_examples",
    "RCNN",
    "CaptchaDataset",
    "LexiconDecoder",
    "BKTree",
]

_EXPORTS = {
    "training_pipeline": (".pipeline", "training_pipeline"),
    "searcher_for_problem_examples": (".analysis", "searcher_for_problem_examples"),
    "RCNN": (".model", "RCNN"),
    "CaptchaDataset": (".dataset", "CaptchaDataset"),
    "LexiconDecoder": (".lexicon", "LexiconDecoder"),
    "BKTree": (".bktree", "BKTree"),
}


def __getattr__(name: str) -> Any:
    """Импортирует matplotlib/OpenCV только когда соответствующий API запрошен."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value

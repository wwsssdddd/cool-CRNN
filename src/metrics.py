"""Метрики качества распознавания капчи."""

from __future__ import annotations


def _validate_pairs(predictions: list[str], targets: list[str]) -> None:
    if len(predictions) != len(targets):
        raise ValueError(
            f"Число предсказаний ({len(predictions)}) не равно числу ответов ({len(targets)})"
        )


def levenshtein_distance(prediction: str, target: str) -> int:
    """Вычисляет расстояние Левенштейна за O(len(prediction) * len(target))."""
    if len(prediction) > len(target):
        prediction, target = target, prediction
    previous = list(range(len(prediction) + 1))
    for target_index, target_char in enumerate(target, start=1):
        current = [target_index]
        for prediction_index, prediction_char in enumerate(prediction, start=1):
            insertion = current[-1] + 1
            deletion = previous[prediction_index] + 1
            substitution = previous[prediction_index - 1] + (prediction_char != target_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def char_error_rate(predictions: list[str], targets: list[str]) -> float:
    """CER: сумма редакционных расстояний, делённая на число символов в ответах."""
    _validate_pairs(predictions, targets)
    total_chars = sum(len(target) for target in targets)
    if total_chars == 0:
        return 0.0 if not predictions or all(not prediction for prediction in predictions) else float("inf")
    errors = sum(
        levenshtein_distance(prediction, target)
        for prediction, target in zip(predictions, targets)
    )
    return errors / total_chars


def word_accuracy(predictions: list[str], targets: list[str]) -> float:
    """Доля строк, распознанных полностью без ошибок."""
    _validate_pairs(predictions, targets)
    if not targets:
        return 0.0
    return sum(prediction == target for prediction, target in zip(predictions, targets)) / len(targets)

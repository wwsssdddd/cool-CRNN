"""BK-tree для поиска строк в радиусе расстояния Левенштейна."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from .metrics import levenshtein_distance


Distance = Callable[[str, str], int]


@dataclass
class _Node:
    value: str
    children: dict[int, "_Node"] = field(default_factory=dict)


class BKTree:
    """Метрическое дерево, использованное в lexicon search статьи."""

    def __init__(
        self,
        values: Iterable[str] = (),
        distance: Distance = levenshtein_distance,
    ) -> None:
        self.distance = distance
        self.root: _Node | None = None
        self.size = 0
        for value in values:
            self.add(value)

    def add(self, value: str) -> None:
        if self.root is None:
            self.root = _Node(value)
            self.size = 1
            return
        node = self.root
        while True:
            edge = self.distance(value, node.value)
            if edge == 0:
                return
            child = node.children.get(edge)
            if child is None:
                node.children[edge] = _Node(value)
                self.size += 1
                return
            node = child

    def query(self, value: str, radius: int) -> list[tuple[int, str]]:
        """Возвращает пары ``(distance, word)`` с расстоянием не больше radius."""
        if radius < 0:
            raise ValueError("radius должен быть неотрицательным")
        if self.root is None:
            return []
        results: list[tuple[int, str]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = self.distance(value, node.value)
            if distance <= radius:
                results.append((distance, node.value))
            lower = distance - radius
            upper = distance + radius
            stack.extend(
                child
                for edge, child in node.children.items()
                if lower <= edge <= upper
            )
        results.sort(key=lambda item: (item[0], item[1]))
        return results

"""Ленивая загрузка изображений и подготовка CTC-батчей переменной длины."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .config import (
    BLANK_IDX,
    CHARS,
    DATA_PATH,
    IMAGE_HEIGHT,
    MANIFEST_PATH,
    MIN_IMAGE_WIDTH,
    RANDOM_STATE,
    TEST_SIZE,
)

CHAR_2_LABEL = {char: idx for idx, char in enumerate(CHARS)}
LABEL_2_CHAR = {idx: char for idx, char in enumerate(CHARS)}
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def convert_label_to_string(
    label: Sequence[int] | np.ndarray | torch.Tensor,
    is_prediction: bool = False,
) -> str:
    """Переводит метки в строку; для предсказания выполняет CTC greedy decode."""
    if isinstance(label, torch.Tensor):
        values = label.detach().cpu().reshape(-1).tolist()
    elif isinstance(label, np.ndarray):
        values = label.reshape(-1).tolist()
    else:
        values = list(label)

    result: list[str] = []
    previous: int | None = None
    for raw_value in values:
        value = int(raw_value)
        if is_prediction:
            # В CTC повтор разделяется только blank-символом: a, blank, a -> aa.
            if value != previous and value != BLANK_IDX:
                if value not in LABEL_2_CHAR:
                    raise ValueError(f"Неизвестная метка класса: {value}")
                result.append(LABEL_2_CHAR[value])
            previous = value
        else:
            if value not in LABEL_2_CHAR:
                raise ValueError(f"Таргет содержит blank или неизвестную метку: {value}")
            result.append(LABEL_2_CHAR[value])
    return "".join(result)


def ctc_required_steps(label: str) -> int:
    """Минимум CTC-шагов: повторы одинаковых букв требуют blank между ними."""
    return len(label) + sum(left == right for left, right in zip(label, label[1:]))


def read_image(
    img_name: str | Path,
    minimum_width: int = MIN_IMAGE_WIDTH,
    fixed_width: int | None = None,
) -> torch.Tensor:
    """Приводит изображение к высоте 32, сохраняя пропорции и переменную ширину."""
    image = cv2.imread(str(img_name), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {img_name}")
    source_height, source_width = image.shape
    resized_width = fixed_width or max(
        minimum_width,
        round(source_width * IMAGE_HEIGHT / source_height),
    )
    interpolation = cv2.INTER_AREA if resized_width < source_width else cv2.INTER_CUBIC
    image = cv2.resize(image, (resized_width, IMAGE_HEIGHT), interpolation=interpolation)
    # Нормализация как в распространённых реализациях CRNN: [0, 255] -> [-1, 1].
    image = image.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(image).unsqueeze(0)


@dataclass
class CTCBatch:
    """Батч с padded-изображениями и конкатенированными CTC-таргетами."""

    images: torch.Tensor
    targets: torch.LongTensor
    input_widths: torch.LongTensor
    target_lengths: torch.LongTensor
    texts: list[str]


def ctc_collate_fn(samples: list[tuple[torch.Tensor, torch.LongTensor]]) -> CTCBatch:
    """Дополняет изображения справа белым фоном, не дополняя таргеты."""
    if not samples:
        raise ValueError("Нельзя собрать пустой батч")
    images, labels = zip(*samples)
    widths = torch.tensor([image.shape[-1] for image in images], dtype=torch.long)
    max_width = int(widths.max())
    batch_images = torch.ones(len(images), 1, IMAGE_HEIGHT, max_width, dtype=torch.float32)
    for index, image in enumerate(images):
        batch_images[index, :, :, : image.shape[-1]] = image

    target_lengths = torch.tensor([label.numel() for label in labels], dtype=torch.long)
    targets = torch.cat(labels).to(dtype=torch.long)
    texts = [convert_label_to_string(label) for label in labels]
    return CTCBatch(batch_images, targets, widths, target_lengths, texts)


class CaptchaDataset(Dataset):
    """Детерминированные train/test части; изображения читаются только по запросу."""

    def __init__(
        self,
        data_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        split: bool = True,
        fixed_width: int | None = None,
    ) -> None:
        data_dir = Path(data_path or DATA_PATH).expanduser()
        if not data_dir.is_dir():
            raise FileNotFoundError(
                f"Папка с данными не найдена: {data_dir}. "
                "Создайте её или задайте переменную CRNN_DATA_PATH."
            )

        selected_manifest = manifest_path or MANIFEST_PATH
        if selected_manifest:
            manifest = Path(selected_manifest).expanduser()
            if not manifest.is_file():
                raise FileNotFoundError(f"Manifest не найден: {manifest}")
            paths: list[Path] = []
            labels: list[str] = []
            for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
                if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                    continue
                parts = raw_line.split("\t", maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"Строка {line_number} manifest должна иметь path<TAB>label")
                relative_path, label = parts
                path = Path(relative_path).expanduser()
                if not path.is_absolute():
                    path = data_dir / path
                paths.append(path)
                labels.append(label.strip().lower())
        else:
            paths = sorted(
                path
                for path in data_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            labels = [path.stem.lower() for path in paths]

        if len(paths) < 2:
            raise ValueError(f"В {data_dir} нужно минимум два изображения, найдено: {len(paths)}")
        for path, label in zip(paths, labels):
            if not path.is_file():
                raise FileNotFoundError(f"Изображение из manifest не найдено: {path}")
            if not label:
                raise ValueError(f"Пустой ответ в имени файла: {path.name}")
            unknown = set(label) - set(CHARS)
            if unknown:
                raise ValueError(f"Файл {path.name} содержит символы вне CHARS: {sorted(unknown)}")

        if split:
            partitions = train_test_split(
                paths,
                labels,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                shuffle=True,
            )
            (
                self.train_paths,
                self.test_paths,
                self.train_labels,
                self.test_labels,
            ) = partitions
        else:
            self.train_paths = paths
            self.train_labels = labels
            self.test_paths = []
            self.test_labels = []
        self.fixed_width = fixed_width
        self.state = "train"

    def set_state(self, state: str) -> None:
        if state not in {"train", "test"}:
            raise ValueError("state должен быть 'train' или 'test'")
        self.state = state

    def _label_2_longtensor(self, label: str) -> torch.LongTensor:
        try:
            encoded = [CHAR_2_LABEL[char] for char in label]
        except KeyError as error:
            raise ValueError(f"Неизвестный символ в ответе {label!r}: {error.args[0]!r}") from error
        return torch.tensor(encoded, dtype=torch.long)

    @property
    def labels(self) -> list[str]:
        return self.train_labels if self.state == "train" else self.test_labels

    @property
    def paths(self) -> list[Path]:
        return self.train_paths if self.state == "train" else self.test_paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.LongTensor]:
        label = self.labels[idx]
        # Для T ≈ W/4 + 1 ширина 4*required_steps гарантирует допустимое
        # CTC-выравнивание даже для слов с повторяющимися буквами.
        minimum_width = max(MIN_IMAGE_WIDTH, 4 * ctc_required_steps(label))
        if self.fixed_width is not None:
            available_steps = self.fixed_width // 4 + 1
            if ctc_required_steps(label) > available_steps:
                raise ValueError(
                    f"Слово {label!r} требует {ctc_required_steps(label)} CTC-шагов, "
                    f"но fixed width {self.fixed_width} даёт только {available_steps}"
                )
        return (
            read_image(self.paths[idx], minimum_width, self.fixed_width),
            self._label_2_longtensor(label),
        )

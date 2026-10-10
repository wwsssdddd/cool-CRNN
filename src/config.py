"""Константы, общие для всех модулей."""

from __future__ import annotations

import os
import string
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------- Данные ----------
# Папка с картинками. Имя файла = правильный ответ, например "2b827.png".
DATA_PATH = os.environ.get("CRNN_DATA_PATH", str(PROJECT_ROOT / "data"))

# Необязательный TSV: относительный_путь<TAB>текст. Нужен, когда ответ нельзя
# получить из имени файла.
MANIFEST_PATH = os.environ.get("CRNN_MANIFEST")
TRAIN_MANIFEST_PATH = os.environ.get("CRNN_TRAIN_MANIFEST")
VAL_MANIFEST_PATH = os.environ.get("CRNN_VAL_MANIFEST")

# Папка, куда сохраняется лучшая модель.
MODELS_PATH = os.environ.get("CRNN_MODELS_PATH", str(PROJECT_ROOT / "models"))

TEST_SIZE = float(os.environ.get("CRNN_TEST_SIZE", "0.2")) # доля датасета, которая пойдёт в тестовую (валидационную) часть
RANDOM_STATE = int(os.environ.get("CRNN_RANDOM_STATE", "42")) # для деления train / test

# ---------- Алфавит ----------
CHARS = string.digits + string.ascii_lowercase
BLANK_IDX = len(CHARS)
NUM_CLASSES = len(CHARS) + 1

# ---------- Изображения ----------
IMAGE_HEIGHT = 32
# Минимальная ширина соответствует протоколу статьи. Более длинные изображения
# сохраняют пропорции и могут иметь любую большую ширину.
MIN_IMAGE_WIDTH = int(os.environ.get("CRNN_MIN_IMAGE_WIDTH", "100"))
_train_width = int(os.environ.get("CRNN_TRAIN_IMAGE_WIDTH", "0"))
TRAIN_IMAGE_WIDTH = _train_width or None


def _best_available_device() -> str:
    """Выбирает ускоритель без привязки к конкретному компьютеру."""
    requested = os.environ.get("CRNN_DEVICE")
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------- Обучение ----------
DEVICE = _best_available_device()
SEED = int(os.environ.get("CRNN_SEED", "0")) # для всех случайных действий кроме деления train / test
EPOCHS = int(os.environ.get("CRNN_EPOCHS", "1"))
BATCH_SIZE = int(os.environ.get("CRNN_BATCH_SIZE", "64")) # в одной эпохе все данные делятся на батчи, по которым идет итерация
LR = float(os.environ.get("CRNN_LR", "1.0")) # Learning rate
NUM_WORKERS = int(os.environ.get("CRNN_NUM_WORKERS", "0")) # для мультипоточности, сколько потоков
RESUME_PATH = os.environ.get("CRNN_RESUME") # Чекпоинт, с него можно продолить если надо

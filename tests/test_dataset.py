from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.dataset import read_image


def test_read_image_shape_range_and_dtype(tmp_path: Path) -> None:
    path = tmp_path / "abc12.png"
    image = np.arange(48 * 160, dtype=np.uint8).reshape(48, 160)
    assert cv2.imwrite(str(path), image)
    tensor = read_image(path)
    assert tensor.shape == (1, 32, 107)
    assert tensor.dtype == torch.float32
    assert -1.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_read_image_reports_bad_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Не удалось прочитать"):
        read_image(tmp_path / "missing.png")

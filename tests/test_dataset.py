from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.dataset import CaptchaDataset, convert_label_to_string, read_image
from src.prepare_manifest import (
    collect_samples,
    validate_samples,
    write_mjsynth_manifest_streaming,
)


def test_read_image_shape_range_and_dtype(tmp_path: Path) -> None:
    path = tmp_path / "abc12.png"
    image = np.arange(48 * 160, dtype=np.uint8).reshape(48, 160)
    assert cv2.imwrite(str(path), image)
    tensor = read_image(path)
    assert tensor.shape == (1, 32, 107)
    assert tensor.dtype == torch.float32
    assert -1.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_read_image_uses_original_minus_128_over_128_normalization(tmp_path: Path) -> None:
    black_path = tmp_path / "black.png"
    white_path = tmp_path / "white.png"
    assert cv2.imwrite(str(black_path), np.zeros((32, 100), dtype=np.uint8))
    assert cv2.imwrite(str(white_path), np.full((32, 100), 255, dtype=np.uint8))
    black = read_image(black_path, fixed_width=100)
    white = read_image(white_path, fixed_width=100)
    assert torch.all(black == -1.0)
    assert torch.all(white == 127.0 / 128.0)


def test_read_image_reports_bad_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Не удалось прочитать"):
        read_image(tmp_path / "missing.png")


def test_dataset_skips_unreadable_image(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not a jpeg")
    good = tmp_path / "good.jpg"
    assert cv2.imwrite(str(good), np.zeros((20, 40), dtype=np.uint8))
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("bad.jpg\tbad\ngood.jpg\tgood\n", encoding="utf-8")
    dataset = CaptchaDataset(
        data_path=tmp_path,
        manifest_path=manifest,
        split=False,
        fixed_width=100,
    )

    with pytest.warns(RuntimeWarning, match="Нечитаемое изображение пропущено"):
        image, target = dataset[0]

    assert image.shape == (1, 32, 100)
    assert convert_label_to_string(target) == "good"


def test_icdar_xml_maps_word_prefix_and_labels(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_path = image_root / "1" / "1.jpg"
    image_path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(image_path), np.zeros((20, 40), dtype=np.uint8))
    annotations = tmp_path / "labels.xml"
    annotations.write_text(
        '<?xml version="1.0"?><imagelist>'
        '<image file="word/1/1.jpg" tag="Hello" />'
        '</imagelist>',
        encoding="utf-8",
    )

    samples = collect_samples(image_root, annotations, "icdar_xml")
    assert samples == [(image_path, "hello")]
    valid, skipped = validate_samples(samples, skip_invalid=False)
    assert valid == samples
    assert skipped == 0


def test_write_mjsynth_manifest_streaming(tmp_path: Path) -> None:
    root = tmp_path / "mjsynth"
    image = root / "12" / "3" / "7_Hello_42.jpg"
    image.parent.mkdir(parents=True)
    assert cv2.imwrite(str(image), np.zeros((20, 40), dtype=np.uint8))
    annotations = root / "annotation_train.txt"
    annotations.write_text("./12/3/7_Hello_42.jpg 42\n", encoding="utf-8")
    manifest = tmp_path / "train.tsv"

    written, skipped = write_mjsynth_manifest_streaming(
        root,
        annotations,
        manifest,
        skip_invalid=False,
        skip_file_check=False,
    )

    assert (written, skipped) == (1, 0)
    assert manifest.read_text(encoding="utf-8") == "12/3/7_Hello_42.jpg\thello\n"

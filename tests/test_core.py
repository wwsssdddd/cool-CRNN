from __future__ import annotations

import torch
from torch import nn

from src.config import BLANK_IDX, NUM_CLASSES
from src.dataset import CHAR_2_LABEL, convert_label_to_string, ctc_collate_fn
from src.metrics import char_error_rate, levenshtein_distance, word_accuracy
from src.model import RCNN
from src.train import compute_ctc_loss, decode_predictions


def test_model_output_shape_and_gradients() -> None:
    model = RCNN()
    images = torch.rand(2, 1, 32, 148)
    widths = torch.tensor([100, 148])
    logits = model(images, widths)
    assert logits.shape == (38, 2, NUM_CLASSES)
    assert model.output_lengths(widths).tolist() == [26, 38]
    logits.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_ctc_decode_collapses_repeats_before_removing_blank() -> None:
    a = CHAR_2_LABEL["a"]
    b = CHAR_2_LABEL["b"]
    sequence = [a, a, BLANK_IDX, a, b, b, BLANK_IDX]
    assert convert_label_to_string(sequence, is_prediction=True) == "aab"


def test_decode_predictions_handles_batch_dimension() -> None:
    a = CHAR_2_LABEL["a"]
    logits = torch.full((3, 2, NUM_CLASSES), -10.0)
    logits[:, :, BLANK_IDX] = 10.0
    logits[0, 0, a] = 20.0
    assert decode_predictions(logits) == ["a", ""]


def test_ctc_loss_is_finite_and_differentiable() -> None:
    logits = torch.randn(31, 2, NUM_CLASSES, requires_grad=True)
    words = ["cat", "longword12"]
    labels = torch.tensor([CHAR_2_LABEL[c] for word in words for c in word], dtype=torch.long)
    target_lengths = torch.tensor([len(word) for word in words])
    input_lengths = torch.tensor([26, 31])
    criterion = nn.CTCLoss(blank=BLANK_IDX, reduction="sum", zero_infinity=True)
    loss = compute_ctc_loss(logits, labels, criterion, input_lengths, target_lengths)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_collate_supports_different_image_and_target_lengths() -> None:
    first = (torch.zeros(1, 32, 100), torch.tensor([1, 2, 3]))
    second = (torch.zeros(1, 32, 140), torch.tensor([4, 5, 6, 7, 8, 9, 10]))
    batch = ctc_collate_fn([first, second])
    assert batch.images.shape == (2, 1, 32, 140)
    assert batch.input_widths.tolist() == [100, 140]
    assert batch.target_lengths.tolist() == [3, 7]
    assert batch.targets.shape == (10,)


def test_variable_length_batch_runs_end_to_end() -> None:
    words = ["cat", "balloon7"]
    samples = [
        (
            torch.rand(1, 32, width),
            torch.tensor([CHAR_2_LABEL[char] for char in word]),
        )
        for word, width in zip(words, [100, 144])
    ]
    batch = ctc_collate_fn(samples)
    model = RCNN()
    logits = model(batch.images, batch.input_widths)
    input_lengths = model.output_lengths(batch.input_widths)
    criterion = nn.CTCLoss(blank=BLANK_IDX, reduction="sum", zero_infinity=True)
    loss = compute_ctc_loss(
        logits,
        batch.targets,
        criterion,
        input_lengths,
        batch.target_lengths,
    )
    assert torch.isfinite(loss)
    loss.backward()


def test_metrics() -> None:
    predictions = ["2b87", "abcde"]
    targets = ["2b827", "abcde"]
    assert levenshtein_distance(predictions[0], targets[0]) == 1
    assert char_error_rate(predictions, targets) == 0.1
    assert word_accuracy(predictions, targets) == 0.5

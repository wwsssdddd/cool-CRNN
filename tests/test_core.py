from __future__ import annotations

import torch
from torch import nn

from src.config import BLANK_IDX, DEVICE, NUM_CLASSES
from src.dataset import CHAR_2_LABEL, convert_label_to_string, ctc_collate_fn
from src.metrics import char_error_rate, levenshtein_distance, word_accuracy
from src.model import RCNN
from src.train import compute_ctc_loss, decode_predictions, normalize_batch_loss, train


def test_original_class_indices() -> None:
    assert BLANK_IDX == 0
    assert CHAR_2_LABEL["0"] == 1
    assert CHAR_2_LABEL["9"] == 10
    assert CHAR_2_LABEL["a"] == 11
    assert CHAR_2_LABEL["z"] == 36


def test_original_batch_norm_positions_and_bidirectional_join() -> None:
    model = RCNN()
    convolution_number = 0
    batch_norm_after: list[int] = []
    for module in model.conv_layer:
        if isinstance(module, nn.Conv2d):
            convolution_number += 1
        elif isinstance(module, nn.BatchNorm2d):
            batch_norm_after.append(convolution_number)
    assert convolution_number == 7
    assert batch_norm_after == [3, 5, 7]
    assert hasattr(model.rnn_layer_1, "forward_projection")
    assert hasattr(model.rnn_layer_1, "backward_projection")


def test_original_recurrent_initialization() -> None:
    model = RCNN()
    for layer in (model.rnn_layer_1, model.rnn_layer_2):
        for name, parameter in layer.rnn.named_parameters():
            if "weight" in name:
                assert float(parameter.detach().min()) >= -0.080001
                assert float(parameter.detach().max()) <= 0.080001
            else:
                assert torch.count_nonzero(parameter) == 0
        assert torch.count_nonzero(layer.forward_projection.bias) == 0
        assert torch.count_nonzero(layer.backward_projection.bias) == 0


def test_sum_ctc_loss_is_normalized_by_actual_batch_size() -> None:
    parameter = torch.tensor(2.0, requires_grad=True)
    summed_loss = parameter * 64
    normalized = normalize_batch_loss(summed_loss, batch_size=64)
    normalized.backward()
    assert float(normalized.detach()) == 2.0
    assert float(parameter.grad) == 1.0


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


def test_original_iteration_training_loop_smoke() -> None:
    words = ["cat", "book", "hello", "word7"]
    dataset = [
        (
            torch.rand(1, 32, 100) * 2 - 1,
            torch.tensor([CHAR_2_LABEL[char] for char in word]),
        )
        for word in words
    ]
    model = RCNN().to(DEVICE)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-6)
    criterion = nn.CTCLoss(blank=BLANK_IDX, reduction="sum", zero_infinity=True)
    history, _ = train(
        dataset,
        dataset,
        model,
        optimizer,
        criterion,
        batch_size=2,
        max_iterations=2,
        display_interval=1,
        validation_interval=1,
        validation_samples=2,
        val_batch_size=2,
        snapshot_interval=2,
    )
    assert history["iteration"] == [0, 1, 2]
    assert all(torch.isfinite(torch.tensor(history["test_loss"])))


def test_metrics() -> None:
    predictions = ["2b87", "abcde"]
    targets = ["2b827", "abcde"]
    assert levenshtein_distance(predictions[0], targets[0]) == 1
    assert char_error_rate(predictions, targets) == 0.1
    assert word_accuracy(predictions, targets) == 0.5

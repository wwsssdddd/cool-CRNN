"""CRNN из статьи: CNN -> последовательность -> два BiLSTM -> CTC logits."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import NUM_CLASSES


class BidirectionalLSTM(nn.Module):
    """Один двунаправленный LSTM и проекция его двух направлений."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, sequence: torch.Tensor, lengths: torch.LongTensor) -> torch.Tensor:
        total_length = sequence.shape[0]
        packed = pack_padded_sequence(
            sequence,
            lengths.cpu(),
            enforce_sorted=False,
        )
        packed_output, _ = self.rnn(packed)
        recurrent, _ = pad_packed_sequence(packed_output, total_length=total_length)
        time_steps, batch_size, features = recurrent.shape
        projected = self.embedding(recurrent.reshape(time_steps * batch_size, features))
        return projected.reshape(time_steps, batch_size, -1)


class RCNN(nn.Module):
    """Полностью свёрточная по ширине CRNN, принимающая слова разной длины."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Асимметричный pooling сохраняет горизонтальное разрешение.
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.rnn_layer_1 = BidirectionalLSTM(512, 256, 256)
        self.rnn_layer_2 = BidirectionalLSTM(256, 256, NUM_CLASSES)
        self.apply(self._init_weights)

    @staticmethod
    def output_lengths(input_widths: torch.LongTensor) -> torch.LongTensor:
        """Ширина после CNN: два /2 pooling и итоговое смещение +1."""
        return torch.div(input_widths, 4, rounding_mode="floor") + 1

    def forward(
        self,
        images: torch.Tensor,
        input_widths: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """Возвращает logits формы ``(T_max, B, NUM_CLASSES)``."""
        features = self.conv_layer(images)
        batch_size, channels, height, time_steps = features.shape
        if height != 1:
            raise ValueError(
                f"После CNN высота должна быть 1, получена форма {tuple(features.shape)}"
            )
        sequence = features.squeeze(2).permute(2, 0, 1).contiguous()

        if input_widths is None:
            sequence_lengths = torch.full(
                (batch_size,), time_steps, dtype=torch.long, device="cpu"
            )
        else:
            sequence_lengths = self.output_lengths(input_widths.to(dtype=torch.long, device="cpu"))
            if torch.any(sequence_lengths > time_steps):
                raise ValueError("Рассчитанная длина последовательности превышает выход CNN")

        sequence = self.rnn_layer_1(sequence, sequence_lengths)
        return self.rnn_layer_2(sequence, sequence_lengths)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
        elif isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
            nn.init.zeros_(module.bias)

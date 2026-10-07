"""Параметры и размер CRNN для проверки Table 3."""

from __future__ import annotations

import json

from .model import RCNN


def main() -> None:
    model = RCNN()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        json.dumps(
            {
                "parameters": parameters,
                "trainable_parameters": trainable,
                "fp32_size_mb_decimal": parameters * 4 / 1_000_000,
                "fp32_size_mib": parameters * 4 / 1024**2,
                "paper_reported_parameters": 8_300_000,
                "paper_reported_size_mb": 33,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

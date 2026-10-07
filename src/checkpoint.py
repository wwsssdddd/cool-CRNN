"""Загрузка весов из model/best.pt или полного last.pt checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .config import DEVICE
from .model import RCNN


def load_model(path: str | Path, device: str = DEVICE) -> nn.Module:
    model = RCNN()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if "model_state_dict" in payload:
        payload = payload.get("best_model_state_dict", payload["model_state_dict"])
    model.load_state_dict(payload)
    model.to(device)
    model.eval()
    return model

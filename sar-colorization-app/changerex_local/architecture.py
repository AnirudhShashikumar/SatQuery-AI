"""Strict checkpoint-compatible standalone ChangerEx model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional

from .backbone import IAResNetV1c18
from .decoder import ChangerDecoder


class ChangerEx(nn.Module):
    """ChangerEx with names and tensor shapes matching the official checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = IAResNetV1c18()
        self.decode_head = ChangerDecoder()

    def forward_quarter_resolution(self, pair: torch.Tensor) -> torch.Tensor:
        if pair.ndim != 4 or pair.shape[1] != 6:
            raise ValueError("ChangerEx expects an NCHW tensor with six RGB-pair channels")
        earlier, later = torch.chunk(pair, 2, dim=1)
        return self.decode_head(self.backbone(earlier, later))

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        logits = self.forward_quarter_resolution(pair)
        return functional.interpolate(
            logits, size=pair.shape[-2:], mode="bilinear", align_corners=False
        )


def build_model() -> ChangerEx:
    return ChangerEx()


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

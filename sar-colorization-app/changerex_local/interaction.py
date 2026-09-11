"""Changer feature-exchange layers.

Adapted from Open-CD's Apache-2.0 licensed
``opencd/models/utils/interaction_layer.py`` at the pinned commit.
"""

from __future__ import annotations

import torch
from torch import nn


class TwoIdentity(nn.Module):
    def forward(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return first, second


class SpatialExchange(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0 < p <= 1:
            raise ValueError("p must be in (0, 1]")
        self.period = int(1 / p)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.arange(first.shape[-1], device=first.device) % self.period == 0
        output_first = torch.zeros_like(first)
        output_second = torch.zeros_like(second)
        output_first[..., ~mask] = first[..., ~mask]
        output_second[..., ~mask] = second[..., ~mask]
        output_first[..., mask] = second[..., mask]
        output_second[..., mask] = first[..., mask]
        return output_first, output_second


class ChannelExchange(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0 < p <= 1:
            raise ValueError("p must be in (0, 1]")
        self.period = int(1 / p)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.arange(first.shape[1], device=first.device) % self.period == 0
        output_first = torch.zeros_like(first)
        output_second = torch.zeros_like(second)
        output_first[:, ~mask] = first[:, ~mask]
        output_second[:, ~mask] = second[:, ~mask]
        output_first[:, mask] = second[:, mask]
        output_second[:, mask] = first[:, mask]
        return output_first, output_second

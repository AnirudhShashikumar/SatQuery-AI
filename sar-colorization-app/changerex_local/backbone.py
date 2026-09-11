"""Pure-PyTorch IA-ResNetV1c-18 backbone for ChangerEx.

Module names intentionally match the official Open-CD checkpoint. Architecture
is adapted from Apache-2.0 licensed Open-CD and MMSegmentation ResNet sources.
"""

from __future__ import annotations

import torch
from torch import nn

from .interaction import ChannelExchange, SpatialExchange, TwoIdentity


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.SyncBatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.SyncBatchNorm(out_channels)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.SyncBatchNorm(out_channels),
            )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        identity = tensor if self.downsample is None else self.downsample(tensor)
        output = self.relu(self.bn1(self.conv1(tensor)))
        output = self.bn2(self.conv2(output))
        return self.relu(output + identity)


class IAResNetV1c18(nn.Module):
    """Exact four-stage, deep-stem interaction ResNet-18."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.SyncBatchNorm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.SyncBatchNorm(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 1)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)
        self.layer4 = self._make_layer(256, 512, 2)
        self.ccs = nn.ModuleList(
            [TwoIdentity(), SpatialExchange(0.5), ChannelExchange(0.5), ChannelExchange(0.5)]
        )

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels),
        )

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, ...]:
        first = self.maxpool(self.stem(first))
        second = self.maxpool(self.stem(second))
        outputs: list[torch.Tensor] = []
        for layer, interaction in zip(
            (self.layer1, self.layer2, self.layer3, self.layer4), self.ccs
        ):
            first = layer(first)
            second = layer(second)
            first, second = interaction(first, second)
            outputs.append(torch.cat((first, second), dim=1))
        return tuple(outputs)

"""Pure-PyTorch Changer decoder head.

Adapted from Open-CD's Apache-2.0 licensed
``opencd/models/decode_heads/changer.py`` at the pinned commit.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class ConvModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.SyncBatchNorm(out_channels)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(tensor)))


class FDAF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flow_make = nn.Sequential(
            nn.Conv2d(128, 128, 5, padding=2, groups=128, bias=True),
            nn.InstanceNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 4, 1, bias=False),
        )

    @staticmethod
    def warp(tensor: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = tensor.shape
        norm = tensor.new_tensor([width, height]).view(1, 1, 1, 2)
        column = torch.linspace(-1.0, 1.0, height, device=tensor.device, dtype=tensor.dtype)
        column = column.view(-1, 1).repeat(1, width)
        row = torch.linspace(-1.0, 1.0, width, device=tensor.device, dtype=tensor.dtype)
        row = row.repeat(height, 1)
        grid = torch.stack((row, column), dim=2).repeat(batch, 1, 1, 1)
        return functional.grid_sample(
            tensor, grid + flow.permute(0, 2, 3, 1) / norm, align_corners=True
        )

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first_flow, second_flow = torch.chunk(
            self.flow_make(torch.cat((first, second), dim=1)), 2, dim=1
        )
        first_difference = self.warp(first, first_flow) - second
        second_difference = self.warp(second, second_flow) - first
        return torch.cat((first_difference, second_difference), dim=1)


class MixFFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=True),
            nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=True),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Conv2d(128, 128, 1, bias=True),
            nn.Dropout(0.0),
        )
        self.dropout_layer = nn.Identity()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.dropout_layer(self.layers(tensor))


class ChangerDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_seg = nn.Conv2d(128, 2, 1)
        self.convs = nn.ModuleList([ConvModule(channels, 128) for channels in (64, 128, 256, 512)])
        self.fusion_conv = ConvModule(512, 64)
        self.neck_layer = FDAF()
        self.discriminator = MixFFN()

    def _base_forward(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        projections = []
        for projection, feature in zip(self.convs, features):
            projected = projection(feature)
            if projected.shape[-2:] != target_size:
                projected = functional.interpolate(
                    projected, size=target_size, mode="bilinear", align_corners=False
                )
            projections.append(projected)
        return self.fusion_conv(torch.cat(projections, dim=1))

    def forward(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        split = [torch.chunk(feature, 2, dim=1) for feature in inputs]
        first = tuple(item[0] for item in split)
        second = tuple(item[1] for item in split)
        aligned = self.neck_layer(self._base_forward(first), self._base_forward(second))
        return self.conv_seg(self.discriminator(aligned))

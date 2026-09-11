"""Training-compatible SARFusionFormer and normalized CIE Lab conversion."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class MultiScaleEncoder(nn.Module):
    def __init__(self, input_channels: int, base_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.GELU(),
        )
        self.fine = nn.Sequential(
            ResidualBlock(base_channels, dropout), ResidualBlock(base_channels, dropout)
        )
        self.down_mid = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.mid = nn.Sequential(
            ResidualBlock(base_channels * 2, dropout),
            ResidualBlock(base_channels * 2, dropout),
        )
        self.down_coarse = nn.Conv2d(
            base_channels * 2, base_channels * 4, 3, stride=2, padding=1
        )
        self.coarse = nn.Sequential(
            ResidualBlock(base_channels * 4, dropout),
            ResidualBlock(base_channels * 4, dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fine = self.fine(self.stem(x))
        mid = self.mid(self.down_mid(fine))
        coarse = self.coarse(self.down_coarse(mid))
        return fine, mid, coarse


class FeatureFusion(nn.Module):
    def __init__(self, base_channels: int) -> None:
        super().__init__()
        channels = base_channels * 4
        self.fine_down = nn.Conv2d(base_channels, channels, 5, stride=4, padding=2)
        self.mid_down = nn.Conv2d(base_channels * 2, channels, 3, stride=2, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
            ResidualBlock(channels),
        )

    def forward(
        self, fine: torch.Tensor, mid: torch.Tensor, coarse: torch.Tensor
    ) -> torch.Tensor:
        return self.fuse(torch.cat((self.fine_down(fine), self.mid_down(mid), coarse), dim=1))


def _window_partition(x: torch.Tensor, window: int) -> torch.Tensor:
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window, window, width // window, window, channels)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window, channels)


def _window_reverse(
    windows: torch.Tensor, window: int, batch: int, height: int, width: int, channels: int
) -> torch.Tensor:
    x = windows.view(batch, height // window, width // window, window, window, channels)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, channels)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, heads: int, window_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("Embedding dimension must be divisible by attention heads.")
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.window_size = window_size

    def forward(self, x: torch.Tensor, shift: bool = False) -> torch.Tensor:
        batch, height, width, channels = x.shape
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        padded = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
        padded_h, padded_w = padded.shape[1:3]
        if shift:
            padded = torch.roll(
                padded,
                shifts=(-self.window_size // 2, -self.window_size // 2),
                dims=(1, 2),
            )
        windows = _window_partition(padded, self.window_size)
        attended, _ = self.attention(windows, windows, windows, need_weights=False)
        output = _window_reverse(
            attended, self.window_size, batch, padded_h, padded_w, channels
        )
        if shift:
            output = torch.roll(
                output,
                shifts=(self.window_size // 2, self.window_size // 2),
                dims=(1, 2),
            )
        return output[:, :height, :width]


class SwinBlock(nn.Module):
    def __init__(
        self, dim: int, heads: int, window_size: int, shift: bool, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention(dim, heads, window_size, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.permute(0, 2, 3, 1)
        tokens = tokens + self.attention(self.norm1(tokens), self.shift)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.permute(0, 3, 1, 2)


class SwinBottleneck(nn.Module):
    def __init__(
        self, dim: int, depth: int, heads: int, window_size: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                SwinBlock(dim, heads, window_size, index % 2 == 1, dropout)
                for index in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class CrossAttention2d(nn.Module):
    def __init__(
        self,
        query_channels: int,
        context_channels: int,
        heads: int = 4,
        dropout: float = 0.0,
        max_context_tokens: int = 256,
    ) -> None:
        super().__init__()
        if query_channels % heads:
            raise ValueError("Query channels must be divisible by attention heads.")
        self.query_norm = nn.LayerNorm(query_channels)
        self.context_norm = nn.LayerNorm(context_channels)
        self.attention = nn.MultiheadAttention(
            query_channels,
            heads,
            kdim=context_channels,
            vdim=context_channels,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(query_channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(query_channels, query_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_channels * 2, query_channels),
        )
        self.max_context_tokens = max_context_tokens

    def forward(
        self, query_map: torch.Tensor, context_map: torch.Tensor, return_attention: bool = False
    ):
        batch, channels, height, width = query_map.shape
        if context_map.shape[-2] * context_map.shape[-1] > self.max_context_tokens:
            side = max(1, int(self.max_context_tokens**0.5))
            context_map = F.adaptive_avg_pool2d(context_map, (side, side))
        query = query_map.flatten(2).transpose(1, 2)
        context = context_map.flatten(2).transpose(1, 2)
        attended, weights = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=return_attention,
            average_attn_weights=True,
        )
        output = query + attended
        output = output + self.feed_forward(self.out_norm(output))
        output = output.transpose(1, 2).reshape(batch, channels, height, width)
        return (output, weights) if return_attention else output


class DualLabDecoder(nn.Module):
    def __init__(self, base_channels: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        channels = base_channels
        self.up_mid = nn.ConvTranspose2d(channels * 4, channels * 2, 2, stride=2)
        self.mid_attention = CrossAttention2d(
            channels * 2, channels * 2, heads=max(1, min(heads, channels * 2)), dropout=dropout
        )
        self.mid_refine = ResidualBlock(channels * 2, dropout)
        self.up_full = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
        self.full_attention = CrossAttention2d(
            channels, channels, heads=max(1, min(heads, channels)), dropout=dropout
        )
        self.shared = ResidualBlock(channels, dropout)
        self.structure = nn.Sequential(
            ResidualBlock(channels, dropout), nn.Conv2d(channels, 1, 3, padding=1), nn.Sigmoid()
        )
        self.color_attention = CrossAttention2d(
            channels, channels, heads=max(1, min(heads, channels)), dropout=dropout
        )
        self.color = nn.Sequential(
            ResidualBlock(channels, dropout), nn.Conv2d(channels, 2, 3, padding=1), nn.Sigmoid()
        )
        self.aux_quarter = nn.Sequential(nn.Conv2d(channels * 4, 3, 1), nn.Sigmoid())
        self.aux_half = nn.Sequential(nn.Conv2d(channels * 2, 3, 1), nn.Sigmoid())

    def forward(
        self,
        bottleneck: torch.Tensor,
        mid_sar: torch.Tensor,
        fine_sar: torch.Tensor,
        explain: bool = False,
    ) -> Dict[str, object]:
        mid = self.mid_refine(self.mid_attention(self.up_mid(bottleneck), mid_sar))
        full = self.up_full(mid)
        if explain:
            full, attention = self.full_attention(full, fine_sar, return_attention=True)
        else:
            full, attention = self.full_attention(full, fine_sar), None
        shared = self.shared(full)
        luminance = self.structure(shared)
        chroma = self.color(self.color_attention(shared, fine_sar))
        lab = torch.cat((luminance, chroma), dim=1)
        return {
            "lab": lab,
            "aux": [self.aux_quarter(bottleneck), self.aux_half(mid)],
            "features": [shared, mid, bottleneck],
            "attention": attention,
        }


class SARFusionFormer(nn.Module):
    """Multi-scale CNN, shifted-window Transformer, and dual Lab decoder."""

    def __init__(
        self,
        input_channels: int = 2,
        output_channels: int = 3,
        base_channels: int = 48,
        transformer_depth: int = 4,
        attention_heads: int = 6,
        window_size: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if output_channels != 3:
            raise ValueError("SARFusionFormer requires three normalized Lab output channels.")
        dimension = base_channels * 4
        if dimension % attention_heads:
            raise ValueError("base_channels * 4 must be divisible by attention heads.")
        self.encoder = MultiScaleEncoder(input_channels, base_channels, dropout)
        self.fusion = FeatureFusion(base_channels)
        self.transformer = SwinBottleneck(
            dimension, transformer_depth, attention_heads, window_size, dropout
        )
        self.decoder = DualLabDecoder(base_channels, attention_heads, dropout)

    def forward(self, sar: torch.Tensor, explain: bool = False) -> Dict[str, object]:
        if sar.ndim != 4:
            raise ValueError("SARFusionFormer expects a B×2×H×W tensor.")
        height, width = sar.shape[-2:]
        pad_h, pad_w = (-height) % 4, (-width) % 4
        padded = F.pad(sar, (0, pad_w, 0, pad_h), mode="replicate") if pad_h or pad_w else sar
        fine, mid, coarse = self.encoder(padded)
        contextual = self.transformer(self.fusion(fine, mid, coarse))
        output = self.decoder(contextual, mid, fine, explain=explain)
        if pad_h or pad_w:
            output["lab"] = output["lab"][..., :height, :width]
            output["aux"] = [
                item[..., : max(1, height // divisor), : max(1, width // divisor)]
                for item, divisor in zip(output["aux"], (4, 2))
            ]
        return output


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """Convert normalized Lab (L/100, (a,b + 128)/255) to clipped sRGB."""
    if lab.ndim != 4 or lab.shape[1] != 3:
        raise ValueError("LAB prediction must have shape [batch, 3, height, width].")
    lightness, a_channel, b_channel = (
        lab[:, 0:1] * 100,
        lab[:, 1:2] * 255 - 128,
        lab[:, 2:3] * 255 - 128,
    )
    fy = (lightness + 16) / 116
    fx, fz = a_channel / 500 + fy, fy - b_channel / 200
    epsilon, kappa = 216 / 24389, 24389 / 27

    def inverse_f(value: torch.Tensor) -> torch.Tensor:
        cube = value.pow(3)
        return torch.where(cube > epsilon, cube, (116 * value - 16) / kappa)

    xyz = torch.cat(
        (inverse_f(fx) * 0.95047, inverse_f(fy), inverse_f(fz) * 1.08883), dim=1
    )
    matrix = lab.new_tensor(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    rgb_linear = torch.einsum("ij,bjhw->bihw", matrix, xyz)
    return torch.where(
        rgb_linear > 0.0031308,
        1.055 * rgb_linear.clamp_min(0).pow(1 / 2.4) - 0.055,
        12.92 * rgb_linear,
    ).clamp(0, 1)

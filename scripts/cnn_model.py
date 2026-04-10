from __future__ import annotations

import math

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv-BN-ReLU convenience block used by the deeper CNN."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class QuickDrawCNN(nn.Module):
    """A compact LeNet-style CNN for 28x28 grayscale QuickDraw images."""

    def __init__(
        self,
        num_classes: int,
        *,
        input_size: int = 28,
        conv_channels: tuple[int, int] = (32, 64),
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}.")

        c1, c2 = conv_channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        flattened_dim = c2 * math.floor(input_size / 4) * math.floor(input_size / 4)
        if flattened_dim <= 0:
            raise ValueError(
                f"input_size={input_size} becomes too small after pooling; expected at least 4."
            )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class QuickDrawDeepCNN(nn.Module):
    """
    A stronger 3-stage CNN with batch normalization and adaptive pooling.

    This is intended as a drop-in upgrade over the original LeNet-style model:
    - deeper feature extractor
    - batch normalization for stabler optimization
    - adaptive average pooling to reduce parameter count in the classifier
    """

    def __init__(
        self,
        num_classes: int,
        *,
        input_size: int = 28,
        conv_channels: tuple[int, ...] = (64, 128, 256),
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}.")
        if len(conv_channels) != 3:
            raise ValueError(
                "QuickDrawDeepCNN expects exactly 3 conv channel values, "
                f"got {conv_channels}."
            )

        c1, c2, c3 = conv_channels
        self.features = nn.Sequential(
            ConvBlock(1, c1),
            ConvBlock(c1, c1),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout * 0.25),

            ConvBlock(c1, c2),
            ConvBlock(c2, c2),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout * 0.25),

            ConvBlock(c2, c3),
            ConvBlock(c3, c3),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

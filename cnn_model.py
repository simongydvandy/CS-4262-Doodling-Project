from __future__ import annotations

import math

import torch
from torch import nn


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

"""Small interpretable 1D CNN for averaged NASA Z-903 spectra."""

from __future__ import annotations

from torch import Tensor, nn


class Z903CNN(nn.Module):
    """Strided convolutional baseline producing six composition logits."""

    def __init__(self, output_size: int = 6) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[1:] != (1, 23401):
            raise ValueError(f"Expected input shape [batch, 1, 23401], got {tuple(inputs.shape)}")
        return self.classifier(self.features(inputs).flatten(1))
"""Six independent target-window branches for physics-constrained Z-903 learning."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PhysicsConstrainedZ903CNN(nn.Module):
    def __init__(self, branch_lengths: list[int], output_size: int = 6) -> None:
        super().__init__()
        if len(branch_lengths) != output_size:
            raise ValueError("branch_lengths must contain one length per target")
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, 8, kernel_size=5, padding=2),
                    nn.BatchNorm1d(8),
                    nn.ReLU(),
                    nn.Conv1d(8, 16, kernel_size=5, padding=2),
                    nn.BatchNorm1d(16),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                for _ in branch_lengths
            ]
        )
        self.heads = nn.ModuleList([nn.Linear(16, 1) for _ in branch_lengths])

    def forward(self, inputs: list[Tensor]) -> Tensor:
        if len(inputs) != len(self.branches):
            raise ValueError(f"Expected {len(self.branches)} target branches, got {len(inputs)}")
        logits = []
        for branch, head, values in zip(self.branches, self.heads, inputs, strict=True):
            if values.ndim != 3 or values.shape[1] != 1:
                raise ValueError(f"Each branch input must be [batch, 1, length], got {tuple(values.shape)}")
            logits.append(head(branch(values).flatten(1)))
        return torch.cat(logits, dim=1)
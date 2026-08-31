"""Small gated hybrid full-spectrum and target-window Z-903 CNN."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class HybridZ903CNN(nn.Module):
    def __init__(self, branch_lengths: list[int], output_size: int = 6) -> None:
        super().__init__()
        self.global_backbone = nn.Sequential(
            nn.Conv1d(1, 8, 11, stride=4, padding=5),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, 9, stride=4, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.global_projection = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Dropout(0.35))
        self.line_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, 8, 5, padding=2),
                    nn.BatchNorm1d(8),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                for _ in branch_lengths
            ]
        )
        self.line_projection = nn.ModuleList([nn.Linear(8, 8) for _ in branch_lengths])
        self.gates = nn.ModuleList(
            [nn.Sequential(nn.Linear(24, 8), nn.Sigmoid()) for _ in branch_lengths]
        )
        self.heads = nn.ModuleList([nn.Linear(24, 1) for _ in branch_lengths])

    def forward(
        self,
        full: Tensor,
        branches: list[Tensor],
        disable_global: bool = False,
        disable_lines: bool = False,
    ) -> Tensor:
        if full.ndim != 3 or full.shape[1:] != (1, 23401) or len(branches) != len(self.heads):
            raise ValueError("Hybrid inputs must be full [batch,1,23401] plus six branch tensors")
        global_features = self.global_projection(self.global_backbone(full).flatten(1))
        logits = []
        for branch, projection, gate, head, values in zip(
            self.line_branches, self.line_projection, self.gates, self.heads, branches, strict=True
        ):
            line_features = projection(branch(values).flatten(1))
            if disable_global:
                global_for_target = torch.zeros_like(global_features)
            else:
                global_for_target = global_features
            if disable_lines:
                line_features = torch.zeros_like(line_features)
            fused = torch.cat([global_for_target, line_features], dim=1)
            gated_line = line_features * gate(fused)
            logits.append(head(torch.cat([global_for_target, gated_line], dim=1)))
        return torch.cat(logits, dim=1)


from __future__ import annotations

import torch


def predict_probs(model: torch.nn.Module, images: torch.Tensor, device: str = "cpu") -> list[float]:
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        logits = model(images)
        probs = torch.sigmoid(logits)
    return probs.cpu().numpy()


__all__ = ["predict_probs"]

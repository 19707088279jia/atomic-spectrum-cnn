from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from atomic_spectrum_ai.mixture_dataset import MixtureDataset


def train_simple(model: nn.Module, manifest_csv: str, root_dir: str | None = None, epochs: int = 2, batch_size: int = 8, lr: float = 1e-3, device: str = "cpu") -> dict:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    ds = MixtureDataset(manifest_csv, root_dir=root_dir, transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _epoch in range(epochs):
        total_loss = 0.0
        for imgs, labels in dl:
            imgs = imgs.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * imgs.size(0)
        avg_loss = total_loss / len(ds)
    return {"loss": avg_loss}


def save_checkpoint(model: nn.Module, path: str, class_names=None):
    state = {"model_state_dict": model.state_dict(), "class_names": class_names}
    torch.save(state, path)


def load_checkpoint(path: str, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    chk = torch.load(path, map_location=device)
    return chk, chk.get("class_names")


__all__ = ["train_simple", "save_checkpoint", "load_checkpoint"]

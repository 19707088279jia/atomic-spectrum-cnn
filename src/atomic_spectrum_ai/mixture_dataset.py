from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

LABEL_COLUMNS = ["Zn", "Mn", "Cd", "Mg", "Cu", "Pb"]


class MixtureDataset(Dataset):
    def __init__(self, manifest_csv: str, root_dir: str | None = None, transform=None):
        self.manifest_csv = Path(manifest_csv)
        if root_dir is None:
            self.root_dir = self.manifest_csv.parent
        else:
            self.root_dir = Path(root_dir)
        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_csv}")
        self.df = pd.read_csv(self.manifest_csv)
        for c in LABEL_COLUMNS:
            if c not in self.df.columns:
                raise ValueError(f"Missing label column in manifest: {c}")
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row['file']
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            # lazy import to avoid heavy deps at module import time
            from torchvision.transforms import ToTensor

            img = ToTensor()(img)
        labels = torch.tensor([float(row[c]) for c in LABEL_COLUMNS], dtype=torch.float32)
        return img, labels


__all__ = ["MixtureDataset", "LABEL_COLUMNS"]

#!/usr/bin/env python
"""Load a mixture checkpoint and predict on one generated sample."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from atomic_spectrum_ai.mixture_model import build_mixture_model  # noqa: E402


def load_checkpoint_to_model(path: str):
    chk = torch.load(path, map_location='cpu')
    model = build_mixture_model(num_classes=len(chk.get('class_names', [0])), pretrained=False)
    model.load_state_dict(chk['model_state_dict'])
    model.eval()
    return model, chk


def main():
    # find a generated sample
    data_dir = Path("data/mixture/train")
    files = list(data_dir.glob("mix_*.png"))
    if not files:
        print("No generated samples found under data/mixture/train")
        return
    img_path = files[0]
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    x = transform(img).unsqueeze(0)
    chk_path = Path("outputs/mixture_smoke/mixture_smoke.pt")
    if not chk_path.exists():
        print("Checkpoint not found:", chk_path)
        return
    model, chk = load_checkpoint_to_model(str(chk_path))
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(0).tolist()
    out = {c: float(p) for c, p in zip(chk.get('class_names', []), probs, strict=True)}
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

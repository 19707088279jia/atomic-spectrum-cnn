"""Dataset and transform construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(config: dict[str, Any], training: bool) -> transforms.Compose:
    """Build scientifically conservative image transforms."""
    image_size = int(config["data"]["image_size"])
    steps: list[Any] = [transforms.Resize((image_size, image_size))]

    if training:
        augmentation = config.get("augmentation", {})
        rotation = float(augmentation.get("rotation_degrees", 0.0))
        vertical_shift = float(augmentation.get("vertical_shift", 0.0))
        brightness = float(augmentation.get("brightness", 0.0))
        contrast = float(augmentation.get("contrast", 0.0))

        # Horizontal translation and horizontal flip are intentionally disabled because
        # x-position carries wavelength meaning. Vertical shift only is allowed.
        if rotation or vertical_shift:
            steps.append(
                transforms.RandomAffine(
                    degrees=rotation,
                    translate=(0.0, vertical_shift),
                    fill=255,
                )
            )
        if brightness or contrast:
            steps.append(
                transforms.ColorJitter(brightness=brightness, contrast=contrast)
            )

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(steps)


def build_dataloaders(config: dict[str, Any]) -> tuple[dict[str, DataLoader], list[str]]:
    """Build train, validation, and test ImageFolder loaders."""
    root = Path(config["data"]["root"])
    batch_size = int(config["training"]["batch_size"])
    workers = int(config["data"].get("num_workers", 0))

    loaders: dict[str, DataLoader] = {}
    expected_classes = list(config["data"]["classes"])
    discovered_classes: list[str] | None = None

    for split in ("train", "validation", "test"):
        split_path = root / split
        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Missing dataset split: {split_path}. Generate or add data first."
            )
        dataset = datasets.ImageFolder(
            split_path,
            transform=build_transforms(config, training=split == "train"),
        )
        if not dataset.samples:
            raise ValueError(f"No images found in {split_path}")

        if discovered_classes is None:
            discovered_classes = dataset.classes
        elif dataset.classes != discovered_classes:
            raise ValueError(
                f"Class folders differ across splits: {split} has {dataset.classes}, "
                f"expected {discovered_classes}"
            )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=workers,
            pin_memory=False,
        )

    if discovered_classes != sorted(expected_classes):
        raise ValueError(
            "Class folders must match config classes. "
            f"Found {discovered_classes}, expected {sorted(expected_classes)}"
        )
    return loaders, discovered_classes

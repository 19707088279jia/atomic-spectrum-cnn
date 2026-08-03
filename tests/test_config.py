from pathlib import Path

import pytest

from atomic_spectrum_ai.config import load_config


def test_load_quickstart_config() -> None:
    config = load_config(Path("configs/quickstart.yaml"))
    assert config["model"]["name"] == "simple_cnn"
    assert config["data"]["classes"] == ["Fe", "Cu", "Na", "Ca", "Mg"]


def test_missing_config_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("does-not-exist.yaml")

import torch

from atomic_spectrum_ai.config import load_config
from atomic_spectrum_ai.model import SimpleSpectrumCNN, create_model


def test_simple_cnn_output_shape() -> None:
    model = SimpleSpectrumCNN(num_classes=5)
    output = model(torch.randn(2, 3, 96, 96))
    assert output.shape == (2, 5)


def test_create_model_from_config() -> None:
    config = load_config("configs/quickstart.yaml")
    model = create_model(config, num_classes=5)
    assert isinstance(model, SimpleSpectrumCNN)

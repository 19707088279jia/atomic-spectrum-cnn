import torch

from atomic_spectrum_ai.mixture_model import build_mixture_model


def test_model_output_shape():
    model = build_mixture_model(num_classes=6, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 6)

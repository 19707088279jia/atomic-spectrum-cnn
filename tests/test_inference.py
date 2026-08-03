import torch
from PIL import Image

from atomic_spectrum_ai.inference import predict_image
from atomic_spectrum_ai.model import SimpleSpectrumCNN


def test_predict_image_returns_probabilities() -> None:
    model = SimpleSpectrumCNN(num_classes=2)
    model.eval()
    checkpoint = {
        "class_names": ["Cu", "Fe"],
        "image_size": 64,
        "config": {
            "data": {"image_size": 64},
            "augmentation": {},
        },
    }
    image = Image.new("RGB", (120, 80), "white")
    results = predict_image(image, model, checkpoint, torch.device("cpu"), top_k=2)
    assert len(results) == 2
    assert {item["element"] for item in results} == {"Cu", "Fe"}
    assert abs(sum(float(item["probability"]) for item in results) - 1.0) < 1e-5

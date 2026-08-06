# Atomic Spectrum CNN — GitHub Copilot Ready

A ready-to-run atomic spectrum image classification project that can be uploaded directly to GitHub and further developed using GitHub Copilot.

This repository includes:

* A lightweight and runnable CNN model
* An optional ResNet18 transfer-learning model
* A five-class classification framework for Fe, Cu, Na, Ca, and Mg
* A synthetic spectrum image generator
* Training, validation, testing, and single-image prediction scripts
* A Streamlit interface for uploading and classifying images
* Repository-level instructions for GitHub Copilot
* A dedicated Spectrum AI Engineer agent for Copilot
* GitHub Actions tests
* A GitHub Codespaces development environment
* A Google Colab starter notebook

> **Important:** The synthetic data is provided only to demonstrate that the complete program workflow functions correctly. It must not be used to support scientific conclusions. Formal research must use verified, real atomic spectrum images.

## 1. Quick Start

Python 3.11 is required.

Create a virtual environment:

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS/Linux

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Generate synthetic data:

```bash
python scripts/generate_synthetic_data.py --config configs/quickstart.yaml
```

Train the first CNN model:

```bash
python scripts/train.py --config configs/quickstart.yaml
```

Evaluate the model:

```bash
python scripts/evaluate.py \
  --config configs/quickstart.yaml \
  --checkpoint outputs/quickstart/best_model.pt
```

Predict a single image:

```bash
python scripts/predict.py \
  --config configs/quickstart.yaml \
  --checkpoint outputs/quickstart/best_model.pt \
  --image data/synthetic/test/Fe/Fe_0000.png
```

Launch the web application:

```bash
streamlit run app/streamlit_app.py
```

## 2. Using ResNet18

First generate the dataset, and then run:

```bash
python scripts/train.py --config configs/resnet18.yaml
```

The `configs/resnet18.yaml` file disables pretrained-weight downloads by default so that the project can run in an offline environment.

When an internet connection is available, change:

```yaml
model:
  pretrained: false
```

to:

```yaml
model:
  pretrained: true
```

This enables the use of ImageNet pretrained weights.

## 3. Using Real Data

Real spectrum images should follow the PyTorch `ImageFolder` directory structure:

```text
data/real/
├── train/
│   ├── Fe/
│   ├── Cu/
│   ├── Na/
│   ├── Ca/
│   └── Mg/
├── validation/
│   ├── Fe/
│   ├── Cu/
│   ├── Na/
│   ├── Ca/
│   └── Mg/
└── test/
    ├── Fe/
    ├── Cu/
    ├── Na/
    ├── Ca/
    └── Mg/
```

Copy:

```text
configs/real_data.yaml.example
```

to:

```text
configs/real_data.yaml
```

Update the dataset paths, and then start training:

```bash
python scripts/train.py --config configs/real_data.yaml
```

## 4. Using GitHub Copilot

After the repository has been uploaded to GitHub, Copilot will automatically read the following files:

```text
.github/copilot-instructions.md
AGENTS.md
.github/agents/spectrum-ai-engineer.agent.md
```

Open GitHub Copilot Agent, select this repository, and enter:

```text
Read COPILOT_START_HERE.md and complete the next unfinished milestone.
Run the tests before and after editing. Do not claim success unless the
commands actually pass. Preserve the scientific meaning of wavelength position.
```

You can also copy the contents of:

```text
.github/prompts/build-spectrum-ai.prompt.md
```

and send them directly to Copilot as a development task.

## 5. Project Structure

```text
atomic-spectrum-cnn-copilot/
├── .github/
│   ├── agents/
│   ├── prompts/
│   ├── workflows/
│   └── copilot-instructions.md
├── app/
├── configs/
├── notebooks/
├── scripts/
├── src/atomic_spectrum_ai/
├── tests/
├── AGENTS.md
├── COPILOT_START_HERE.md
├── requirements.txt
└── README.md
```

## 6. Scientific Limitations

The first version of this project performs **single-label classification**: each image is assumed to represent only one element.

The model must not be directly interpreted as a system for qualitative elemental analysis of mixed samples.

For formal research, care must be taken to prevent the model from learning non-scientific information, including:

* Element names displayed in image titles or legends
* File names
* Instrument software interfaces
* Class-specific background colours
* Image dimensions that are fixed for particular classes
* Images from different instruments being assigned to different classes

The horizontal position in a spectrum represents wavelength. Therefore, horizontal flipping must not be used as a data augmentation method.

## 7. Code Quality Checks

Run the tests:

```bash
pytest -q
```

Run the linter:

```bash
ruff check .
```

## 8. Outputs

Training outputs are saved in:

```text
outputs/<run_name>/
├── best_model.pt
├── class_names.json
├── history.json
├── training_curves.png
├── confusion_matrix.png
└── metrics.json
```

The output files include the best model checkpoint, class-name mapping, training history, training curves, confusion matrix, and evaluation metrics.

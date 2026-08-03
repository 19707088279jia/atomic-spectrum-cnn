# Atomic Spectrum CNN — GitHub Copilot Ready

一个可直接上传到 GitHub、交给 GitHub Copilot 继续开发的原子光谱图像分类项目。

本仓库已经包含：

- 可运行的小型 CNN；
- 可选的 ResNet18 迁移学习模型；
- Fe、Cu、Na、Ca、Mg 五分类框架；
- 模拟光谱图片生成器；
- 训练、验证、测试和单图预测脚本；
- Streamlit 上传图片演示界面；
- GitHub Copilot 仓库指令；
- Copilot 专用 Spectrum AI Engineer agent；
- GitHub Actions 测试；
- Codespaces 开发环境；
- Google Colab 启动 notebook。

> 重要：模拟数据仅用于证明程序流程能够运行，不能作为最终科学结论。正式研究必须换成经过确认的真实原子光谱图。

## 1. 最快启动

要求 Python 3.11。

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

生成模拟数据：

```bash
python scripts/generate_synthetic_data.py --config configs/quickstart.yaml
```

训练第一版 CNN：

```bash
python scripts/train.py --config configs/quickstart.yaml
```

测试模型：

```bash
python scripts/evaluate.py --config configs/quickstart.yaml --checkpoint outputs/quickstart/best_model.pt
```

预测单张图片：

```bash
python scripts/predict.py \
  --config configs/quickstart.yaml \
  --checkpoint outputs/quickstart/best_model.pt \
  --image data/synthetic/test/Fe/Fe_0000.png
```

启动网页：

```bash
streamlit run app/streamlit_app.py
```

## 2. 使用 ResNet18

先生成数据，然后运行：

```bash
python scripts/train.py --config configs/resnet18.yaml
```

`configs/resnet18.yaml` 默认关闭预训练权重下载，保证离线环境也能运行。连接互联网后，可将：

```yaml
model:
  pretrained: true
```

改为 `true` 使用 ImageNet 预训练权重。

## 3. 使用真实数据

真实图片应采用 ImageFolder 目录格式：

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

复制 `configs/real_data.yaml.example` 为 `configs/real_data.yaml`，修改路径后训练：

```bash
python scripts/train.py --config configs/real_data.yaml
```

## 4. 在 GitHub Copilot 中使用

上传仓库后，Copilot 会自动读取：

- `.github/copilot-instructions.md`
- `AGENTS.md`
- `.github/agents/spectrum-ai-engineer.agent.md`

打开 GitHub Copilot Agent，选择本仓库，然后粘贴：

```text
Read COPILOT_START_HERE.md and complete the next unfinished milestone.
Run the tests before and after editing. Do not claim success unless the
commands actually pass. Preserve the scientific meaning of wavelength position.
```

也可以把 `.github/prompts/build-spectrum-ai.prompt.md` 的内容直接作为任务发送给 Copilot。

## 5. 项目结构

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

## 6. 科学限制

第一版任务是单标签分类：一张图片只对应一种元素。不要把它直接解释成混合样品的元素定性分析。

正式研究时必须避免模型根据以下非科学信息分类：

- 标题或图例中的元素名称；
- 文件名；
- 仪器界面；
- 特定背景颜色；
- 不同类别固定使用不同图片尺寸；
- 不同类别固定来自不同仪器。

水平位置代表波长，禁止使用水平翻转作为数据增强。

## 7. 运行检查

```bash
pytest -q
ruff check .
```

## 8. 输出

训练结果保存在：

```text
outputs/<run_name>/
├── best_model.pt
├── class_names.json
├── history.json
├── training_curves.png
├── confusion_matrix.png
└── metrics.json
```

## License

MIT

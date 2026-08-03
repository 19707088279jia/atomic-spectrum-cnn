# Copilot Start Here

## 项目目标

开发一个基于 CNN 的原子发射光谱图片分类系统。第一阶段只处理单元素、单标签图片，并分类为：

- Fe
- Cu
- Na
- Ca
- Mg

## 当前已经完成

- 模拟光谱数据生成；
- 小型 CNN；
- ResNet18 选项；
- 训练与测试；
- 单图预测；
- Streamlit 界面；
- CI、单元测试和配置文件。

## Copilot 下一步任务顺序

### Milestone 1：验证基础项目

1. 安装依赖。
2. 运行 `pytest -q`。
3. 运行模拟数据生成器。
4. 使用 quickstart 配置训练 1–2 个 epoch。
5. 确认 checkpoint 和 metrics 文件生成。
6. 修复所有错误，不要跳过测试。

### Milestone 2：真实数据接入

1. 新增真实光谱图片数据检查器。
2. 检测标题文字、图例、图片尺寸和类别数量。
3. 输出数据审计报告。
4. 禁止根据文件名向模型泄露标签。
5. 保持现有命令兼容。

### Milestone 3：模型可解释性

1. 完善 Grad-CAM。
2. 在 Streamlit 中叠加热力图。
3. 输出 Top-3 预测。
4. 明确 Grad-CAM 只能说明关注区域，不能证明化学因果关系。

### Milestone 4：研究级评估

1. 加入每类 precision、recall、F1。
2. 保存混淆矩阵。
3. 增加 bootstrap 置信区间。
4. 增加按实验批次拆分数据的接口。
5. 禁止同一原始光谱的增强版本跨 train/test。

### Milestone 5：多元素扩展

只有在单元素系统稳定后才开始：

1. 将单标签 softmax 改为多标签 sigmoid。
2. 使用 BCEWithLogitsLoss。
3. 每张图可包含多个元素。
4. 输出每种元素独立概率和阈值。
5. 保留旧单标签模式。

## 给 Copilot 的首条指令

```text
Work on Milestone 1 only. First inspect the repository and run the existing
unit tests. Then generate a very small synthetic dataset and perform a one-epoch
smoke training run. Fix any failures you encounter. Keep the public CLI stable,
add tests for every fix, and report the exact commands and results. Do not begin
Milestone 2 until Milestone 1 passes.
```

# Motion Generation using Qwen2.5

本项目基于 **Qwen2.5-1.5B-Instruct** 模型进行微调，旨在根据文本描述生成人体动作序列。

## 1. 环境准备

确保已安装 `uv` (Python 包管理器)，然后同步环境：

```bash
uv sync
```

或者手动安装依赖 (参考 `pyproject.toml`)：
```bash
pip install torch transformers peft datasets trl bitsandbytes matplotlib scipy
```

## 2. 数据准备

本项目使用 KIT-ML 数据集。数据处理流程如下：

### 2.1 原始数据转换 (prepare_data.ipynb)
这一步将原始 `.npy` 动作文件转换为离散化的 Token 序列。
- 降采样到 40 帧。
- 进行 Min-Max 归一化并量化为 0-255 的整数。
- 生成 `train_v1.jsonl`, `val_v1.jsonl`, `test_v1.jsonl`。

### 2.2 特殊 Token 转换 (prepare_data_v3.py)
为了让模型更好地理解动作数据，我们将数字序列转换为特殊 Token 格式（例如 `<157>`）。这可以大幅缩短序列长度并提升模型性能。

```bash
uv run prepare_data_v3.py
```
输出目录：`KIT-ML/qwen_ready_v3`

## 3. 模型微调 (Finetuning)

使用 QLoRA 技术对 Qwen2.5-1.5B 进行指令微调。

- **脚本**: `finetune_v2.py`
- **特点**:
    - 向 Tokenizer 添加了 `<0>` 到 `<255>` 的新 Token。
    - 训练了 Embedding 层和 LM Head 层以适应新 Token。
    - 开启了梯度检查点 (Gradient Checkpointing) 以节省显存。
    - 上下文长度优化为 2700。

运行微调：
```bash
uv run finetune_v2.py
```
输出目录：`Qwen-Motion-Finetuned-v2`

可以使用 TensorBoard 查看训练进度：
```bash
uv run tensorboard --logdir Qwen-Motion-Finetuned-v2/runs
```

## 4. 推理与可视化 (Inference)

使用微调后的模型生成动作并保存为 GIF 动画。

- **脚本**: `inference.py`
- **输出**: 默认保存在 `Generation/` 文件夹下。

基本用法：
```bash
uv run inference.py "A person walks forward"
```

指定输出文件名：
```bash
uv run inference.py "A person jumps high" --output Generation/jump.gif
```

不生成可视化（只保存 .npy 数据）：
```bash
uv run inference.py "A person sits down" --no_viz
```

## 5. 结果分析

可以使用 `analyze_motion.py` 对生成的 `.npy` 文件进行详细分析（速度、轨迹图）。

```bash
uv run analyze_motion.py Generation/generated_motion.npy
```
结果将保存在同级目录下，包含速度曲线和关节轨迹图。

## 目录结构

- `KIT-ML/`: 数据集目录
- `Qwen-Motion-Finetuned-v2/`: 微调后的模型权重 (Adapter)
- `Generation/`: 生成结果 (GIF, NPY, Analysis)
- `finetune_v2.py`: 微调脚本
- `inference.py`: 推理脚本
- `prepare_data_v3.py`: 数据预处理脚本
- `visualize_motion.py`: 动作可视化工具
- `analyze_motion.py`: 动作数据分析工具


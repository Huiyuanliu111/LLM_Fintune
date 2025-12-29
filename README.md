# Motion Generation using Qwen2.5

本项目基于 **Qwen2.5-1.5B-Instruct** 模型进行微调，旨在根据文本描述生成人体动作序列。

## 1. 环境准备

确保已安装 `uv` (Python 包管理器)，然后同步环境：

```bash
uv sync
```

或者手动安装依赖 (参考 `pyproject.toml`)：
```bash
pip install torch transformers peft datasets trl bitsandbytes matplotlib scipy wandb
```

## 2. 数据准备

本项目使用 KIT-ML 数据集。为了解决长序列显存溢出 (OOM) 问题，采用了降采样策略。

### 2.1 原始数据转换 (prepare_data.ipynb)
将原始 `.npy` 动作文件转换为离散化的 Token 序列。

### 2.2 特殊 Token 转换 v3 (prepare_data_v3.py)
生成 `<0>` 到 `<255>` 的特殊 Token 格式。

### 2.3 **[关键] 降采样处理 v4 (process_data_v4.py)**
为了大幅降低显存占用，我们将数据进行 **1/4 降采样**（每 4 帧取 1 帧）。
- 原始 40 帧 -> 10 帧 (约 630 tokens)
- 原始 120 帧 -> 30 帧 (约 1890 tokens)
这使得 `MAX_SEQ_LENGTH` 可以安全地设置为 **1024**，从而避免训练崩溃。

生成 v4 数据集：
```bash
uv run python process_data_v4.py
```
输出目录：`KIT-ML/qwen_ready_v4`

## 3. 模型微调 (Finetuning)

使用 QLoRA 技术对 Qwen2.5-1.5B 进行指令微调。脚本经过深度优化以适应消费级显卡。

- **脚本**: `finetune_v2.py`
- **数据集**: `KIT-ML/qwen_ready_v4`
- **核心配置**:
    - **显存优化**:
        - `MAX_SEQ_LENGTH = 1024` (配合 v4 数据)
        - `paged_adamw_8bit` 优化器 (节省 75% 优化器显存)
        - `gradient_accumulation_steps = 8`
    - **LoRA 配置**:
        - Rank = 8, Alpha = 32
        - Target Modules: `q_proj`, `v_proj`, `o_proj` (去掉了 MLP 层以省显存)
    - **训练策略**:
        - 向 Tokenizer 添加了 `<0>` 到 `<255>` 的新 Token。
        - **Gradient Hook**: 使用梯度钩子技术，**只更新新 Token 的 Embedding**，完全冻结原有的 15万+ 个词表向量。这既防止了灾难性遗忘，又让新 Token 能够被学习。

### 运行微调

1. (可选) 登录 WandB 以查看实时训练曲线：
   ```bash
   uv run wandb login
   ```

2. 开始训练：
   ```bash
   uv run finetune_v2.py
   ```
   输出目录：`Qwen-Motion-Finetuned-v2`

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
  - `qwen_ready_v4/`: 降采样后的训练数据 (JSONL)
- `Qwen-Motion-Finetuned-v2/`: 微调后的模型权重 (Adapter)
- `Generation/`: 生成结果 (GIF, NPY, Analysis)
- `finetune_v2.py`: **核心微调脚本**
- `process_data_v4.py`: **数据降采样脚本**
- `inference.py`: 推理脚本
- `prepare_data_v3.py`: 旧版数据预处理脚本
- `visualize_motion.py`: 动作可视化工具
- `analyze_motion.py`: 动作数据分析工具

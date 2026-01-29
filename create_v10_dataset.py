"""
创建 v10 数据集 - 使用全局 min/max quantize

修复 v6-v9 数据集的问题：之前每个文件用自己的 min/max quantize，
导致 tokens 含义不一致，无法正确 dequantize。

本脚本使用全局 min/max，确保 tokens 含义一致。
"""
import os
import json
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm

# 配置
DATA_DIR = Path("KIT-ML")
MOTION_DIR = DATA_DIR / "new_joints"  # 原始关节数据（未 quantize）
DOWNSAMPLE_DIR = DATA_DIR / "new_joints_40"  # 降采样后的数据
TEXT_DIR = DATA_DIR / "texts"
OUTPUT_DIR = DATA_DIR / "qwen_ready_v10"

# 全局范围（从所有 new_joints 数据计算得到）
V_MIN = -6429.9
V_MAX = 7001.1
BINS = 256

# 降采样配置
TARGET_FRAMES = 40


def compute_global_range(motion_dir: Path):
    """计算全局 min/max"""
    files = list(motion_dir.glob("*.npy"))
    global_min = float('inf')
    global_max = float('-inf')
    
    print(f"Computing global range from {len(files)} files...")
    for f in tqdm(files):
        data = np.load(f)
        global_min = min(global_min, data.min())
        global_max = max(global_max, data.max())
    
    print(f"Global min: {global_min}")
    print(f"Global max: {global_max}")
    return global_min, global_max


def uniform_downsample(motion: np.ndarray, target_frames: int = 40) -> np.ndarray:
    """均匀降采样"""
    T = motion.shape[0]
    if T <= target_frames:
        return motion
    idx = np.linspace(0, T - 1, target_frames).astype(int)
    return motion[idx]


def quantize_motion(motion: np.ndarray, v_min: float = V_MIN, v_max: float = V_MAX, bins: int = BINS) -> np.ndarray:
    """使用全局范围 quantize"""
    norm = (motion - v_min) / (v_max - v_min)
    quantized = np.round(norm * (bins - 1)).astype(np.int32)
    quantized = np.clip(quantized, 0, bins - 1)
    return quantized


def format_motion_to_tokens(motion: np.ndarray) -> str:
    """将 motion 数组转换为 <token> 格式的字符串"""
    flat = motion.flatten()
    tokens = "".join([f"<{int(v)}>" for v in flat])
    return tokens


def get_motion_ids(motion_dir: Path, text_dir: Path, max_frames: int = 120) -> list:
    """获取同时有 motion 和 text 的轨迹 ID，且帧数 <= max_frames"""
    motion_files = sorted(motion_dir.glob("*.npy"))
    motion_ids = set(f.stem for f in motion_files)
    
    text_files = sorted(text_dir.glob("*.txt"))
    text_ids = set(f.stem for f in text_files)
    
    common_ids = sorted(motion_ids & text_ids)
    
    # 过滤帧数
    valid_ids = []
    for motion_id in common_ids:
        motion_path = motion_dir / f"{motion_id}.npy"
        motion = np.load(motion_path)
        if motion.shape[0] <= max_frames:
            valid_ids.append(motion_id)
    
    print(f"Motion files: {len(motion_ids)}")
    print(f"Text files: {len(text_ids)}")
    print(f"Common: {len(common_ids)}")
    print(f"Valid (frames <= {max_frames}): {len(valid_ids)}")
    
    return valid_ids


def create_dataset_entry(motion_id: str) -> dict:
    """创建一条数据集条目"""
    # 加载原始 motion
    motion_path = MOTION_DIR / f"{motion_id}.npy"
    motion = np.load(motion_path)  # shape: (T, 21, 3)
    
    # 降采样
    motion = uniform_downsample(motion, TARGET_FRAMES)
    
    # 使用全局范围 quantize
    motion_q = quantize_motion(motion)
    
    # 加载 text
    text_path = TEXT_DIR / f"{motion_id}.txt"
    with open(text_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    text = first_line.split("#")[0]
    
    # 转换为 token 格式
    motion_tokens = format_motion_to_tokens(motion_q)
    
    # 构建 messages 格式
    messages = [
        {
            "role": "system",
            "content": "You are a motion generation assistant. Generate the motion sequence (flattened integer tokens) based on the text description."
        },
        {
            "role": "user", 
            "content": text
        },
        {
            "role": "assistant",
            "content": motion_tokens
        }
    ]
    
    return {"messages": messages}


def main():
    # 验证全局范围
    print("=== Verifying global range ===")
    computed_min, computed_max = compute_global_range(MOTION_DIR)
    print(f"Using V_MIN={V_MIN}, V_MAX={V_MAX}")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 获取所有可配对的轨迹 ID
    print("\n=== Collecting motion IDs ===")
    motion_ids = get_motion_ids(MOTION_DIR, TEXT_DIR)
    print(f"Using {len(motion_ids)} motion-text pairs")
    
    # 收集数据
    print("\n=== Processing data ===")
    records = []
    for motion_id in tqdm(motion_ids):
        try:
            entry = create_dataset_entry(motion_id)
            records.append(entry)
        except Exception as e:
            print(f"Failed to load {motion_id}: {e}")
    
    print(f"\nTotal: {len(records)} records")
    
    # 划分 train/val/test (80/10/10)
    n_train = int(len(records) * 0.8)
    n_val = int(len(records) * 0.1)
    
    train_records = records[:n_train]
    val_records = records[n_train:n_train + n_val]
    test_records = records[n_train + n_val:]
    
    print(f"Split: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")
    
    # 写入 JSONL 文件
    splits = [
        ("train_v10.jsonl", train_records),
        ("val_v10.jsonl", val_records),
        ("test_v10.jsonl", test_records),
    ]
    
    for filename, data in splits:
        path = OUTPUT_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved: {path} ({len(data)} records)")
    
    # 保存配置
    config = {
        "V_MIN": V_MIN,
        "V_MAX": V_MAX,
        "BINS": BINS,
        "TARGET_FRAMES": TARGET_FRAMES,
        "num_samples": len(records),
    }
    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")
    
    # 打印示例
    print("\n=== Example Entry ===")
    example = records[0]
    print(f"Text: {example['messages'][1]['content']}")
    print(f"Motion tokens (first 100 chars): {example['messages'][2]['content'][:100]}...")


if __name__ == "__main__":
    main()

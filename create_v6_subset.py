"""
创建 qwen_ready_v6 子集数据集
包含前 500 条轨迹
"""
import os
import json
import numpy as np
from pathlib import Path

# 配置
DATA_DIR = Path("KIT-ML")
MOTION_DIR = DATA_DIR / "new_joints_40_bin256"
TEXT_DIR = DATA_DIR / "texts"
OUTPUT_DIR = DATA_DIR / "qwen_ready_v7"

# 选取数量（设为 None 表示全部）
NUM_SAMPLES = None  # 改为 None 获取所有可配对的数据

def get_motion_ids(motion_dir: Path, text_dir: Path, num_samples: int = None) -> list:
    """获取同时有 motion 和 text 的轨迹 ID"""
    # 获取所有 .npy 文件
    motion_files = sorted(motion_dir.glob("*.npy"))
    motion_ids = set(f.stem for f in motion_files)
    
    # 获取所有 .txt 文件
    text_files = sorted(text_dir.glob("*.txt"))
    text_ids = set(f.stem for f in text_files)
    
    # 取交集（同时有 motion 和 text 的）
    common_ids = sorted(motion_ids & text_ids)
    
    print(f"Motion files: {len(motion_ids)}")
    print(f"Text files: {len(text_ids)}")
    print(f"Common (can be paired): {len(common_ids)}")
    
    # 取前 N 条（如果指定了的话）
    if num_samples is not None:
        return common_ids[:num_samples]
    return common_ids

def format_motion_to_tokens(motion: np.ndarray) -> str:
    """将 motion 数组转换为 <token> 格式的字符串"""
    # motion shape: (T, 21, 3)
    flat = motion.flatten()
    tokens = "".join([f"<{int(v)}>" for v in flat])
    return tokens

def create_dataset_entry(motion_id: str) -> dict:
    """创建一条数据集条目"""
    # 加载 motion
    motion_path = MOTION_DIR / f"{motion_id}.npy"
    motion = np.load(motion_path)  # shape: (T, 21, 3)
    
    # 加载 text
    text_path = TEXT_DIR / f"{motion_id}.txt"
    with open(text_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    text = first_line.split("#")[0]
    
    # 转换为 token 格式
    motion_tokens = format_motion_to_tokens(motion)
    
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
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 获取所有可配对的轨迹 ID
    motion_ids = get_motion_ids(MOTION_DIR, TEXT_DIR, NUM_SAMPLES)
    print(f"Using {len(motion_ids)} motion-text pairs")
    
    # 收集数据
    records = []
    for i, motion_id in enumerate(motion_ids):
        try:
            entry = create_dataset_entry(motion_id)
            records.append(entry)
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(motion_ids)} records...")
        except Exception as e:
            print(f"✗ Failed to load {motion_id}: {e}")
    
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
        ("train_v6.jsonl", train_records),
        ("val_v6.jsonl", val_records),
        ("test_v6.jsonl", test_records),
    ]
    
    for filename, data in splits:
        path = OUTPUT_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved: {path} ({len(data)} records)")
    
    # 打印示例
    print("\n=== Example Entry ===")
    example = records[0]
    print(f"Text: {example['messages'][1]['content']}")
    print(f"Motion tokens (first 100 chars): {example['messages'][2]['content'][:100]}...")
    print(f"Motion tokens length: {len(example['messages'][2]['content'])} chars")

if __name__ == "__main__":
    main()


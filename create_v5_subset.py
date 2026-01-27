"""
创建 qwen_ready_v5 子集数据集
只包含 10 条指定的原始轨迹: 00001, 00002, 00003, 00004, 00005, 00006, 00008, 00010, 00013, 00014
"""
import os
import json
import numpy as np
from pathlib import Path

# 配置
DATA_DIR = Path("KIT-ML")
MOTION_DIR = DATA_DIR / "new_joints_40_bin256"
TEXT_DIR = DATA_DIR / "texts"
OUTPUT_DIR = DATA_DIR / "qwen_ready_v5"

# 指定的 10 条轨迹 ID
SELECTED_IDS = ["00001", "00002", "00003", "00004", "00005", "00006", "00008", "00010", "00013", "00014"]

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
    
    # 收集数据
    records = []
    for motion_id in SELECTED_IDS:
        try:
            entry = create_dataset_entry(motion_id)
            records.append(entry)
            print(f"✓ Loaded {motion_id}: {entry['messages'][1]['content'][:50]}...")
        except Exception as e:
            print(f"✗ Failed to load {motion_id}: {e}")
    
    print(f"\nTotal: {len(records)} records")
    
    # 写入 JSONL 文件
    # 为了 overfit 测试，train 和 val 使用相同的数据
    train_path = OUTPUT_DIR / "train_v5.jsonl"
    val_path = OUTPUT_DIR / "val_v5.jsonl"
    test_path = OUTPUT_DIR / "test_v5.jsonl"
    
    for path in [train_path, val_path, test_path]:
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved: {path}")
    
    # 打印示例
    print("\n=== Example Entry ===")
    example = records[0]
    print(f"Text: {example['messages'][1]['content']}")
    print(f"Motion tokens (first 100 chars): {example['messages'][2]['content'][:100]}...")
    print(f"Motion tokens length: {len(example['messages'][2]['content'])} chars")

if __name__ == "__main__":
    main()


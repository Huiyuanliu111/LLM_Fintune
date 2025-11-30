
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- 配置 ---
DATA_DIR = Path("KIT-ML")
MOTION_DIR = DATA_DIR / "new_joints_40_bin256"
TEXT_DIR = DATA_DIR / "texts"
OUTPUT_DIR = DATA_DIR / "qwen_ready"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_split(split_name):
    split_file = DATA_DIR / f"{split_name}.txt"
    if not split_file.exists():
        print(f"Warning: {split_file} not found, skipping.")
        return

    id_list = split_file.read_text(encoding='utf-8').strip().splitlines()
    
    output_path = OUTPUT_DIR / f"{split_name}_full.jsonl"
    print(f"Processing {split_name} ({len(id_list)} IDs)...")

    count = 0
    with open(output_path, "w", encoding="utf-8") as f_out:
        for motion_id in tqdm(id_list):
            motion_path = MOTION_DIR / f"{motion_id}.npy"
            text_path = TEXT_DIR / f"{motion_id}.txt"
            
            # 1. 检查文件是否存在
            if not motion_path.exists():
                # print(f"Warning: Motion file {motion_path} not found.")
                continue
            if not text_path.exists():
                print(f"Warning: Text file {text_path} not found.")
                continue
                
            # 2. 加载动作数据
            try:
                motion_data = np.load(motion_path) # Shape: (T, 21, 3) 或类似
            except Exception as e:
                print(f"Error loading {motion_path}: {e}")
                continue

            # 将动作数据展平并转换为 token 字符串 (例如: "12 45 255 ...")
            # 注意：这里假设 motion_data 已经是量化好的整数 (0-255)
            # 如果你的量化逻辑不同，请调整。
            # 根据之前的 notebook，数据是 (T, 21, 3)，范围 0-255
            
            flat_motion = motion_data.flatten().tolist()
            motion_tokens_str = " ".join(map(str, flat_motion))

            # 3. 加载文本数据 (读取所有行)
            try:
                with open(text_path, "r", encoding="utf-8") as f_txt:
                    lines = f_txt.readlines()
            except Exception as e:
                print(f"Error reading {text_path}: {e}")
                continue
                
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 解析文本: "A person walks forward.#A/DET ...#0.0#0.0"
                # 我们只需要第一部分
                parts = line.split("#")
                if not parts:
                    continue
                
                text_desc = parts[0].strip()
                if not text_desc:
                    continue

                # 4. 构建对话格式 (Qwen Chat template)
                # User: 文本描述
                # Assistant: <motion_start> tokens <motion_end>
                
                # 为了方便后续解析，建议加上特殊标记，或者直接输出数字序列
                # 这里我们简单地输出数字序列
                
                messages = [
                    {"role": "system", "content": "You are a motion generation assistant. Generate motion tokens based on the text description."},
                    {"role": "user", "content": text_desc},
                    {"role": "assistant", "content": motion_tokens_str}
                ]
                
                # 写入 JSONL
                json_line = json.dumps({"messages": messages}, ensure_ascii=False)
                f_out.write(json_line + "\n")
                count += 1

    print(f"Saved {count} samples to {output_path}")

def main():
    # 确保动作目录存在
    if not MOTION_DIR.exists():
        print(f"Error: Motion directory {MOTION_DIR} not found. Please run the quantization step first.")
        return

    for split in ["train", "val", "test"]:
        process_split(split)

if __name__ == "__main__":
    main()


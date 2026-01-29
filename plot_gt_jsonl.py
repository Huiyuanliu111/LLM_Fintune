import numpy as np
import os
import json
import argparse
import re
from visualize_motion import parse_motion_tokens, plot_motion

# Constants (Must match inference.py)
BINS = 256
V_MIN = -3000.0
V_MAX = 3000.0

def dequantize(motion_data, bins=BINS, v_min=V_MIN, v_max=V_MAX):
    """
    反量化：将 [0, bins-1] 的整数映射回 [v_min, v_max]
    """
    norm = motion_data.astype(float) / (bins - 1)
    v_range = v_max - v_min
    denorm_data = (norm * v_range) + v_min
    return denorm_data

def extract_tokens_from_jsonl(jsonl_path, index=0, search_text=None):
    """
    从 jsonl 文件中提取指定索引或匹配文本的 assistant tokens
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    target_line = None
    if search_text:
        for line in lines:
            data = json.loads(line)
            user_prompt = data["messages"][1]["content"]
            if search_text.lower() in user_prompt.lower():
                target_line = data
                print(f"Found match: '{user_prompt}'")
                break
    else:
        if 0 <= index < len(lines):
            target_line = json.loads(lines[index])
            print(f"Using entry at index {index}: '{target_line['messages'][1]['content']}'")
    
    if not target_line:
        print("Error: Could not find matching entry in JSONL.")
        return None
    
    # 提取 assistant 的回复内容
    token_str = target_line["messages"][2]["content"]
    # 将 <33><114> 转换为 "33 114"
    tokens = re.findall(r'\d+', token_str)
    return " ".join(tokens)

def main():
    parser = argparse.ArgumentParser(description="Plot ground truth motion from downsampled JSONL dataset")
    parser.add_argument("--index", type=int, default=0, help="Index of the entry in JSONL (default: 0)")
    parser.add_argument("--search", type=str, default=None, help="Search for prompt text in JSONL")
    parser.add_argument("--file", type=str, default="KIT-ML/qwen_ready_v5/train_v5.jsonl", help="Path to the JSONL file")
    parser.add_argument("--output", type=str, default=None, help="Output filename for the GIF")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found at {args.file}")
        return

    print(f"Reading from: {args.file}")
    clean_token_str = extract_tokens_from_jsonl(args.file, args.index, args.search)
    
    if not clean_token_str:
        return

    # 解析为 (T, 21, 3)
    motion_tokens = parse_motion_tokens(clean_token_str)
    print(f"Parsed motion shape: {motion_tokens.shape}")

    # 反量化
    print("Dequantizing motion...")
    motion_data = dequantize(motion_tokens)

    # 确定输出文件名
    if args.output:
        output_file = args.output
    else:
        output_dir = "Generation/gt"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        tag = f"idx{args.index}" if not args.search else "search"
        output_file = os.path.join(output_dir, f"gt_jsonl_{tag}.gif")

    # 绘图
    plot_motion(motion_data, output_file=output_file)
    print(f"Ground truth (from JSONL) visualization saved to: {output_file}")

if __name__ == "__main__":
    main()

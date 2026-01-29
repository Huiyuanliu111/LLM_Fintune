"""检查各版本训练数据的序列长度"""
import os
import json
import re
import numpy as np

def count_motion_tokens(content):
    """统计 <N> 格式的 token 数量"""
    matches = re.findall(r"<(\d+)>", content)
    return len(matches)

def analyze_jsonl(file_path):
    """分析 JSONL 文件的序列长度"""
    if not os.path.exists(file_path):
        return None
    
    lengths = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            for msg in data.get("messages", []):
                if msg["role"] == "assistant":
                    token_count = count_motion_tokens(msg["content"])
                    lengths.append(token_count)
                    break
    return lengths

def main():
    print("=" * 70)
    print("SEQUENCE LENGTH ANALYSIS FOR EACH VERSION")
    print("=" * 70)
    
    versions = ["v5", "v6", "v7", "v8", "v9"]
    
    for version in versions:
        data_dir = f"KIT-ML/qwen_ready_{version}"
        train_file = os.path.join(data_dir, f"train_{version}.jsonl")
        
        lengths = analyze_jsonl(train_file)
        
        if lengths is None:
            print(f"\n{version}: Data not found ({train_file})")
            continue
        
        lengths = np.array(lengths)
        
        print(f"\n{version.upper()}:")
        print(f"  File: {train_file}")
        print(f"  Samples: {len(lengths)}")
        print(f"  Token counts:")
        print(f"    Min:    {lengths.min()}")
        print(f"    Max:    {lengths.max()}")
        print(f"    Mean:   {lengths.mean():.1f}")
        print(f"    Median: {np.median(lengths):.1f}")
        
        # 计算帧数 (每帧 63 tokens = 21 joints * 3 coords)
        frames = lengths / 63
        print(f"  Frame counts (tokens / 63):")
        print(f"    Min:    {frames.min():.1f}")
        print(f"    Max:    {frames.max():.1f}")
        print(f"    Mean:   {frames.mean():.1f}")
        
        # 检查是否有截断
        unique_lengths = np.unique(lengths)
        if len(unique_lengths) == 1:
            print(f"  NOTE: All samples have SAME length -> likely truncated/padded")
        elif len(unique_lengths) <= 5:
            print(f"  Unique lengths: {sorted(unique_lengths)}")
    
    print("\n" + "=" * 70)
    print("SUMMARY: Expected tokens for different frame counts")
    print("=" * 70)
    for frames in [10, 20, 30, 40]:
        tokens = frames * 63
        print(f"  {frames} frames = {tokens} tokens")

if __name__ == "__main__":
    main()

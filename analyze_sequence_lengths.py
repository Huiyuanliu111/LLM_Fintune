import json
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def count_tokens_in_content(content):
    """统计内容中的token数量"""
    matches = re.findall(r"<(\d+)>", content)
    return len(matches)

def analyze_file(file_path, label):
    """分析一个文件的序列长度分布"""
    lengths = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                for msg in messages:
                    if msg["role"] == "assistant":
                        content = msg["content"]
                        token_count = count_tokens_in_content(content)
                        if token_count > 0:  # 只统计非空序列
                            lengths.append(token_count)
                        break
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num + 1} in {file_path}: {e}")
                continue

    return lengths

def plot_histograms(original_lengths, downsampled_lengths):
    """画直方图比较分布"""
    plt.figure(figsize=(15, 6))

    # 子图1: 原始序列长度分布
    plt.subplot(1, 2, 1)
    plt.hist(original_lengths, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Original Train Sequence Lengths')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    plt.axvline(np.mean(original_lengths), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(original_lengths):.1f}')
    plt.axvline(np.median(original_lengths), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(original_lengths):.1f}')
    plt.legend()

    # 子图2: 降采样后序列长度分布
    plt.subplot(1, 2, 2)
    plt.hist(downsampled_lengths, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Downsampled Train Sequence Lengths')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    plt.axvline(np.mean(downsampled_lengths), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(downsampled_lengths):.1f}')
    plt.axvline(np.median(downsampled_lengths), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(downsampled_lengths):.1f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig('sequence_length_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印统计信息
    print("=" * 60)
    print("ORIGINAL TRAIN SEQUENCES STATISTICS:")
    print("=" * 60)
    print(f"Total sequences: {len(original_lengths)}")
    print(f"Mean length: {np.mean(original_lengths):.1f} tokens")
    print(f"Median length: {np.median(original_lengths):.1f} tokens")
    print(f"Min length: {np.min(original_lengths)} tokens")
    print(f"Max length: {np.max(original_lengths)} tokens")
    print(f"Std deviation: {np.std(original_lengths):.1f} tokens")

    print("\n" + "=" * 60)
    print("DOWNSAMPLED TRAIN SEQUENCES STATISTICS:")
    print("=" * 60)
    print(f"Total sequences: {len(downsampled_lengths)}")
    print(f"Mean length: {np.mean(downsampled_lengths):.1f} tokens")
    print(f"Median length: {np.median(downsampled_lengths):.1f} tokens")
    print(f"Min length: {np.min(downsampled_lengths)} tokens")
    print(f"Max length: {np.max(downsampled_lengths)} tokens")
    print(f"Std deviation: {np.std(downsampled_lengths):.1f} tokens")

    # 计算压缩比例
    ratio = np.mean(downsampled_lengths) / np.mean(original_lengths)
    print(f"\nCompression ratio: {ratio:.3f} ({1/ratio:.1f}x compression)")
def main():
    print("Analyzing sequence lengths...")

    # 只分析训练数据
    print("Processing original train data (v3)...")
    original_lengths = []
    train_v3_file = 'train_v3.jsonl'
    file_path = f"KIT-ML/qwen_ready_v3/{train_v3_file}"
    lengths = analyze_file(file_path, f"Original {train_v3_file}")
    original_lengths.extend(lengths)
    print(f"  {train_v3_file}: {len(lengths)} sequences")

    # 分析降采样后的训练数据
    print("\nProcessing downsampled train data (v4)...")
    downsampled_lengths = []
    train_v4_file = 'train_v4.jsonl'
    file_path = f"KIT-ML/qwen_ready_v4/{train_v4_file}"
    lengths = analyze_file(file_path, f"Downsampled {train_v4_file}")
    downsampled_lengths.extend(lengths)
    print(f"  {train_v4_file}: {len(lengths)} sequences")

    # 画直方图
    print("\nGenerating histograms...")
    plot_histograms(original_lengths, downsampled_lengths)

    print("\nAnalysis complete! Histogram saved as 'sequence_length_comparison.png'")

if __name__ == "__main__":
    main()

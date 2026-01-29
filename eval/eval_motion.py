"""
Motion Generation Evaluation Script

评估 LLM 微调模型生成的动作质量。

用法:
    python -m eval.eval_motion --gen_dir Generation/v9 --gt_dir KIT-ML/test_motions
    python -m eval.eval_motion --gen_npy path/to/generated.npy --gt_npy path/to/groundtruth.npy
"""

import os
import sys
import json
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.evaluator import MotionEvaluator, run_evaluation_with_replications
from eval.metrics import calculate_motion_statistics


def load_motion_from_npy(npy_path: str) -> np.ndarray:
    """
    从 .npy 文件加载动作数据
    
    Args:
        npy_path: .npy 文件路径
    
    Returns:
        motion: (T, J, 3) 动作数据
    """
    motion = np.load(npy_path)
    
    # 检查并修正形状
    if len(motion.shape) == 2:
        # 假设是 (T*J*3,) 或 (T, J*3)
        num_joints = 21
        if motion.shape[1] == num_joints * 3:
            T = motion.shape[0]
            motion = motion.reshape(T, num_joints, 3)
        else:
            # 尝试 reshape
            total = motion.size
            if total % (num_joints * 3) == 0:
                T = total // (num_joints * 3)
                motion = motion.reshape(T, num_joints, 3)
    
    return motion


def load_motions_from_directory(
    directory: str, 
    pattern: str = "*.npy"
) -> Tuple[List[np.ndarray], List[str]]:
    """
    从目录加载所有动作文件
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式
    
    Returns:
        motions: 动作列表
        filenames: 文件名列表
    """
    npy_files = sorted(glob(os.path.join(directory, pattern)))
    
    motions = []
    filenames = []
    
    for npy_file in tqdm(npy_files, desc="Loading motions"):
        try:
            motion = load_motion_from_npy(npy_file)
            if motion.shape[0] > 0:  # 确保有帧
                motions.append(motion)
                filenames.append(os.path.basename(npy_file))
        except Exception as e:
            print(f"Warning: Failed to load {npy_file}: {e}")
    
    return motions, filenames


def load_gt_from_jsonl(
    jsonl_path: str,
    motion_dir: Optional[str] = None
) -> Tuple[List[np.ndarray], List[str]]:
    """
    从 JSONL 文件加载 ground truth 数据
    
    如果 JSONL 包含动作文件路径，则从对应文件加载动作。
    
    Args:
        jsonl_path: JSONL 文件路径
        motion_dir: 动作文件所在目录 (可选)
    
    Returns:
        motions: 动作列表
        texts: 文本描述列表
    """
    motions = []
    texts = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GT from JSONL"):
            data = json.loads(line)
            
            # 提取文本
            messages = data.get('messages', [])
            text = None
            for msg in messages:
                if msg['role'] == 'user':
                    text = msg['content']
                    break
            
            if text:
                texts.append(text)
            
            # 如果有动作路径，加载动作
            motion_path = data.get('motion_path')
            if motion_path and motion_dir:
                full_path = os.path.join(motion_dir, motion_path)
                if os.path.exists(full_path):
                    try:
                        motion = load_motion_from_npy(full_path)
                        motions.append(motion)
                    except:
                        pass
    
    return motions, texts


def generate_motions_batch(
    model,
    tokenizer,
    texts: List[str],
    num_per_text: int = 1,
    max_new_tokens: int = 1024,
    dequantize_fn=None,
) -> List[List[np.ndarray]]:
    """
    批量生成动作
    
    Args:
        model: 生成模型
        tokenizer: tokenizer
        texts: 文本列表
        num_per_text: 每个文本生成的动作数量
        max_new_tokens: 最大生成 token 数
        dequantize_fn: 反量化函数
    
    Returns:
        motions_per_text: [[motion1, motion2, ...], ...]
    """
    import torch
    import re
    
    motions_per_text = []
    
    for text in tqdm(texts, desc="Generating motions"):
        text_motions = []
        
        for _ in range(num_per_text):
            messages = [
                {"role": "system", "content": "You are a motion generation assistant. Generate the motion sequence (flattened integer tokens) based on the text description."},
                {"role": "user", "content": text}
            ]
            
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=num_per_text > 1,  # 多次生成时使用采样
                    temperature=0.7 if num_per_text > 1 else 1.0,
                    top_p=0.9 if num_per_text > 1 else 1.0,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 解析动作
            try:
                tokens = re.findall(r'\d+', response)
                if tokens:
                    motion_flat = np.array([int(t) for t in tokens])
                    
                    # reshape 为 (T, 21, 3)
                    frame_size = 21 * 3
                    valid_len = (len(motion_flat) // frame_size) * frame_size
                    if valid_len > 0:
                        motion = motion_flat[:valid_len].reshape(-1, 21, 3)
                        
                        # 反量化
                        if dequantize_fn:
                            motion = dequantize_fn(motion)
                        
                        text_motions.append(motion)
            except Exception as e:
                print(f"Warning: Failed to parse motion: {e}")
        
        motions_per_text.append(text_motions)
    
    return motions_per_text


def main():
    parser = argparse.ArgumentParser(description="Evaluate motion generation quality")
    
    # 数据路径
    parser.add_argument("--gen_dir", type=str, help="Directory containing generated .npy files")
    parser.add_argument("--gen_npy", type=str, help="Single generated .npy file")
    parser.add_argument("--gt_dir", type=str, help="Directory containing ground truth .npy files")
    parser.add_argument("--gt_npy", type=str, help="Single ground truth .npy file")
    parser.add_argument("--gt_jsonl", type=str, help="JSONL file with ground truth data")
    parser.add_argument("--motion_dir", type=str, help="Directory for motion files referenced in JSONL")
    
    # 评估参数
    parser.add_argument("--num_joints", type=int, default=21, help="Number of joints (default: 21 for KIT-ML)")
    parser.add_argument("--feature_dim", type=int, default=256, help="Feature dimension")
    parser.add_argument("--diversity_times", type=int, default=300, help="Diversity sampling times")
    parser.add_argument("--replication_times", type=int, default=1, help="Number of evaluation replications")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # 输出
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 加载生成的动作
    gen_motions = []
    gen_filenames = []
    
    if args.gen_dir:
        gen_motions, gen_filenames = load_motions_from_directory(args.gen_dir)
    elif args.gen_npy:
        motion = load_motion_from_npy(args.gen_npy)
        gen_motions = [motion]
        gen_filenames = [os.path.basename(args.gen_npy)]
    
    if not gen_motions:
        print("Error: No generated motions loaded!")
        print("Please provide --gen_dir or --gen_npy")
        return
    
    print(f"Loaded {len(gen_motions)} generated motions")
    
    # 加载 ground truth 动作
    gt_motions = []
    gt_texts = []
    
    if args.gt_dir:
        gt_motions, _ = load_motions_from_directory(args.gt_dir)
    elif args.gt_npy:
        motion = load_motion_from_npy(args.gt_npy)
        gt_motions = [motion]
    elif args.gt_jsonl:
        gt_motions, gt_texts = load_gt_from_jsonl(args.gt_jsonl, args.motion_dir)
    
    if not gt_motions:
        print("Warning: No ground truth motions loaded!")
        print("FID calculation will be skipped.")
        # 使用生成的动作作为 GT (只计算 diversity)
        gt_motions = gen_motions[:len(gen_motions)//2] if len(gen_motions) > 1 else gen_motions
    
    print(f"Loaded {len(gt_motions)} ground truth motions")
    
    # 初始化评估器
    evaluator = MotionEvaluator(
        num_joints=args.num_joints,
        feature_dim=args.feature_dim,
        diversity_times=args.diversity_times,
        seed=args.seed,
    )
    
    # 运行评估
    if args.replication_times > 1:
        results = run_evaluation_with_replications(
            evaluator=evaluator,
            gt_motions=gt_motions,
            gen_motions=gen_motions,
            texts=gt_texts if gt_texts else None,
            replication_times=args.replication_times,
        )
        # 转换格式用于保存
        save_results = {
            k: {"mean": v[0], "confidence_interval": v[1]} 
            for k, v in results.items()
        }
    else:
        results = evaluator.evaluate_all(
            gt_motions=gt_motions,
            gen_motions=gen_motions,
            texts=gt_texts if gt_texts else None,
            compute_mm=False,
        )
        save_results = results
    
    # 保存结果
    evaluator.save_results(save_results, args.output)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

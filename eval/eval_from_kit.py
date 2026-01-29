"""
从 KIT-ML 原始数据直接评估 LLM 模型

用法:
    uv run python -m eval.eval_from_kit --num_samples 20
"""

import os
import sys
import json
import argparse
import re
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from eval.evaluator import MotionEvaluator

# 配置常量
BASE_MODEL_NAME = "Qwen/Qwen3-0.6B"
BINS = 256
V_MIN = -3000.0
V_MAX = 3000.0
NUM_JOINTS = 21

# KIT-ML 数据路径
KIT_ML_DIR = "KIT-ML"
MOTION_DIR = "new_joints_40_bin256"  # 量化后的数据
TEXT_DIR = "texts"
TEST_FILE = "test.txt"
DOWNSAMPLE_RATE = 4  # 训练时的下采样率


def downsample_motion(motion: np.ndarray, rate: int = DOWNSAMPLE_RATE) -> np.ndarray:
    """下采样动作数据，与训练数据保持一致"""
    return motion[::rate]


def dequantize(motion_data: np.ndarray, bins=BINS, v_min=V_MIN, v_max=V_MAX) -> np.ndarray:
    """反量化：将 [0, bins-1] 的整数映射回 [v_min, v_max]"""
    norm = motion_data.astype(float) / (bins - 1)
    v_range = v_max - v_min
    denorm_data = (norm * v_range) + v_min
    return denorm_data


def load_model(adapter_path: str, use_4bit: bool = True):
    """加载模型和 tokenizer"""
    print(f"Loading tokenizer from {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    
    print("Adding new tokens <0>...<255>...")
    new_tokens = [f"<{i}>" for i in range(256)]
    tokenizer.add_tokens(new_tokens)
    
    if use_4bit:
        print("Loading base model with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        print("Loading base model with fp16...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    return model, tokenizer


def load_kit_ml_test_data(
    kit_ml_dir: str = KIT_ML_DIR,
    num_samples: Optional[int] = None
) -> List[Dict]:
    """
    从 KIT-ML 数据集加载测试数据
    
    Returns:
        test_data: [{'id': str, 'text': str, 'motion': np.ndarray}, ...]
    """
    test_file = os.path.join(kit_ml_dir, TEST_FILE)
    motion_dir = os.path.join(kit_ml_dir, MOTION_DIR)
    text_dir = os.path.join(kit_ml_dir, TEXT_DIR)
    
    # 读取测试 ID
    with open(test_file, 'r') as f:
        test_ids = [line.strip() for line in f if line.strip()]
    
    # 过滤掉以 M 开头的 ID (mirror 数据)
    test_ids = [tid for tid in test_ids if not tid.startswith('M')]
    
    if num_samples:
        test_ids = test_ids[:num_samples]
    
    print(f"Loading {len(test_ids)} test samples...")
    
    test_data = []
    for tid in tqdm(test_ids, desc="Loading data"):
        motion_path = os.path.join(motion_dir, f"{tid}.npy")
        text_path = os.path.join(text_dir, f"{tid}.txt")
        
        if not os.path.exists(motion_path) or not os.path.exists(text_path):
            continue
        
        # 加载动作并下采样（与训练数据一致）
        motion = np.load(motion_path)
        motion = downsample_motion(motion)  # 40帧 -> 10帧
        
        # 加载文本 (取第一行，只取 # 前面的部分)
        with open(text_path, 'r', encoding='utf-8') as f:
            text_line = f.readline().strip()
            text = text_line.split('#')[0].strip()
        
        test_data.append({
            'id': tid,
            'text': text,
            'motion': motion,  # (10, 21, 3), int32, 0-255 (downsampled)
        })
    
    return test_data


def generate_motion(
    model, 
    tokenizer, 
    text: str,
    max_new_tokens: int = 1024,
) -> Optional[np.ndarray]:
    """生成动作"""
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
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # 解析动作
    tokens = re.findall(r'\d+', response)
    if not tokens:
        return None
    
    motion_flat = np.array([int(t) for t in tokens])
    
    # reshape 为 (T, 21, 3)
    frame_size = NUM_JOINTS * 3
    valid_len = (len(motion_flat) // frame_size) * frame_size
    
    if valid_len == 0:
        return None
    
    motion = motion_flat[:valid_len].reshape(-1, NUM_JOINTS, 3)
    return motion


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM model on KIT-ML test set")
    
    parser.add_argument("--adapter_path", type=str, default="Qwen-Motion-Overfit-v9")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"kit_eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print("=" * 60)
    print("Loading Model")
    print("=" * 60)
    model, tokenizer = load_model(args.adapter_path, use_4bit=not args.no_4bit)
    
    # 加载测试数据
    print("\n" + "=" * 60)
    print("Loading Test Data")
    print("=" * 60)
    test_data = load_kit_ml_test_data(num_samples=args.num_samples)
    print(f"Loaded {len(test_data)} test samples")
    
    # 初始化评估器
    evaluator = MotionEvaluator(num_joints=NUM_JOINTS, seed=args.seed)
    
    # 生成动作并收集结果
    print("\n" + "=" * 60)
    print("Generating Motions")
    print("=" * 60)
    
    gt_motions = []
    gen_motions = []
    texts = []
    success_count = 0
    fail_count = 0
    
    for i, sample in enumerate(tqdm(test_data, desc="Generating")):
        text = sample['text']
        gt_motion = sample['motion']  # (T, 21, 3), int 0-255
        
        # 生成动作
        gen_motion = generate_motion(model, tokenizer, text)
        
        if gen_motion is None:
            fail_count += 1
            if args.verbose:
                print(f"\nFailed to generate for: {text[:50]}...")
            continue
        
        # 反量化
        gt_motion_dq = dequantize(gt_motion)
        gen_motion_dq = dequantize(gen_motion)
        
        gt_motions.append(gt_motion_dq)
        gen_motions.append(gen_motion_dq)
        texts.append(text)
        success_count += 1
        
        if args.verbose and i % 10 == 0:
            print(f"\n[{i}] {text[:40]}...")
            print(f"    GT: {gt_motion.shape}, Gen: {gen_motion.shape}")
        
        # 保存生成的动作
        np.save(os.path.join(output_dir, f"gen_{sample['id']}.npy"), gen_motion_dq)
        np.save(os.path.join(output_dir, f"gt_{sample['id']}.npy"), gt_motion_dq)
    
    print(f"\nSuccess: {success_count}, Failed: {fail_count}")
    
    if success_count == 0:
        print("ERROR: No motions generated successfully!")
        return
    
    # 运行评估
    print("\n" + "=" * 60)
    print("Running Evaluation")
    print("=" * 60)
    
    results = evaluator.evaluate_all(
        gt_motions=gt_motions,
        gen_motions=gen_motions,
        texts=texts,
        compute_mm=False,
    )
    
    # 添加额外统计
    results['num_samples'] = success_count
    results['failed_count'] = fail_count
    results['success_rate'] = success_count / len(test_data)
    
    gt_frames = [m.shape[0] for m in gt_motions]
    gen_frames = [m.shape[0] for m in gen_motions]
    results['GT_avg_frames'] = float(np.mean(gt_frames))
    results['Gen_avg_frames'] = float(np.mean(gen_frames))
    results['frame_ratio'] = float(np.mean(gen_frames) / np.mean(gt_frames))
    
    # 保存结果
    results_path = os.path.join(output_dir, 'results.json')
    evaluator.save_results(results, results_path)
    
    # 打印最终结果
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    print(f"Samples: {success_count}/{len(test_data)} ({results['success_rate']*100:.1f}%)")
    print(f"FID: {results['FID']:.4f}")
    print(f"GT Diversity: {results['GT_Diversity']:.4f}")
    print(f"Gen Diversity: {results['Gen_Diversity']:.4f}")
    
    if results['Matching_Score'] is not None:
        print(f"Matching Score: {results['Matching_Score']:.4f}")
    if results['R_precision_top1'] is not None:
        print(f"R-precision: Top1={results['R_precision_top1']:.4f}, Top2={results['R_precision_top2']:.4f}, Top3={results['R_precision_top3']:.4f}")
    
    print(f"Frame ratio (gen/gt): {results['frame_ratio']:.2f}")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

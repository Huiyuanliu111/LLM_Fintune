"""
通用模型评估脚本 - 支持不同版本的模型

用法:
    uv run python -m eval.eval_model --version v6 --num_samples 20
    uv run python -m eval.eval_model --version v7 --num_samples 20
"""

import os
import sys
import json
import argparse
import re
import numpy as np
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from eval.evaluator import MotionEvaluator

# 基础配置
BASE_MODEL_NAME = "Qwen/Qwen3-0.6B"
BINS = 256
V_MIN = -3000.0
V_MAX = 3000.0
NUM_JOINTS = 21

# 版本配置
# 注意：训练时的 MAX_SEQ_LENGTH 限制会截断数据！
# v6/v7 用 finetune_v2.py 训练，MAX_SEQ_LENGTH=1024 -> 约 16 帧
# v9 用 finetune_v5.py 训练，MAX_SEQ_LENGTH=4096 -> 不截断
VERSION_CONFIG = {
    "v5": {
        "adapter_path": "Qwen-Motion-Overfit-v5",
        "data_dir": "KIT-ML/qwen_ready_v5",
        "max_seq_length": 4096,  # 不截断
        "expected_frames": 40,
        "expected_tokens": 2520,
    },
    "v6": {
        "adapter_path": "Qwen-Motion-Overfit-v6",
        "data_dir": "KIT-ML/qwen_ready_v6",
        "max_seq_length": 1024,  # 被截断！1024/63 ≈ 16 帧
        "expected_frames": 16,  # 实际训练只看到 16 帧
        "expected_tokens": 1008,  # 16 * 63
    },
    "v7": {
        "adapter_path": "Qwen-Motion-Overfit-v7",
        "data_dir": "KIT-ML/qwen_ready_v7",
        "max_seq_length": 1024,  # 被截断！
        "expected_frames": 16,  # 实际训练只看到 16 帧
        "expected_tokens": 1008,
    },
    "v8": {
        "adapter_path": "Qwen-Motion-Overfit-v8",
        "data_dir": "KIT-ML/qwen_ready_v8",
        "max_seq_length": 4096,  # 不截断
        "expected_frames": 40,
        "expected_tokens": 2520,
    },
    "v9": {
        "adapter_path": "Qwen-Motion-Overfit-v9",
        "data_dir": "KIT-ML/qwen_ready_v9",
        "max_seq_length": 4096,  # 不截断
        "expected_frames": 10,  # 训练数据本身就是 10 帧（下采样 4x）
        "expected_tokens": 630,
    },
}


def dequantize(motion_data, bins=BINS, v_min=V_MIN, v_max=V_MAX):
    """反量化"""
    norm = motion_data.astype(float) / (bins - 1)
    return (norm * (v_max - v_min)) + v_min


def load_model(adapter_path, use_4bit=True):
    """加载模型"""
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


def load_test_data_from_jsonl(data_dir, version, num_samples=None):
    """从 JSONL 加载测试数据"""
    # 尝试不同的文件名
    possible_files = [
        f"test_{version}.jsonl",
        f"val_{version}.jsonl",
        "test.jsonl",
        "val.jsonl",
    ]
    
    test_file = None
    for fname in possible_files:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            test_file = path
            break
    
    if test_file is None:
        # 如果没有测试文件，使用训练文件
        train_file = os.path.join(data_dir, f"train_{version}.jsonl")
        if os.path.exists(train_file):
            test_file = train_file
            print(f"Warning: No test file found, using train file: {train_file}")
        else:
            raise FileNotFoundError(f"No data file found in {data_dir}")
    
    print(f"Loading data from: {test_file}")
    
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            messages = data.get('messages', [])
            
            text = None
            gt_tokens = None
            
            for msg in messages:
                if msg['role'] == 'user':
                    text = msg['content']
                elif msg['role'] == 'assistant':
                    gt_tokens = msg['content']
            
            if text and gt_tokens:
                test_data.append({
                    'text': text,
                    'gt_tokens': gt_tokens,
                })
    
    if num_samples:
        test_data = test_data[:num_samples]
    
    return test_data


def parse_motion_tokens(token_str, max_frames=None):
    """
    解析 motion tokens
    
    Args:
        token_str: token 字符串
        max_frames: 最大帧数限制（模拟训练时的截断）
    """
    tokens = re.findall(r'\d+', token_str)
    if not tokens:
        return None
    
    motion_flat = np.array([int(t) for t in tokens])
    frame_size = NUM_JOINTS * 3
    valid_len = (len(motion_flat) // frame_size) * frame_size
    
    if valid_len == 0:
        return None
    
    motion = motion_flat[:valid_len].reshape(-1, NUM_JOINTS, 3)
    
    # 模拟训练时的截断
    if max_frames is not None and motion.shape[0] > max_frames:
        motion = motion[:max_frames]
    
    return motion


def generate_motion(model, tokenizer, text, max_new_tokens=4096):
    """生成动作"""
    messages = [
        {'role': 'system', 'content': 'You are a motion generation assistant. Generate the motion sequence (flattened integer tokens) based on the text description.'},
        {'role': 'user', 'content': text}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Evaluate motion generation model")
    parser.add_argument("--version", type=str, required=True, choices=list(VERSION_CONFIG.keys()))
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 获取版本配置
    config = VERSION_CONFIG[args.version]
    print(f"\n{'='*60}")
    print(f"Evaluating {args.version.upper()}")
    print(f"{'='*60}")
    print(f"Adapter: {config['adapter_path']}")
    print(f"Data: {config['data_dir']}")
    print(f"Expected frames: {config['expected_frames']}")
    print(f"Expected tokens: {config['expected_tokens']}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, args.version, f"eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print(f"\n{'='*60}")
    print("Loading Model")
    print(f"{'='*60}")
    model, tokenizer = load_model(config['adapter_path'])
    
    # 加载测试数据
    print(f"\n{'='*60}")
    print("Loading Test Data")
    print(f"{'='*60}")
    test_data = load_test_data_from_jsonl(config['data_dir'], args.version, args.num_samples)
    print(f"Loaded {len(test_data)} samples")
    
    # 评估
    print(f"\n{'='*60}")
    print("Generating and Evaluating")
    print(f"{'='*60}")
    
    gt_motions = []
    gen_motions = []
    texts = []
    success_count = 0
    fail_count = 0
    
    # 计算训练时的最大帧数（模拟截断）
    max_frames = config['expected_frames']
    # 根据版本配置设置生成的最大 token 数（v6/v7 用 1024，其他用 expected_tokens + 余量）
    max_gen_tokens = config['max_seq_length'] if config['max_seq_length'] <= 1024 else config['expected_tokens'] + 100
    print(f"Max generation tokens: {max_gen_tokens}")
    
    for i, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        text = sample['text']
        gt_tokens = sample['gt_tokens']
        
        # 解析 GT（按训练时的截断处理）
        gt_motion = parse_motion_tokens(gt_tokens, max_frames=max_frames)
        if gt_motion is None:
            fail_count += 1
            continue
        
        # 生成（使用版本对应的 max_new_tokens）
        gen_response = generate_motion(model, tokenizer, text, max_new_tokens=max_gen_tokens)
        gen_motion = parse_motion_tokens(gen_response)
        
        if gen_motion is None:
            if args.verbose:
                print(f"\n[{i}] FAILED: {text[:50]}...")
                print(f"    Response: {gen_response[:100]}...")
            fail_count += 1
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
        
        # 保存
        np.save(os.path.join(output_dir, f"gen_{i:04d}.npy"), gen_motion_dq)
        np.save(os.path.join(output_dir, f"gt_{i:04d}.npy"), gt_motion_dq)
    
    print(f"\nSuccess: {success_count}, Failed: {fail_count}")
    
    if success_count == 0:
        print("ERROR: No valid generations!")
        return
    
    # 运行评估
    print(f"\n{'='*60}")
    print("Computing Metrics")
    print(f"{'='*60}")
    
    evaluator = MotionEvaluator(num_joints=NUM_JOINTS, seed=args.seed)
    results = evaluator.evaluate_all(
        gt_motions=gt_motions,
        gen_motions=gen_motions,
        texts=texts,
        compute_mm=False,
    )
    
    # 添加统计
    results['version'] = args.version
    results['num_samples'] = success_count
    results['failed_count'] = fail_count
    results['success_rate'] = success_count / len(test_data)
    
    gt_frames = [m.shape[0] for m in gt_motions]
    gen_frames = [m.shape[0] for m in gen_motions]
    results['GT_avg_frames'] = float(np.mean(gt_frames))
    results['Gen_avg_frames'] = float(np.mean(gen_frames))
    results['frame_ratio'] = float(np.mean(gen_frames) / np.mean(gt_frames)) if np.mean(gt_frames) > 0 else 0
    results['expected_frames'] = config['expected_frames']
    
    # 保存结果
    results_path = os.path.join(output_dir, 'results.json')
    evaluator.save_results(results, results_path)
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - {args.version.upper()}")
    print(f"{'='*60}")
    print(f"Success rate: {results['success_rate']*100:.1f}% ({success_count}/{len(test_data)})")
    print(f"Expected frames: {config['expected_frames']}")
    print(f"GT avg frames: {results['GT_avg_frames']:.1f}")
    print(f"Gen avg frames: {results['Gen_avg_frames']:.1f}")
    print(f"Frame ratio: {results['frame_ratio']:.2f}")
    print(f"FID: {results['FID']:.4f}")
    print(f"GT Diversity: {results['GT_Diversity']:.4f}")
    print(f"Gen Diversity: {results['Gen_Diversity']:.4f}")
    if results['Matching_Score'] is not None:
        print(f"Matching Score: {results['Matching_Score']:.4f}")
    if results['R_precision_top1'] is not None:
        print(f"R-precision: Top1={results['R_precision_top1']:.4f}, Top2={results['R_precision_top2']:.4f}, Top3={results['R_precision_top3']:.4f}")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

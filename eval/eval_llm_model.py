"""
LLM Motion Model Evaluation Script

完整的评估流程：
1. 加载微调后的 LLM 模型
2. 从测试集生成动作
3. 与 ground truth 对比计算评估指标

用法:
    python -m eval.eval_llm_model \
        --adapter_path Qwen-Motion-Overfit-v9 \
        --test_jsonl KIT-ML/qwen_ready_v9/test_v9.jsonl \
        --output_dir eval_results/v9
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

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from eval.evaluator import MotionEvaluator
from eval.metrics import calculate_motion_statistics


# 配置常量
BASE_MODEL_NAME = "Qwen/Qwen3-0.6B"
BINS = 256
V_MIN = -3000.0
V_MAX = 3000.0
NUM_JOINTS = 21


def dequantize(motion_data: np.ndarray, bins=BINS, v_min=V_MIN, v_max=V_MAX) -> np.ndarray:
    """
    反量化：将 [0, bins-1] 的整数映射回 [v_min, v_max]
    """
    norm = motion_data.astype(float) / (bins - 1)
    v_range = v_max - v_min
    denorm_data = (norm * v_range) + v_min
    return denorm_data


def load_model(adapter_path: str, use_4bit: bool = True):
    """加载模型和 tokenizer"""
    print(f"Loading tokenizer from {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    
    # 添加新 Token
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


def parse_motion_from_tokens(token_str: str, num_joints: int = NUM_JOINTS) -> Optional[np.ndarray]:
    """
    从 token 字符串解析动作数据
    
    支持两种格式:
    1. "<0><123><45>..." 格式
    2. "0 123 45..." 空格分隔格式
    """
    # 提取所有数字
    tokens = re.findall(r'\d+', token_str)
    
    if not tokens:
        return None
    
    motion_flat = np.array([int(t) for t in tokens])
    
    # reshape 为 (T, J, 3)
    frame_size = num_joints * 3
    valid_len = (len(motion_flat) // frame_size) * frame_size
    
    if valid_len == 0:
        return None
    
    motion = motion_flat[:valid_len].reshape(-1, num_joints, 3)
    return motion


def load_test_data(jsonl_path: str) -> List[Dict]:
    """加载测试数据"""
    test_data = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
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
    
    return test_data


def generate_motion(
    model, 
    tokenizer, 
    text: str,
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    """生成动作 token 序列"""
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
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response


def evaluate_model(
    model,
    tokenizer,
    test_data: List[Dict],
    evaluator: MotionEvaluator,
    num_samples: Optional[int] = None,
    mm_num_repeats: int = 3,
    compute_mm: bool = False,
    output_dir: Optional[str] = None,
    verbose: bool = False,
) -> Dict:
    """
    评估模型
    
    Args:
        model: 生成模型
        tokenizer: tokenizer
        test_data: 测试数据
        evaluator: 评估器
        num_samples: 评估的样本数 (None = 全部)
        mm_num_repeats: 多模态性的重复生成次数
        compute_mm: 是否计算多模态性
        output_dir: 输出目录 (保存生成的动作)
        verbose: 是否打印详细信息
    """
    if num_samples:
        test_data = test_data[:num_samples]
    
    print(f"\nEvaluating on {len(test_data)} samples...")
    
    gt_motions = []
    gen_motions = []
    texts = []
    motions_per_text = [] if compute_mm else None
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        gen_dir = os.path.join(output_dir, 'generated')
        gt_dir = os.path.join(output_dir, 'groundtruth')
        os.makedirs(gen_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
    
    failed_count = 0
    
    for idx, sample in enumerate(tqdm(test_data, desc="Generating and evaluating")):
        text = sample['text']
        gt_tokens = sample['gt_tokens']
        
        # 解析 GT 动作
        gt_motion = parse_motion_from_tokens(gt_tokens)
        if gt_motion is None:
            if verbose:
                print(f"Warning: Failed to parse GT motion for sample {idx}")
            failed_count += 1
            continue
        
        # 反量化 GT
        gt_motion = dequantize(gt_motion)
        
        # 生成动作
        gen_token_str = generate_motion(model, tokenizer, text)
        gen_motion = parse_motion_from_tokens(gen_token_str)
        
        if gen_motion is None:
            if verbose:
                print(f"Warning: Failed to generate valid motion for sample {idx}")
            failed_count += 1
            continue
        
        # 反量化生成的动作
        gen_motion = dequantize(gen_motion)
        
        gt_motions.append(gt_motion)
        gen_motions.append(gen_motion)
        texts.append(text)
        
        # 保存动作
        if output_dir:
            np.save(os.path.join(gen_dir, f'motion_{idx:04d}.npy'), gen_motion)
            np.save(os.path.join(gt_dir, f'motion_{idx:04d}.npy'), gt_motion)
        
        # 多模态性: 为每个文本多次生成
        if compute_mm and mm_num_repeats > 1:
            text_motions = [gen_motion]
            for _ in range(mm_num_repeats - 1):
                gen_str = generate_motion(model, tokenizer, text, do_sample=True, temperature=0.8)
                motion = parse_motion_from_tokens(gen_str)
                if motion is not None:
                    text_motions.append(dequantize(motion))
            motions_per_text.append(text_motions)
        
        if verbose and idx % 20 == 0:
            print(f"\nSample {idx}:")
            print(f"  Text: {text[:60]}...")
            print(f"  GT frames: {gt_motion.shape[0]}, Gen frames: {gen_motion.shape[0]}")
    
    print(f"\nSuccessfully processed: {len(gen_motions)} / {len(test_data)}")
    print(f"Failed: {failed_count}")
    
    if len(gen_motions) == 0:
        print("Error: No valid motions generated!")
        return {}
    
    # 运行评估
    results = evaluator.evaluate_all(
        gt_motions=gt_motions,
        gen_motions=gen_motions,
        texts=texts,
        motions_per_text=motions_per_text,
        compute_mm=compute_mm,
    )
    
    # 添加额外统计
    results['num_samples'] = len(gen_motions)
    results['failed_count'] = failed_count
    results['success_rate'] = len(gen_motions) / len(test_data)
    
    # 计算帧数统计
    gt_frame_counts = [m.shape[0] for m in gt_motions]
    gen_frame_counts = [m.shape[0] for m in gen_motions]
    
    results['GT_avg_frames'] = float(np.mean(gt_frame_counts))
    results['Gen_avg_frames'] = float(np.mean(gen_frame_counts))
    results['frame_ratio'] = float(np.mean(gen_frame_counts) / np.mean(gt_frame_counts))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM motion generation model")
    
    # 模型参数
    parser.add_argument("--adapter_path", type=str, default="Qwen-Motion-Overfit-v9",
                        help="Path to LoRA adapter")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL_NAME,
                        help="Base model name")
    parser.add_argument("--no_4bit", action="store_true",
                        help="Disable 4-bit quantization")
    
    # 数据参数
    parser.add_argument("--test_jsonl", type=str, required=True,
                        help="Path to test JSONL file")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    
    # 评估参数
    parser.add_argument("--compute_mm", action="store_true",
                        help="Compute multimodality (slower)")
    parser.add_argument("--mm_repeats", type=int, default=3,
                        help="Number of repeats for multimodality")
    parser.add_argument("--diversity_times", type=int, default=300,
                        help="Diversity sampling times")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Output directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config = vars(args)
    config['timestamp'] = timestamp
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 加载模型
    print("=" * 60)
    print("Loading Model")
    print("=" * 60)
    model, tokenizer = load_model(args.adapter_path, use_4bit=not args.no_4bit)
    
    # 加载测试数据
    print("\n" + "=" * 60)
    print("Loading Test Data")
    print("=" * 60)
    test_data = load_test_data(args.test_jsonl)
    print(f"Loaded {len(test_data)} test samples")
    
    # 初始化评估器
    evaluator = MotionEvaluator(
        num_joints=NUM_JOINTS,
        diversity_times=args.diversity_times,
        seed=args.seed,
    )
    
    # 运行评估
    print("\n" + "=" * 60)
    print("Running Evaluation")
    print("=" * 60)
    
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        evaluator=evaluator,
        num_samples=args.num_samples,
        mm_num_repeats=args.mm_repeats,
        compute_mm=args.compute_mm,
        output_dir=output_dir,
        verbose=args.verbose,
    )
    
    # 保存结果
    results_path = os.path.join(output_dir, 'results.json')
    evaluator.save_results(results, results_path)
    
    # 打印最终结果
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    key_metrics = ['FID', 'GT_Diversity', 'Gen_Diversity', 'Matching_Score',
                   'R_precision_top1', 'R_precision_top2', 'R_precision_top3',
                   'MultiModality', 'success_rate', 'frame_ratio']
    
    for key in key_metrics:
        if key in results and results[key] is not None:
            print(f"{key}: {results[key]:.4f}")
    
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

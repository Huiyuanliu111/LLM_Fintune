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
from eval.metrics import calculate_batch_mse

# 尝试导入 MDM evaluator
try:
    from eval.mdm_evaluator import MDMEvaluator, check_mdm_available
    MDM_AVAILABLE = check_mdm_available()
except ImportError:
    MDM_AVAILABLE = False

# 基础配置
BASE_MODEL_NAME = "Qwen/Qwen3-0.6B"
BINS = 256
NUM_JOINTS = 21

# 原始数据目录（用于获取真实的 min/max）
RAW_MOTION_DIR = "KIT-ML/new_joints"
DOWNSAMPLED_DIR = "KIT-ML/new_joints_40"
TEXTS_DIR = "KIT-ML/texts"


def build_text_to_motion_id_map(texts_dir=TEXTS_DIR):
    """
    建立 文本 -> motion_id 的映射
    
    Returns:
        dict: {text: motion_id}
    """
    from pathlib import Path
    
    text_to_id = {}
    texts_path = Path(texts_dir)
    
    if not texts_path.exists():
        print(f"Warning: texts directory not found: {texts_dir}")
        return text_to_id
    
    for txt_file in texts_path.glob("*.txt"):
        motion_id = txt_file.stem  # e.g., "00001"
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 一个文件可能有多行文本描述
        for line in content.split('\n'):
            if not line.strip():
                continue
            # 格式：text#POS_tags#score1#score2
            text = line.split('#')[0].strip()
            if text:
                text_to_id[text] = motion_id
    
    print(f"Built text->motion_id mapping with {len(text_to_id)} entries")
    return text_to_id


def get_motion_minmax(motion_id, raw_dir=DOWNSAMPLED_DIR):
    """
    从原始 .npy 文件获取 min/max
    
    Args:
        motion_id: motion ID (e.g., "00001")
        raw_dir: directory containing raw .npy files
    
    Returns:
        raw_motion, v_min, v_max (or None, None, None if not found)
    """
    raw_path = os.path.join(raw_dir, f"{motion_id}.npy")
    if not os.path.exists(raw_path):
        # 尝试不带前导零的版本
        raw_path = os.path.join(raw_dir, f"{int(motion_id)}.npy")
        if not os.path.exists(raw_path):
            return None, None, None
    
    raw_motion = np.load(raw_path)
    v_min = float(raw_motion.min())
    v_max = float(raw_motion.max())
    return raw_motion, v_min, v_max

# 版本配置
# 注意：训练时的 MAX_SEQ_LENGTH 限制会截断数据！
# v6/v7 用 finetune_v2.py 训练，MAX_SEQ_LENGTH=1024 -> 约 16 帧
# v9 用 finetune_v5.py 训练，MAX_SEQ_LENGTH=4096 -> 不截断
VERSION_CONFIG = {
    "v5": {
        "adapter_path": "Qwen-Motion-Overfit-v5",
        "data_dir": "KIT-ML/qwen_ready_v5",
        "max_seq_length": 1024,  # 截断到 16 帧，与 v6 公平比较
        "expected_frames": 16,   # 16 帧
        "expected_tokens": 1008, # 16 * 63
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
    "v10": {
        "adapter_path": "Qwen-Motion-Overfit-v10",
        "data_dir": "KIT-ML/qwen_ready_v10",
        "max_seq_length": 4096,  # 不截断
        "expected_frames": 40,
        "expected_tokens": 2520,  # 40 * 63
        # v10 使用全局 min/max: V_MIN=-6429.9, V_MAX=7001.1
    },
}


def dequantize(motion_data, bins=BINS, v_min=None, v_max=None):
    """
    反量化
    
    Args:
        motion_data: quantized motion data
        bins: number of bins
        v_min: min value (if None, cannot dequantize properly)
        v_max: max value (if None, cannot dequantize properly)
    """
    if v_min is None or v_max is None:
        raise ValueError("v_min and v_max must be provided for correct dequantization")
    norm = motion_data.astype(float) / (bins - 1)
    return (norm * (v_max - v_min)) + v_min


def uniform_downsample(motion: np.ndarray, target_frames: int = 40) -> np.ndarray:
    """均匀降采样"""
    T = motion.shape[0]
    if T <= target_frames:
        return motion
    idx = np.linspace(0, T - 1, target_frames).astype(int)
    return motion[idx]


def load_test_data_from_raw(text_dir, motion_dir, num_samples=None, max_frames=120):
    """
    直接从原始数据目录加载测试数据
    
    Returns:
        list of dict with keys: motion_id, text, raw_motion, v_min, v_max
    """
    from glob import glob
    
    # 找到所有配对的数据
    motion_files = sorted(glob(os.path.join(motion_dir, "*.npy")))
    
    test_data = []
    for motion_path in motion_files:
        motion_id = os.path.splitext(os.path.basename(motion_path))[0]
        text_path = os.path.join(text_dir, f"{motion_id}.txt")
        
        if not os.path.exists(text_path):
            continue
        
        # 加载原始 motion
        raw_motion = np.load(motion_path)
        
        # 跳过太长的序列
        if raw_motion.shape[0] > max_frames:
            continue
        
        # 加载文本
        with open(text_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        text = first_line.split("#")[0]
        
        # 计算该文件的 min/max（用于正确 dequantize）
        v_min = float(raw_motion.min())
        v_max = float(raw_motion.max())
        
        test_data.append({
            'motion_id': motion_id,
            'text': text,
            'raw_motion': raw_motion,
            'v_min': v_min,
            'v_max': v_max,
        })
    
    if num_samples:
        test_data = test_data[:num_samples]
    
    return test_data


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
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_mdm", action="store_true", help="Use MDM pretrained evaluator for FID (requires downloading MDM models)")
    
    args = parser.parse_args()
    
    # 检查 MDM 可用性
    use_mdm = args.use_mdm and MDM_AVAILABLE
    if args.use_mdm and not MDM_AVAILABLE:
        print("Warning: --use_mdm specified but MDM evaluator not available.")
        print("Please download MDM models first:")
        print("  cd ../motion-diffusion-model && bash prepare/download_t2m_evaluators.sh")
        print("Falling back to statistical features...")
    
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
    
    # 加载测试数据（从 JSONL）
    print(f"\n{'='*60}")
    print("Loading Test Data")
    print(f"{'='*60}")
    test_data = load_test_data_from_jsonl(config['data_dir'], args.version, args.num_samples)
    print(f"Loaded {len(test_data)} samples")
    
    # 构建 text -> motion_id 映射（用于获取正确的 min/max）
    print(f"\n{'='*60}")
    print("Building Text to Motion ID Mapping")
    print(f"{'='*60}")
    text_to_motion_id = build_text_to_motion_id_map()
    
    # 评估
    print(f"\n{'='*60}")
    print("Generating and Evaluating")
    print(f"{'='*60}")
    
    gt_motions = []
    gen_motions = []
    texts = []
    success_count = 0
    fail_count = 0
    minmax_info = []  # 记录每个样本的 min/max
    
    # 计算训练时的最大帧数（模拟截断）
    max_frames = config['expected_frames']
    # 根据版本配置设置生成的最大 token 数
    max_gen_tokens = config['max_seq_length'] if config['max_seq_length'] <= 1024 else config['expected_tokens'] + 100
    print(f"Max generation tokens: {max_gen_tokens}")
    print(f"Max frames: {max_frames}")
    
    # 后备的固定范围（当无法获取原始数据时使用）
    V_MIN_FALLBACK = -3000.0
    V_MAX_FALLBACK = 3000.0
    
    for i, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        text = sample['text']
        gt_tokens = sample['gt_tokens']
        
        # 解析 GT（按训练时的截断处理）
        gt_motion = parse_motion_tokens(gt_tokens, max_frames=max_frames)
        if gt_motion is None:
            fail_count += 1
            continue
        
        # 生成
        gen_response = generate_motion(model, tokenizer, text, max_new_tokens=max_gen_tokens)
        gen_motion = parse_motion_tokens(gen_response, max_frames=max_frames)
        
        if gen_motion is None:
            # 总是打印失败原因
            print(f"\n[{i}] FAILED: {text[:50]}...")
            print(f"    Response (first 200): {gen_response[:200]}...")
            print(f"    Response length: {len(gen_response)}")
            fail_count += 1
            continue
        
        # 获取原始数据的 min/max
        motion_id = text_to_motion_id.get(text)
        if motion_id:
            raw_motion, v_min, v_max = get_motion_minmax(motion_id)
            if raw_motion is not None:
                # 使用原始数据的 min/max（绝对 FID）
                use_raw = True
            else:
                v_min, v_max = V_MIN_FALLBACK, V_MAX_FALLBACK
                use_raw = False
        else:
            v_min, v_max = V_MIN_FALLBACK, V_MAX_FALLBACK
            use_raw = False
            motion_id = "unknown"
        
        # 记录 min/max 信息
        minmax_info.append({
            'motion_id': motion_id,
            'v_min': v_min,
            'v_max': v_max,
            'use_raw': use_raw,
        })
        
        # 反量化
        gt_motion_dq = dequantize(gt_motion, v_min=v_min, v_max=v_max)
        gen_motion_dq = dequantize(gen_motion, v_min=v_min, v_max=v_max)
        
        # 调试：检查 tokens 是否匹配，打印 min/max
        if i == 0 or args.verbose:
            gt_flat = gt_motion.flatten()
            gen_flat = gen_motion.flatten()
            min_len = min(len(gt_flat), len(gen_flat))
            token_match = np.sum(gt_flat[:min_len] == gen_flat[:min_len])
            print(f"\n[{i}] Debug (motion_id={motion_id}):")
            print(f"    v_min={v_min:.2f}, v_max={v_max:.2f}, use_raw={use_raw}")
            print(f"    GT tokens (first 20): {gt_flat[:20]}")
            print(f"    Gen tokens (first 20): {gen_flat[:20]}")
            print(f"    Token match: {token_match}/{min_len} ({100*token_match/min_len:.1f}%)")
            print(f"    GT shape: {gt_motion.shape}, Gen shape: {gen_motion.shape}")
        
        gt_motions.append(gt_motion_dq)
        gen_motions.append(gen_motion_dq)
        texts.append(text)
        success_count += 1
        
        # 保存
        np.save(os.path.join(output_dir, f"gen_{i:04d}.npy"), gen_motion_dq)
        np.save(os.path.join(output_dir, f"gt_{i:04d}.npy"), gt_motion_dq)
    
    print(f"\nSuccess: {success_count}, Failed: {fail_count}")
    
    # 打印 min/max 统计信息
    if minmax_info:
        use_raw_count = sum(1 for info in minmax_info if info['use_raw'])
        v_mins = [info['v_min'] for info in minmax_info]
        v_maxs = [info['v_max'] for info in minmax_info]
        
        print(f"\n{'='*60}")
        print("Min/Max Statistics")
        print(f"{'='*60}")
        print(f"Samples using raw min/max: {use_raw_count}/{len(minmax_info)}")
        print(f"v_min range: [{min(v_mins):.2f}, {max(v_mins):.2f}], mean={np.mean(v_mins):.2f}")
        print(f"v_max range: [{min(v_maxs):.2f}, {max(v_maxs):.2f}], mean={np.mean(v_maxs):.2f}")
        
        # 打印每个样本的详细 min/max
        print("\nPer-sample min/max:")
        for idx, info in enumerate(minmax_info):
            raw_flag = "RAW" if info['use_raw'] else "FALLBACK"
            print(f"  [{idx}] {info['motion_id']}: v_min={info['v_min']:.2f}, v_max={info['v_max']:.2f} ({raw_flag})")
    
    if success_count == 0:
        print("ERROR: No valid generations!")
        return
    
    # 运行评估
    print(f"\n{'='*60}")
    print("Computing Metrics")
    print(f"{'='*60}")
    
    # 计算 MSE（对于 overfit 模型最重要的指标）
    print("\n[MSE] Computing MSE between GT and Generated...")
    mse_results = calculate_batch_mse(gt_motions, gen_motions)
    print(f"MSE: {mse_results['MSE_mean']:.4f} ± {mse_results['MSE_std']:.4f}")
    print(f"MAE: {mse_results['MAE_mean']:.4f} ± {mse_results['MAE_std']:.4f}")
    print(f"RMSE: {mse_results['RMSE_mean']:.4f} ± {mse_results['RMSE_std']:.4f}")
    
    # 选择 evaluator
    if use_mdm:
        print("\n[FID] Using MDM pretrained evaluator...")
        mdm_evaluator = MDMEvaluator('kit', 'cuda' if torch.cuda.is_available() else 'cpu')
        results = mdm_evaluator.evaluate_all(gt_motions, gen_motions)
        results['evaluator'] = 'MDM_pretrained'
        # MDM 不计算 Matching Score 和 R-precision（需要文本嵌入）
        results['Matching_Score'] = None
        results['R_precision_top1'] = None
        results['R_precision_top2'] = None
        results['R_precision_top3'] = None
    else:
        print("\n[FID] Using statistical feature evaluator...")
        evaluator = MotionEvaluator(num_joints=NUM_JOINTS, seed=args.seed)
        results = evaluator.evaluate_all(
            gt_motions=gt_motions,
            gen_motions=gen_motions,
            texts=texts,
            compute_mm=False,
        )
        results['evaluator'] = 'statistical_features'
    
    # 添加 MSE 结果
    results.update(mse_results)
    
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
    # 保存为 JSON
    import json
    serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, (np.float32, np.float64)):
            serializable[key] = float(value)
        elif isinstance(value, dict):
            serializable[key] = {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in value.items()
            }
        else:
            serializable[key] = value
    serializable['timestamp'] = datetime.now().isoformat()
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - {args.version.upper()}")
    print(f"{'='*60}")
    print(f"Evaluator: {results.get('evaluator', 'unknown')}")
    print(f"Success rate: {results['success_rate']*100:.1f}% ({success_count}/{len(test_data)})")
    print(f"Expected frames: {config['expected_frames']}")
    print(f"GT avg frames: {results['GT_avg_frames']:.1f}")
    print(f"Gen avg frames: {results['Gen_avg_frames']:.1f}")
    print(f"Frame ratio: {results['frame_ratio']:.2f}")
    print(f"\n--- Direct Comparison (Lower = Better) ---")
    print(f"MSE: {results['MSE_mean']:.4f} ± {results['MSE_std']:.4f}")
    print(f"MAE: {results['MAE_mean']:.4f} ± {results['MAE_std']:.4f}")
    print(f"RMSE: {results['RMSE_mean']:.4f} ± {results['RMSE_std']:.4f}")
    print(f"\n--- Distribution Metrics (FID from {results.get('evaluator', 'unknown')}) ---")
    print(f"FID: {results['FID']:.4f}")
    print(f"GT Diversity: {results['GT_Diversity']:.4f}")
    print(f"Gen Diversity: {results['Gen_Diversity']:.4f}")
    if results.get('Matching_Score') is not None:
        print(f"Matching Score: {results['Matching_Score']:.4f}")
    if results.get('R_precision_top1') is not None:
        print(f"R-precision: Top1={results['R_precision_top1']:.4f}, Top2={results['R_precision_top2']:.4f}, Top3={results['R_precision_top3']:.4f}")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

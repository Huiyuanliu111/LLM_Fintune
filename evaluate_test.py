"""
在 test 集上评估模型
计算 loss 和生成质量
"""
import os
import torch
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# 配置
BASE_MODEL_NAME = "Qwen/Qwen3-0.6B"
ADAPTER_PATH = "Qwen-Motion-Overfit-v6"  # 训练后的模型路径
DATA_PATH = "KIT-ML/qwen_ready_v7"
TEST_FILE = "test_v6.jsonl"

def load_model():
    """加载模型和 tokenizer"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 添加新 Token
    print("Adding new tokens <0>...<255>...")
    new_tokens = [f"<{i}>" for i in range(256)]
    tokenizer.add_tokens(new_tokens)
    
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    
    return model, tokenizer

def compute_loss(model, tokenizer, text, target_tokens):
    """计算单个样本的 loss"""
    # 构建完整输入
    messages = [
        {"role": "system", "content": "You are a motion generation assistant. Generate the motion sequence (flattened integer tokens) based on the text description."},
        {"role": "user", "content": text},
        {"role": "assistant", "content": target_tokens}
    ]
    
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
    
    return loss

def generate_motion(model, tokenizer, text, max_new_tokens=512):
    """生成动作序列"""
    messages = [
        {"role": "system", "content": "You are a motion generation assistant. Generate the motion sequence (flattened integer tokens) based on the text description."},
        {"role": "user", "content": text}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response

def count_tokens(token_string):
    """统计 <N> 格式的 token 数量"""
    import re
    tokens = re.findall(r'<\d+>', token_string)
    return len(tokens)

def main():
    # 加载模型
    model, tokenizer = load_model()
    
    # 加载 test 数据
    test_path = os.path.join(DATA_PATH, TEST_FILE)
    print(f"\nLoading test data from {test_path}...")
    
    test_data = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            test_data.append(json.loads(line))
    
    print(f"Test samples: {len(test_data)}")
    
    # 评估
    losses = []
    token_accuracies = []
    length_ratios = []
    
    print("\nEvaluating...")
    for i, sample in enumerate(tqdm(test_data)):
        messages = sample["messages"]
        text = messages[1]["content"]  # user prompt
        target = messages[2]["content"]  # assistant response (ground truth)
        
        # 计算 loss
        try:
            loss = compute_loss(model, tokenizer, text, target)
            losses.append(loss)
        except Exception as e:
            print(f"Error computing loss for sample {i}: {e}")
            continue
        
        # 生成并比较（每 10 个样本生成一次，节省时间）
        if i % 10 == 0:
            try:
                generated = generate_motion(model, tokenizer, text)
                
                target_count = count_tokens(target)
                generated_count = count_tokens(generated)
                
                if target_count > 0:
                    length_ratios.append(generated_count / target_count)
                
                # 打印示例
                if i % 50 == 0:
                    print(f"\n--- Sample {i} ---")
                    print(f"Text: {text[:80]}...")
                    print(f"Target tokens: {target_count}")
                    print(f"Generated tokens: {generated_count}")
                    print(f"Generated (first 100 chars): {generated[:100]}...")
            except Exception as e:
                print(f"Error generating for sample {i}: {e}")
    
    # 打印结果
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Test samples: {len(test_data)}")
    print(f"Evaluated samples: {len(losses)}")
    print(f"\nLoss Statistics:")
    print(f"  Mean loss: {np.mean(losses):.4f}")
    print(f"  Std loss: {np.std(losses):.4f}")
    print(f"  Min loss: {np.min(losses):.4f}")
    print(f"  Max loss: {np.max(losses):.4f}")
    
    if length_ratios:
        print(f"\nLength Ratio (generated/target):")
        print(f"  Mean: {np.mean(length_ratios):.2f}")
        print(f"  Std: {np.std(length_ratios):.2f}")
    
    print("="*60)

if __name__ == "__main__":
    main()


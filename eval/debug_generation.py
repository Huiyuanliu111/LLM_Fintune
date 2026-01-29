"""调试生成失败的原因"""
import os
import re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 配置
BASE_MODEL_NAME = 'Qwen/Qwen3-0.6B'
ADAPTER_PATH = 'Qwen-Motion-Overfit-v9'

# 加载模型
print('Loading model...')
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
new_tokens = [f'<{i}>' for i in range(256)]
tokenizer.add_tokens(new_tokens)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True
)
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

# 加载测试数据
print('\nLoading test samples...')
test_ids = []
with open('KIT-ML/test.txt', 'r') as f:
    for line in f:
        tid = line.strip()
        if tid and not tid.startswith('M'):
            test_ids.append(tid)
            if len(test_ids) >= 10:
                break

print(f'Testing {len(test_ids)} samples\n')
print('=' * 60)

for tid in test_ids:
    # 加载文本
    with open(f'KIT-ML/texts/{tid}.txt', 'r', encoding='utf-8') as f:
        text_line = f.readline().strip()
        text = text_line.split('#')[0].strip()
    
    # 加载 GT motion
    gt_motion = np.load(f'KIT-ML/new_joints_40_bin256/{tid}.npy')
    
    print(f'\n[{tid}] Text: {text}')
    print(f'       GT shape: {gt_motion.shape}, GT tokens: {gt_motion.size}')
    
    # 生成
    messages = [
        {'role': 'system', 'content': 'You are a motion generation assistant. Generate the motion sequence (flattened integer tokens) based on the text description.'},
        {'role': 'user', 'content': text}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    # 显示原始输出
    print(f'       Raw output (first 150 chars): {repr(response[:150])}')
    print(f'       Output length: {len(response)} chars')
    
    # 解析数字
    tokens = re.findall(r'\d+', response)
    print(f'       Found {len(tokens)} number tokens')
    
    if tokens:
        motion_flat = np.array([int(t) for t in tokens])
        frame_size = 21 * 3  # 63
        valid_len = (len(motion_flat) // frame_size) * frame_size
        num_frames = valid_len // frame_size
        print(f'       Valid: {valid_len} tokens -> {num_frames} frames')
        
        if num_frames == 0:
            print('       >>> FAILED: Not enough tokens for 1 frame (need 63)')
    else:
        print('       >>> FAILED: No number tokens found!')
    
    print('-' * 60)

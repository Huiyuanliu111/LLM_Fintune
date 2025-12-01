import json
import os
import re
from tqdm import tqdm

# 定义路径
INPUT_DIR = "KIT-ML/qwen_ready"
OUTPUT_DIR = "KIT-ML/qwen_ready_v3"
FILES = ["train_v1.jsonl", "val_v1.jsonl", "test_v1.jsonl"]

def process_file(input_path, output_path):
    print(f"Processing {input_path} -> {output_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin):
            data = json.loads(line)
            messages = data["messages"]
            
            # 找到 assistant 的回复（动作序列）
            for msg in messages:
                if msg["role"] == "assistant":
                    content = msg["content"]
                    
                    # 1. 提取所有数字
                    # 假设原来是 "60 157 60 ..."
                    numbers = re.findall(r'\d+', content)
                    
                    # 2. 转换为新格式: "<60><157><60>..."
                    # 注意：我们去掉空格，因为特殊 Token 之间不需要空格分隔
                    new_content = "".join([f"<{num}>" for num in numbers])
                    
                    msg["content"] = new_content
            
            # 写入新文件
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for filename in FILES:
        input_path = os.path.join(INPUT_DIR, filename)
        # output_path = os.path.join(OUTPUT_DIR, filename)
        new_filename = filename.replace("_v1.jsonl", "_v3.jsonl")
        output_path = os.path.join(OUTPUT_DIR, new_filename)
        
        if os.path.exists(input_path):
            process_file(input_path, output_path)
        else:
            print(f"Skipping {filename}, not found.")

if __name__ == "__main__":
    main()


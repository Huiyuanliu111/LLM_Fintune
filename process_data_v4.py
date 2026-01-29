import json
import re
import os

# 配置
INPUT_DIR = "KIT-ML/qwen_ready_v8"
OUTPUT_DIR = "KIT-ML/qwen_ready_v9"
FILES = ["train_v9.jsonl", "val_v9.jsonl"]
TOKENS_PER_FRAME = 63
DOWNSAMPLE_RATE = 4  # 每4帧取1帧

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_token_string(token_str):
    # 1. 提取所有数字
    # 格式如 <123><45>...
    # 使用正则找到所有 <...> 里的数字
    matches = re.findall(r"<(\d+)>", token_str)
    
    if not matches:
        return token_str
        
    # 2. 检查长度是否完整
    total_tokens = len(matches)
    if total_tokens % TOKENS_PER_FRAME != 0:
        print(f"Warning: Token count {total_tokens} is not divisible by {TOKENS_PER_FRAME}. Skipping checks but processing anyway.")
    
    # 3. 分组成帧
    # frames 是一个 list，每个元素是包含 63 个 token ID 的 list
    frames = [matches[i:i + TOKENS_PER_FRAME] for i in range(0, total_tokens, TOKENS_PER_FRAME)]
    
    # 4. 降采样
    # 从第0帧开始，每隔 DOWNSAMPLE_RATE 取一帧
    downsampled_frames = frames[::DOWNSAMPLE_RATE]
    
    # 5. 重组字符串
    new_token_list = []
    for frame in downsampled_frames:
        for token_id in frame:
            new_token_list.append(f"<{token_id}>")
            
    return "".join(new_token_list)

def process_file(filename):
    input_path = os.path.join(INPUT_DIR, filename)
    output_filename = filename.replace("v3", "v4")
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    print(f"Processing {input_path} -> {output_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            data = json.loads(line)
            
            # 找到 assistant 的回复
            messages = data.get("messages", [])
            for msg in messages:
                if msg["role"] == "assistant":
                    original_content = msg["content"]
                    # 处理内容
                    new_content = process_token_string(original_content)
                    msg["content"] = new_content
            
            # 写入新文件
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            
    print("Done.")

def main():
    for f in FILES:
        if os.path.exists(os.path.join(INPUT_DIR, f)):
            process_file(f)
        else:
            print(f"File not found: {f}")

if __name__ == "__main__":
    main()


from transformers import AutoTokenizer

# 1. 加载 Tokenizer
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# 2. 定义一个标准的输入样本 (这就是 trl/SFTTrainer 要求的格式)
messages = [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "你好，Qwen！"},
    {"role": "assistant", "content": "你好！有什么我可以帮你的吗？"}
]

# 3. 让 Tokenizer 渲染这个格式，看看底层变成了什么
# tokenize=False 表示我们只看转换后的字符串，不转成数字 ID
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

print("=== 原始数据格式 (List of Dicts) ===")
print(messages)
print("\n=== 模型真正看到的字符串 (经过 chat_template) ===")
print(formatted_prompt)

print("\n=== 查看 Tokenizer 内部的 Jinja 模板逻辑 ===")
print(tokenizer.chat_template)
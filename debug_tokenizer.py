from transformers import AutoTokenizer
import sys

# 设置输出编码为 utf-8
sys.stdout.reconfigure(encoding='utf-8')

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# 模拟训练数据中的一个短序列
# 这是我们希望模型看到的格式（使用了新的 Token）
# 注意：因为在这个脚本里我们还没把新 Token 加进 Tokenizer，
# 所以这里我们只是看看 Tokenizer 在没加新 Token 之前会怎么处理这种格式，
# 或者我们手动加进去看看效果。

print("--- 测试添加新 Token 前 ---")
sample_text = "<60><157><60>"
ids = tokenizer.encode(sample_text, add_special_tokens=False)
print(f"Text: '{sample_text}'")
print(f"IDs: {ids}")
for i in ids:
    print(f"ID {i} -> '{tokenizer.decode([i])}'")

print("\n--- 测试添加新 Token 后 ---")
# 模拟我们在 finetune_v2.py 里做的操作
new_tokens = [f"<{i}>" for i in range(256)]
tokenizer.add_tokens(new_tokens)

ids_new = tokenizer.encode(sample_text, add_special_tokens=False)
print(f"Text: '{sample_text}'")
print(f"IDs: {ids_new}")
for i in ids_new:
    print(f"ID {i} -> '{tokenizer.decode([i])}'")

# 还可以测试一下具体的数字
test_nums = ["<0>", "<255>", "<100>"]
print("\n--- 特定 Token 测试 ---")
for t in test_nums:
    i = tokenizer.encode(t, add_special_tokens=False)[0]
    print(f"Token '{t}' -> ID {i}")

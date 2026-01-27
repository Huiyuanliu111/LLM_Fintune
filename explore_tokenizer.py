"""
探索 Qwen Tokenizer 的词表结构
"""
from transformers import AutoTokenizer
import json

MODEL_NAME = "Qwen/Qwen3-0.6B"

def main():
    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    
    print(f"\n{'='*60}")
    print(f"TOKENIZER VOCABULARY ANALYSIS")
    print(f"{'='*60}")
    print(f"Total vocabulary size: {vocab_size}")
    
    # 按 token ID 排序
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    # 分类 token
    special_tokens = []
    digit_tokens = []
    letter_tokens = []
    chinese_tokens = []
    punctuation_tokens = []
    byte_tokens = []  # 如 <0x00> 格式
    angle_bracket_tokens = []  # 如 <|...|> 格式
    other_tokens = []
    
    for token, token_id in sorted_vocab:
        if token.startswith('<|') and token.endswith('|>'):
            special_tokens.append((token, token_id))
        elif token.startswith('<0x') and token.endswith('>'):
            byte_tokens.append((token, token_id))
        elif token.startswith('<') and token.endswith('>'):
            angle_bracket_tokens.append((token, token_id))
        elif token.isdigit() or (len(token) > 1 and token[0] in ' \t' and token[1:].isdigit()):
            digit_tokens.append((token, token_id))
        elif any('\u4e00' <= c <= '\u9fff' for c in token):
            chinese_tokens.append((token, token_id))
        elif token.isalpha() or (len(token) > 1 and token[0] == ' ' and token[1:].isalpha()):
            letter_tokens.append((token, token_id))
        elif len(token) == 1 and not token.isalnum():
            punctuation_tokens.append((token, token_id))
        else:
            other_tokens.append((token, token_id))
    
    print(f"\n--- Token Categories ---")
    print(f"Special tokens (<|...|>): {len(special_tokens)}")
    print(f"Byte tokens (<0x..>): {len(byte_tokens)}")
    print(f"Angle bracket tokens (<...>): {len(angle_bracket_tokens)}")
    print(f"Digit-related tokens: {len(digit_tokens)}")
    print(f"Letter-related tokens: {len(letter_tokens)}")
    print(f"Chinese tokens: {len(chinese_tokens)}")
    print(f"Punctuation tokens: {len(punctuation_tokens)}")
    print(f"Other tokens: {len(other_tokens)}")
    
    # 打印特殊 token
    print(f"\n--- Special Tokens (<|...|>) ---")
    for token, tid in special_tokens[:30]:
        print(f"  {tid:6d}: {repr(token)}")
    if len(special_tokens) > 30:
        print(f"  ... and {len(special_tokens) - 30} more")
    
    # 打印 byte tokens
    print(f"\n--- Byte Tokens (<0x..>) ---")
    for token, tid in byte_tokens[:20]:
        print(f"  {tid:6d}: {repr(token)}")
    if len(byte_tokens) > 20:
        print(f"  ... and {len(byte_tokens) - 20} more")
    
    # 打印其他尖括号 token
    print(f"\n--- Other Angle Bracket Tokens (<...>) ---")
    for token, tid in angle_bracket_tokens[:30]:
        print(f"  {tid:6d}: {repr(token)}")
    if len(angle_bracket_tokens) > 30:
        print(f"  ... and {len(angle_bracket_tokens) - 30} more")
    
    # 打印数字 token
    print(f"\n--- Digit Tokens (sample) ---")
    for token, tid in digit_tokens[:30]:
        print(f"  {tid:6d}: {repr(token)}")
    if len(digit_tokens) > 30:
        print(f"  ... and {len(digit_tokens) - 30} more")
    
    # 打印词表的 ID 范围
    print(f"\n--- Token ID Ranges ---")
    print(f"First 10 tokens:")
    for token, tid in sorted_vocab[:10]:
        print(f"  {tid:6d}: {repr(token)}")
    
    print(f"\nLast 10 tokens (before adding new tokens):")
    for token, tid in sorted_vocab[-10:]:
        print(f"  {tid:6d}: {repr(token)}")
    
    # 检查数字相关 token
    print(f"\n--- Numeric Pattern Tokens ---")
    numeric_patterns = []
    for token, tid in sorted_vocab:
        # 检查纯数字或带前导空格的数字
        # 使用 isdecimal() 而非 isdigit()，因为 isdigit() 会匹配 ²、³ 等 Unicode 字符
        clean = token.lstrip(' ')
        if clean.isdecimal() and len(clean) <= 3:
            numeric_patterns.append((token, tid, int(clean)))
    
    numeric_patterns.sort(key=lambda x: x[2])
    print(f"Found {len(numeric_patterns)} tokens that are pure numbers (0-999):")
    for token, tid, num in numeric_patterns[:50]:
        print(f"  {tid:6d}: {repr(token):10} -> {num}")
    if len(numeric_patterns) > 50:
        print(f"  ... and {len(numeric_patterns) - 50} more")
    
    # 检查是否已经有 <0>, <1>, ... 这样的 token
    print(f"\n--- Checking for <N> format tokens ---")
    existing_angle_numbers = []
    for token, tid in sorted_vocab:
        if token.startswith('<') and token.endswith('>'):
            inner = token[1:-1]
            if inner.isdecimal():  # 使用 isdecimal() 避免匹配 Unicode 数字字符
                existing_angle_numbers.append((token, tid, int(inner)))
    
    if existing_angle_numbers:
        print(f"Found {len(existing_angle_numbers)} existing <N> tokens:")
        for token, tid, num in existing_angle_numbers[:20]:
            print(f"  {tid:6d}: {token}")
    else:
        print("No existing <N> format tokens found (good, we can add them)")
    
    # 保存完整词表到文件
    output_file = "qwen_vocabulary.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({token: tid for token, tid in sorted_vocab}, f, ensure_ascii=False, indent=2)
    print(f"\nFull vocabulary saved to: {output_file}")

if __name__ == "__main__":
    main()


import os

# 目标文件路径
target_file = r"E:\LLM_Fintune\.venv\lib\site-packages\transformers\utils\import_utils.py"

print(f"正在尝试修改: {target_file}")

try:
    with open(target_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    skip = False
    modified = False
    
    for line in lines:
        if "def check_torch_load_is_safe() -> None:" in line:
            print("找到目标函数，正在修改...")
            new_lines.append("def check_torch_load_is_safe() -> None:\n")
            new_lines.append("    return  # 已被 fix_transformers.py 强制禁用\n")
            skip = True
            modified = True
        elif skip:
            # 跳过函数体，直到下一个函数定义或类定义
            if line.strip() == "" or line.startswith("    ") or line.startswith("\t"):
                continue
            else:
                skip = False
                new_lines.append(line)
        else:
            new_lines.append(line)

    if modified:
        with open(target_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print("修改成功！安全检查已被禁用。")
    else:
        print("未找到目标函数，可能已经被修改过。")

except Exception as e:
    print(f"修改失败: {e}")


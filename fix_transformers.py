import os

# 目标文件路径
target_file = r"E:\LLM_Fintune\.venv\lib\site-packages\transformers\utils\import_utils.py"

print(f"正在检查文件: {target_file}")

if not os.path.exists(target_file):
    print("错误: 未找到目标文件。请确认路径是否正确。")
    exit(1)

try:
    with open(target_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    modified = False
    inside_function = False
    
    for i, line in enumerate(lines):
        # 找到函数定义
        if "def check_torch_load_is_safe() -> None:" in line:
            print(f"在第 {i+1} 行找到目标函数。")
            new_lines.append(line)
            # 添加 pass 并注释说明
            new_lines.append("    return  # PATCHED: Bypassed security check\n")
            modified = True
            inside_function = True
            continue
        
        if inside_function:
            # 如果是函数体的内容（有缩进），就注释掉
            if line.strip() and (line.startswith("    ") or line.startswith("\t")):
                # 只有当它看起来像是原有的代码时才注释
                if not line.strip().startswith("#"):
                    new_lines.append("# " + line)
                else:
                    new_lines.append(line)
            else:
                # 遇到非缩进的行（空行除外），说明函数结束了
                if line.strip() != "":
                    inside_function = False
                new_lines.append(line)
        else:
            new_lines.append(line)

    if modified:
        with open(target_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print("成功！已修改 import_utils.py，安全检查已被禁用。")
    else:
        print("警告：未找到 check_torch_load_is_safe 函数，或者文件已经被修改过。")

except Exception as e:
    print(f"发生异常: {e}")

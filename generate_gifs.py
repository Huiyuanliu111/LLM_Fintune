from visualize_motion import parse_motion_tokens, plot_motion
import numpy as np
import re

# Constants for dequantization
BINS = 256
V_MIN = -3000.0
V_MAX = 3000.0

def dequantize(motion_data, bins=BINS, v_min=V_MIN, v_max=V_MAX):
    norm = motion_data.astype(float) / (bins - 1)
    v_range = v_max - v_min
    denorm_data = (norm * v_range) + v_min
    return denorm_data

print("Generating before.gif (original data)...")
# 读取原始数据
with open('original_tokens.txt', 'r') as f:
    original_tokens = f.read().strip()

print('Original tokens length:', len(original_tokens))

# 解析并反量化
tokens = re.findall(r'\d+', original_tokens)
clean_input = ' '.join(tokens)
motion_data = parse_motion_tokens(clean_input)
print('Original motion shape:', motion_data.shape)

motion_data = dequantize(motion_data)
plot_motion(motion_data, output_file='before.gif', fps=10)
print('Saved before.gif')

print("\nGenerating after.gif (downsampled data)...")
# 读取降采样数据
with open('downsampled_tokens.txt', 'r') as f:
    tokens_str = f.read().strip()

print('Downsampled tokens length:', len(tokens_str))

# 解析并反量化
tokens = re.findall(r'\d+', tokens_str)
clean_input = ' '.join(tokens)
motion_data = parse_motion_tokens(clean_input)
print('Downsampled motion shape:', motion_data.shape)

motion_data = dequantize(motion_data)
plot_motion(motion_data, output_file='after.gif', fps=10)
print('Saved after.gif')

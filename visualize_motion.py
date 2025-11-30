import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import argparse

# KIT-ML / MMM (Master Motor Map) 21-joint skeleton approximation
# Reference from Motion Diffusion Model (MDM) / HumanML3D standard paramUtil.py
# This confirms our previous hypothesis:
# 0: Root
# 11: Left Hip ... 15: Left Toe
# 16: Right Hip ... 20: Right Toe
# 1, 2, 3, 4: Spine -> Head
# 5, 6, 7: Left Arm
# 8, 9, 10: Right Arm

SKELETON_EDGES = [
    # Spine / Head
    (0, 1), (1, 2), (2, 3), (3, 4),
    
    # Left Arm (from Neck index 3)
    (3, 5), (5, 6), (6, 7),
    
    # Right Arm (from Neck index 3)
    (3, 8), (8, 9), (9, 10),
    
    # Left Leg (from Root 0)
    (0, 11), (11, 12), (12, 13), (13, 14), (14, 15),
    
    # Right Leg (from Root 0)
    (0, 16), (16, 17), (17, 18), (18, 19), (19, 20)
]
# 注意：KIT-ML 的具体 21 关键点顺序如果没有文档很难猜对。
# 建议先看点云 (Scatter Plot)，点云如果是人形，说明数据解析正确。

def parse_motion_tokens(token_str, num_joints=21):
    """
    将空格分隔的数字字符串解析为 (T, Joints, 3) 的 numpy 数组
    """
    tokens = list(map(int, token_str.strip().split()))
    motion_flat = np.array(tokens)
    
    # 检查形状
    # 每一帧有 num_joints * 3 个数值
    frame_size = num_joints * 3
    if len(motion_flat) % frame_size != 0:
        print(f"Warning: Token count {len(motion_flat)} is not divisible by frame size {frame_size}.")
        # 截断多余的
        valid_len = (len(motion_flat) // frame_size) * frame_size
        motion_flat = motion_flat[:valid_len]
        
    num_frames = len(motion_flat) // frame_size
    motion_data = motion_flat.reshape(num_frames, num_joints, 3)
    return motion_data

def plot_motion(motion_data, output_file="motion_viz.gif", fps=10):
    """
    绘制 3D 动画
    motion_data: (T, 21, 3)
    """
    # 由于数据是量化后的 (0-255)，直接画出来形状是对的，但没有物理单位
    # 我们可以交换坐标轴以符合 matplotlib 的默认视角 (通常 Y 是高，但在某些动捕数据中 Z 是高)
    # 这里假设原始数据是 (x, y, z)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置固定的坐标轴范围，防止画面抖动
    # 获取全局最大最小值
    min_val = motion_data.min()
    max_val = motion_data.max()
    
    # 检测数据范围来决定半径
    data_range = max_val - min_val
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    
    # 初始化绘图对象
    # 减小点的大小 (s=10 -> s=5)，减小线宽 (linewidth=1)
    scatter = ax.scatter([], [], [], c='r', marker='o', s=5)
    lines = [ax.plot([], [], [], 'b-', linewidth=1)[0] for _ in SKELETON_EDGES]
    
    def update(frame_idx):
        frame_data = motion_data[frame_idx] # (21, 3)
        
        # 更新点 - 交换 Y 和 Z 轴，让 XZ 成为水平面，Y 成为垂直高度
        xs = frame_data[:, 0]
        ys = frame_data[:, 2] # 原来的 Z 变成现在的 Y (深度)
        zs = frame_data[:, 1] # 原来的 Y 变成现在的 Z (高度/垂直)
        
        # 动态调整坐标轴范围 (让画面紧跟人物，或者保持一个较大的固定范围)
        # 这里我们保持一个以人为中心的固定框，稍微放大一点视野
        center_x, center_y, center_z = xs.mean(), ys.mean(), zs.mean()
        
        # 使用动态计算的 plot_radius
        plot_radius = data_range * 0.4 # 缩小视野范围以放大人物
        
        ax.set_xlim(center_x - plot_radius, center_x + plot_radius)
        ax.set_ylim(center_y - plot_radius, center_y + plot_radius)
        ax.set_zlim(center_z - plot_radius, center_z + plot_radius)
        
        # Matplotlib 3D scatter 需要特殊的输入格式 (x, y, z)
        scatter._offsets3d = (xs, ys, zs)
        
        # 移除旧的文本标签 (如果存在)
        if hasattr(update, "labels"):
            for label in update.labels:
                label.remove()
        update.labels = []
        
        # 添加新的文本标签 (显示关节索引) - 字体改小
        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            label = ax.text(x, y, z, str(i), fontsize=6, color='black')
            update.labels.append(label)
        
        # 更新线
        if SKELETON_EDGES:
            for line, (start, end) in zip(lines, SKELETON_EDGES):
                # 只有当索引在范围内时才画线
                if start < len(frame_data) and end < len(frame_data):
                    line.set_data([xs[start], xs[end]], [ys[start], ys[end]])
                    line.set_3d_properties([zs[start], zs[end]])
        
        ax.set_title(f"Frame {frame_idx}/{len(motion_data)}")
        return scatter, *lines
    
    ani = FuncAnimation(fig, update, frames=len(motion_data), interval=1000/fps, blit=False)
    
    print(f"Saving animation to {output_file}...")
    ani.save(output_file, writer='pillow', fps=fps)
    print("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='Path to a .npy file to visualize directly')
    args = parser.parse_args()

    if args.file:
        # 直接读取 .npy 文件模式
        try:
            print(f"Loading motion from {args.file}...")
            motion_data = np.load(args.file)
            # 如果是 (40, 21, 3) 这种已经 reshape 好的
            print(f"Shape: {motion_data.shape}")
            plot_motion(motion_data, output_file="dataset_viz.gif")
            return
        except Exception as e:
            print(f"Error loading file: {e}")
            return

    # 交互模式
    print("Paste your motion token sequence here (press Enter twice to finish):")
    # 读取多行输入直到空行
    lines = []
    while True:
        try:
            line = input()
            if not line:
                break
            lines.append(line)
        except EOFError:
            break
    
    raw_input = " ".join(lines)
    
    if not raw_input.strip():
        print("No input provided. Exiting.")
        return

    # 过滤掉非数字字符 (以防包含 <|im_start|> 等)
    # 这里简单地提取所有数字
    import re
    tokens = re.findall(r'\d+', raw_input)
    clean_input = " ".join(tokens)

    motion_data = parse_motion_tokens(clean_input)
    print(f"Parsed motion data shape: {motion_data.shape}")
    
    if motion_data.shape[0] == 0:
        print("Error: No valid frames parsed.")
        return

    plot_motion(motion_data)

if __name__ == "__main__":
    main()

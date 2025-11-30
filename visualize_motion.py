import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import argparse

# KIT-ML / MMM (Master Motor Map) 21-joint skeleton approximation
# Reference from Motion Diffusion Model (MDM) / HumanML3D standard paramUtil.py
KINEMATIC_CHAIN = [
    # 左腿: 0,11,12,13,14,15
    [0, 11, 12, 13, 14, 15],
    # 右腿: 0,16,17,18,19,20
    [0, 16, 17, 18, 19, 20],
    # 躯干: 0,1,2,3,4
    [0, 1, 2, 3, 4],
    # 左臂: 3,5,6,7
    [3, 5, 6, 7],
    # 右臂: 3,8,9,10
    [3, 8, 9, 10]
]

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

def plot_xzPlane(ax, minx, maxx, minz, maxz):
    """绘制地面阴影区域"""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    verts = [
        [minx, 0, minz],
        [minx, 0, maxz],
        [maxx, 0, maxz],
        [maxx, 0, minz]
    ]
    xz_plane = Poly3DCollection([verts])
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.2)) # 半透明灰色
    ax.add_collection3d(xz_plane)

def plot_motion(motion_data, output_file="motion_viz.gif", fps=10):
    """
    绘制 3D 动画 (MDM 风格)
    """
    # 定义颜色 (MDM 风格)
    # 顺序: 左腿, 右腿, 躯干, 左臂, 右臂
    # 对应上面的 kinematic_chain 顺序
    colors = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"] # 蓝色系/绿色系

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # 数据预处理：交换轴以适应 Matplotlib (x, y, z) -> (x, z, y)
    # 并且让 Y 轴向上。原始数据中 Y 是高。
    # 我们在 update 里处理这个映射，但在计算范围时要注意。
    
    # 计算全局范围
    xs = motion_data[:, :, 0]
    ys = motion_data[:, :, 1] # Up
    zs = motion_data[:, :, 2]
    
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    min_z, max_z = zs.min(), zs.max()
    
    # 固定的视野半径
    radius = max(max_x - min_x, max_y - min_y, max_z - min_z) * 0.6
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2
    mid_z = (min_z + max_z) / 2

    # 初始化线条对象
    lines = []
    for chain, color in zip(KINEMATIC_CHAIN, colors):
        # 躯干(索引2)粗一点，其他细一点
        lw = 4.0 if chain == KINEMATIC_CHAIN[2] else 2.0
        lines.append(ax.plot([], [], [], '-', linewidth=lw, color=color)[0])

    def init():
        ax.set_xlim(mid_x - radius, mid_x + radius)
        ax.set_ylim(min_z - radius, min_z + radius) # Matplotlib Y = Real Z
        ax.set_zlim(min_y, min_y + 2*radius)        # Matplotlib Z = Real Y (Height)
        
        # 隐藏坐标轴和背景
        ax.set_axis_off()
        ax.grid(False)
        
        # 画地面 (y=0 in real world -> z=0 in matplotlib?)
        # 这里的地面高度取数据的最低 Y 值
        plot_xzPlane(ax, mid_x - radius, mid_x + radius, mid_z - radius, mid_z + radius)
        return lines

    def update(frame_idx):
        frame_data = motion_data[frame_idx] # (21, 3)
        
        # 映射: 
        # Real X -> Plot X
        # Real Y (Height) -> Plot Z
        # Real Z (Depth) -> Plot Y
        
        for i, (chain, line) in enumerate(zip(KINEMATIC_CHAIN, lines)):
            x_data = frame_data[chain, 0]
            y_data = frame_data[chain, 2] # Swap Y/Z
            z_data = frame_data[chain, 1] # Height
            
            line.set_data(x_data, y_data)
            line.set_3d_properties(z_data)
            
        ax.set_title(f"Frame {frame_idx}/{len(motion_data)}", fontsize=10)
        return lines
    
    ani = FuncAnimation(fig, update, frames=len(motion_data), init_func=init, interval=1000/fps, blit=False)
    
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

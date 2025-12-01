import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def analyze_motion(npy_file, output_prefix="analysis"):
    print(f"Loading motion data from {npy_file}...")
    try:
        motion_data = np.load(npy_file)
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    # Shape check: (T, Joints, 3)
    print(f"Data shape: {motion_data.shape}")
    if motion_data.ndim != 3 or motion_data.shape[2] != 3:
        print("Error: Expected shape (T, Joints, 3)")
        return
        
    frames, joints, dims = motion_data.shape
    
    # 1. 基础统计
    print("-" * 30)
    print(f"Total Frames: {frames}")
    print(f"Total Joints: {joints}")
    print(f"Value Range: Min={motion_data.min():.4f}, Max={motion_data.max():.4f}")
    print(f"Mean Value: {motion_data.mean():.4f}")
    print(f"Standard Deviation: {motion_data.std():.4f}")
    
    # 2. 计算平均速度 (帧间位移)
    # (T-1, Joints, 3)
    diff = motion_data[1:] - motion_data[:-1]
    # (T-1, Joints) - Euclidean distance per joint
    dist_per_joint = np.linalg.norm(diff, axis=2)
    # (T-1,) - Average velocity across all joints
    avg_velocity = dist_per_joint.mean(axis=1)
    
    print(f"Avg Velocity per frame: {avg_velocity.mean():.6f}")
    print(f"Max Velocity: {avg_velocity.max():.6f}")
    if avg_velocity.mean() < 1e-3:
        print("\n⚠️ WARNING: The motion seems extremely static! (Velocity is near zero)")
        print("Possible reasons:")
        print("1. Denormalization was skipped (data is still standardized).")
        print("2. The model generated repeated tokens.")
    print("-" * 30)

    # 3. 绘图：平均速度
    plt.figure(figsize=(12, 4))
    plt.plot(avg_velocity, label="Avg Velocity (All Joints)", color='black', linewidth=1)
    plt.title("Motion Velocity Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Displacement per Frame")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{output_prefix}_velocity.png")
    print(f"Saved velocity plot to {output_prefix}_velocity.png")
    
    # 4. 绘图：关键关节轨迹
    # 选取几个有代表性的关节：0(Root/Hips), 7(Left Hand), 10(Right Hand), 15(Left Foot), 20(Right Foot)
    # 注意：这里的索引基于 KIT-ML/MMM 骨骼结构
    key_joints = {
        0: "Root",
        7: "L_Hand",
        10: "R_Hand",
        15: "L_Foot",
        20: "R_Foot"
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    dim_names = ['X (Side-to-Side)', 'Y (Height)', 'Z (Forward/Back)']
    
    # 检查关节索引是否越界
    valid_joints = {k: v for k, v in key_joints.items() if k < joints}
    
    for dim_idx, ax in enumerate(axes):
        for j_idx, name in valid_joints.items():
            # 画出该关节在该维度的变化曲线
            ax.plot(motion_data[:, j_idx, dim_idx], label=name, linewidth=1.5, alpha=0.8)
        
        ax.set_ylabel(dim_names[dim_idx])
        ax.set_title(f"Trajectory - {dim_names[dim_idx]}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', ncol=len(valid_joints))
        
        # 标注该维度的变化范围
        dim_range = motion_data[:, :, dim_idx].max() - motion_data[:, :, dim_idx].min()
        ax.text(0.01, 0.95, f"Range span: {dim_range:.4f}", transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axes[-1].set_xlabel("Frame Index")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_trajectory.png")
    print(f"Saved trajectory plot to {output_prefix}_trajectory.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to the .npy motion file")
    parser.add_argument("--output", type=str, default="analysis", help="Prefix for output images")
    args = parser.parse_args()
    
    analyze_motion(args.file, args.output)


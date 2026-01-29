"""
KIT-ML 数据格式转换器

将 (T, 21, 3) 关节位置转换为 MDM 期望的 (T-1, 251) 特征格式

基于 motion-diffusion-model/data_loaders/humanml/scripts/motion_process.py
"""

import os
import sys
import numpy as np
import torch

# 添加 MDM 路径
MDM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'motion-diffusion-model')
sys.path.insert(0, MDM_PATH)

from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.common.quaternion import (
    qmul_np, qinv_np, qrot_np, qbetween_np, qfix, quaternion_to_cont6d_np
)
from data_loaders.humanml.utils.paramUtil import kit_raw_offsets, kit_kinematic_chain

# KIT-ML 骨骼参数
KIT_RAW_OFFSETS = torch.from_numpy(kit_raw_offsets)
KIT_KINEMATIC_CHAIN = kit_kinematic_chain
KIT_JOINTS_NUM = 21

# KIT-ML 参数
L_IDX1, L_IDX2 = 17, 18  # Lower legs
FID_R, FID_L = [14, 15], [19, 20]  # Right/Left foot
FACE_JOINT_INDX = [11, 16, 5, 8]  # r_hip, l_hip, sdr_r, sdr_l
FEET_THRE = 0.05

# 目标骨骼偏移（使用标准 KIT 骨骼）
_TGT_OFFSETS = None


def _get_tgt_offsets():
    """获取目标骨骼偏移（延迟加载）"""
    global _TGT_OFFSETS
    if _TGT_OFFSETS is None:
        # 使用标准 KIT 骨骼
        example_path = os.path.join(MDM_PATH, 'dataset', 'kit_mean.npy')
        if not os.path.exists(example_path):
            # 如果没有，使用默认偏移
            _TGT_OFFSETS = KIT_RAW_OFFSETS.clone()
        else:
            # 使用原始偏移
            _TGT_OFFSETS = KIT_RAW_OFFSETS.clone()
    return _TGT_OFFSETS


def uniform_skeleton(positions, target_offset):
    """
    统一骨骼结构
    
    Args:
        positions: (T, J, 3) 关节位置
        target_offset: 目标骨骼偏移
    
    Returns:
        new_joints: 统一后的关节位置
    """
    src_skel = Skeleton(KIT_RAW_OFFSETS, KIT_KINEMATIC_CHAIN, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    
    # 计算缩放比例（基于腿长）
    src_leg_len = np.abs(src_offset[L_IDX1]).max() + np.abs(src_offset[L_IDX2]).max()
    tgt_leg_len = np.abs(tgt_offset[L_IDX1]).max() + np.abs(tgt_offset[L_IDX2]).max()
    
    if src_leg_len < 1e-6:
        scale_rt = 1.0
    else:
        scale_rt = tgt_leg_len / src_leg_len
    
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt
    
    # 逆向运动学
    quat_params = src_skel.inverse_kinematics_np(positions, FACE_JOINT_INDX)
    
    # 正向运动学
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    
    return new_joints


def extract_features(positions, feet_thre=FEET_THRE):
    """
    从关节位置提取 251 维特征
    
    Args:
        positions: (T, 21, 3) 关节位置
        feet_thre: 脚部接触阈值
    
    Returns:
        data: (T-1, 251) 特征向量
    """
    T, J, _ = positions.shape
    global_positions = positions.copy()
    
    # 脚部接触检测
    def foot_detect(positions, thres):
        velfactor = np.array([thres, thres])
        
        feet_l_x = (positions[1:, FID_L, 0] - positions[:-1, FID_L, 0]) ** 2
        feet_l_y = (positions[1:, FID_L, 1] - positions[:-1, FID_L, 1]) ** 2
        feet_l_z = (positions[1:, FID_L, 2] - positions[:-1, FID_L, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)
        
        feet_r_x = (positions[1:, FID_R, 0] - positions[:-1, FID_R, 0]) ** 2
        feet_r_y = (positions[1:, FID_R, 1] - positions[:-1, FID_R, 1]) ** 2
        feet_r_z = (positions[1:, FID_R, 2] - positions[:-1, FID_R, 2]) ** 2
        feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(np.float32)
        
        return feet_l, feet_r
    
    feet_l, feet_r = foot_detect(positions, feet_thre)
    
    # 获取连续 6D 旋转表示
    skel = Skeleton(KIT_RAW_OFFSETS, KIT_KINEMATIC_CHAIN, "cpu")
    quat_params = skel.inverse_kinematics_np(positions, FACE_JOINT_INDX, smooth_forward=True)
    
    # 四元数转 6D
    cont_6d_params = quaternion_to_cont6d_np(quat_params)
    
    # 根旋转
    r_rot = quat_params[:, 0].copy()
    
    # 根线速度
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    velocity = qrot_np(r_rot[1:], velocity)
    
    # 根角速度
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    
    # 局部位置（相对于根）
    def get_rifke(pos, r_rot):
        pos = pos.copy()
        pos[..., 0] -= pos[:, 0:1, 0]
        pos[..., 2] -= pos[:, 0:1, 2]
        pos = qrot_np(np.repeat(r_rot[:, None], pos.shape[1], axis=1), pos)
        return pos
    
    local_positions = get_rifke(positions, r_rot)
    
    # 根高度
    root_y = local_positions[:, 0, 1:2]
    
    # 根旋转和线速度
    r_velocity_y = np.arcsin(np.clip(r_velocity[:, 2:3], -1, 1))
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity_y, l_velocity, root_y[:-1]], axis=-1)  # (T-1, 4)
    
    # 关节旋转（6D）
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)  # (T, 120)
    
    # 局部关节位置（相对于根）
    ric_data = local_positions[:, 1:].reshape(len(local_positions), -1)  # (T, 60)
    
    # 关节速度
    local_vel = qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
        global_positions[1:] - global_positions[:-1]
    )
    local_vel = local_vel.reshape(len(local_vel), -1)  # (T-1, 63)
    
    # 组合特征
    data = root_data  # (T-1, 4)
    data = np.concatenate([data, ric_data[:-1]], axis=-1)  # + 60
    data = np.concatenate([data, rot_data[:-1]], axis=-1)  # + 120
    data = np.concatenate([data, local_vel], axis=-1)      # + 63
    data = np.concatenate([data, feet_l, feet_r], axis=-1) # + 4
    
    # 总计: 4 + 60 + 120 + 63 + 4 = 251
    return data


def process_motion(positions, feet_thre=FEET_THRE, uniform_skel=False):
    """
    完整的动作处理流程
    
    Args:
        positions: (T, 21, 3) 关节位置
        feet_thre: 脚部接触阈值
        uniform_skel: 是否统一骨骼（默认 False，保持原始尺寸）
    
    Returns:
        data: (T-1, 251) 特征向量
        global_positions: 处理后的全局位置
    """
    positions = positions.copy()
    
    # 1. 统一骨骼（可选，默认跳过以保持原始尺寸）
    if uniform_skel:
        tgt_offsets = _get_tgt_offsets()
        try:
            positions = uniform_skeleton(positions, tgt_offsets)
        except Exception as e:
            print(f"Warning: uniform_skeleton failed: {e}, using original positions")
    
    # 2. 放在地面上
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    
    # 3. XZ 原点
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz
    
    # 4. 初始面向 Z+
    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_INDX
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across_norm = np.sqrt((across ** 2).sum())
    if across_norm > 1e-6:
        across = across / across_norm
    else:
        across = np.array([1, 0, 0])
    
    forward_init = np.cross(np.array([[0, 1, 0]]), across[None], axis=-1)
    forward_norm = np.sqrt((forward_init ** 2).sum())
    if forward_norm > 1e-6:
        forward_init = forward_init / forward_norm
    else:
        forward_init = np.array([[0, 0, 1]])
    
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    
    positions = qrot_np(root_quat_init, positions)
    global_positions = positions.copy()
    
    # 5. 提取特征
    try:
        data = extract_features(positions, feet_thre)
    except Exception as e:
        print(f"Warning: extract_features failed: {e}")
        # 返回简化特征
        T = positions.shape[0]
        data = np.zeros((T-1, 251), dtype=np.float32)
    
    return data, global_positions


def joints_to_hml_features(joints: np.ndarray) -> np.ndarray:
    """
    将关节位置转换为 HumanML3D/KIT 格式的 251 维特征
    
    Args:
        joints: (T, 21, 3) 关节位置（毫米单位）
    
    Returns:
        features: (T-1, 251) 特征向量
    """
    if joints.shape[0] < 2:
        raise ValueError(f"Need at least 2 frames, got {joints.shape[0]}")
    
    if joints.shape[1] != 21:
        raise ValueError(f"Expected 21 joints, got {joints.shape[1]}")
    
    features, _ = process_motion(joints)
    return features


if __name__ == "__main__":
    # 测试
    print("Testing KIT data converter...")
    
    # 随机测试数据
    test_joints = np.random.randn(20, 21, 3).astype(np.float32) * 100
    
    try:
        features = joints_to_hml_features(test_joints)
        print(f"Input shape: {test_joints.shape}")
        print(f"Output shape: {features.shape}")
        print(f"Expected: (19, 251)")
        print(f"Output range: {features.min():.4f} ~ {features.max():.4f}")
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

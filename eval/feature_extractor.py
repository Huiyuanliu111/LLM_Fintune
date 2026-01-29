"""
Motion Feature Extractor

将原始动作数据 (T, J, 3) 转换为用于评估的特征向量。
由于 LLM 微调项目没有预训练的 motion encoder，
我们使用基于统计的特征提取方法。
"""

import numpy as np
from typing import Optional, List, Tuple
from scipy.ndimage import gaussian_filter1d


class MotionFeatureExtractor:
    """
    动作特征提取器
    
    将原始关节位置数据转换为紧凑的特征表示，用于评估指标计算。
    
    特征包括:
    1. 全局统计特征 (均值、方差、范围等)
    2. 运动学特征 (速度、加速度)
    3. 关节间关系特征 (骨骼长度、对称性)
    4. 时序特征 (周期性、平稳性)
    """
    
    # KIT-ML 21 关节的骨骼连接定义
    KINEMATIC_CHAIN = [
        [0, 11, 12, 13, 14, 15],  # 左腿
        [0, 16, 17, 18, 19, 20],  # 右腿
        [0, 1, 2, 3, 4],          # 躯干
        [3, 5, 6, 7],             # 左臂
        [3, 8, 9, 10],            # 右臂
    ]
    
    # 骨骼边的定义 (parent, child)
    BONES = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # 躯干
        (3, 5), (5, 6), (6, 7),               # 左臂
        (3, 8), (8, 9), (9, 10),              # 右臂
        (0, 11), (11, 12), (12, 13), (13, 14), (14, 15),  # 左腿
        (0, 16), (16, 17), (17, 18), (18, 19), (19, 20),  # 右腿
    ]
    
    def __init__(
        self, 
        num_joints: int = 21,
        feature_dim: int = 256,
        use_velocity: bool = True,
        use_acceleration: bool = True,
        use_bone_features: bool = True,
        pooling: str = 'mean',  # 'mean', 'max', 'concat'
        normalize: bool = True,  # 是否对输入数据进行标准化
    ):
        """
        Args:
            num_joints: 关节数量 (KIT-ML 为 21)
            feature_dim: 输出特征维度 (用于对齐)
            use_velocity: 是否使用速度特征
            use_acceleration: 是否使用加速度特征
            use_bone_features: 是否使用骨骼特征
            pooling: 时序聚合方式
            normalize: 是否对输入数据进行标准化
        """
        self.num_joints = num_joints
        self.feature_dim = feature_dim
        self.use_velocity = use_velocity
        self.use_acceleration = use_acceleration
        self.use_bone_features = use_bone_features
        self.pooling = pooling
        self.normalize = normalize
    
    def extract_features(self, motion: np.ndarray) -> np.ndarray:
        """
        从单个动作序列提取特征
        
        Args:
            motion: (T, J, 3) 动作序列
        
        Returns:
            features: (D,) 特征向量
        """
        if len(motion.shape) != 3:
            raise ValueError(f"Expected (T, J, 3) input, got {motion.shape}")
        
        # 标准化：减均值除标准差，使数值在合理范围内
        if self.normalize:
            mean = motion.mean()
            std = motion.std()
            if std > 1e-8:
                motion = (motion - mean) / std
            else:
                motion = motion - mean
        
        T, J, _ = motion.shape
        features_list = []
        
        # 1. 位置统计特征
        pos_features = self._extract_position_features(motion)
        features_list.append(pos_features)
        
        # 2. 速度特征
        if self.use_velocity and T > 1:
            vel_features = self._extract_velocity_features(motion)
            features_list.append(vel_features)
        
        # 3. 加速度特征
        if self.use_acceleration and T > 2:
            acc_features = self._extract_acceleration_features(motion)
            features_list.append(acc_features)
        
        # 4. 骨骼特征
        if self.use_bone_features:
            bone_features = self._extract_bone_features(motion)
            features_list.append(bone_features)
        
        # 5. 轨迹特征 (根关节)
        traj_features = self._extract_trajectory_features(motion)
        features_list.append(traj_features)
        
        # 合并所有特征
        all_features = np.concatenate(features_list)
        
        # 调整到目标维度
        if len(all_features) < self.feature_dim:
            # 补零
            all_features = np.pad(all_features, (0, self.feature_dim - len(all_features)))
        elif len(all_features) > self.feature_dim:
            # PCA 或简单截断
            all_features = all_features[:self.feature_dim]
        
        # L2 归一化，使特征向量长度为 1
        norm = np.linalg.norm(all_features)
        if norm > 1e-8:
            all_features = all_features / norm
        
        return all_features
    
    def extract_batch_features(self, motions: List[np.ndarray]) -> np.ndarray:
        """
        批量提取特征
        
        Args:
            motions: 动作序列列表，每个为 (T, J, 3)
        
        Returns:
            features: (N, D) 特征矩阵
        """
        features = []
        for motion in motions:
            feat = self.extract_features(motion)
            features.append(feat)
        return np.stack(features, axis=0)
    
    def _extract_position_features(self, motion: np.ndarray) -> np.ndarray:
        """提取位置统计特征"""
        T, J, _ = motion.shape
        
        # 每个关节的统计量
        joint_mean = motion.mean(axis=0)  # (J, 3)
        joint_std = motion.std(axis=0)    # (J, 3)
        joint_min = motion.min(axis=0)    # (J, 3)
        joint_max = motion.max(axis=0)    # (J, 3)
        
        # 全局统计量
        global_mean = motion.mean()
        global_std = motion.std()
        
        # 相对于根关节 (joint 0) 的位置
        root_relative = motion - motion[:, 0:1, :]  # (T, J, 3)
        rel_mean = root_relative.mean(axis=0)
        rel_std = root_relative.std(axis=0)
        
        features = np.concatenate([
            joint_mean.flatten(),
            joint_std.flatten(),
            (joint_max - joint_min).flatten(),  # 关节运动范围
            rel_mean.flatten(),
            rel_std.flatten(),
            [global_mean, global_std],
        ])
        
        return features
    
    def _extract_velocity_features(self, motion: np.ndarray) -> np.ndarray:
        """提取速度特征"""
        velocity = np.diff(motion, axis=0)  # (T-1, J, 3)
        
        # 速度统计
        vel_mean = velocity.mean(axis=0)  # (J, 3)
        vel_std = velocity.std(axis=0)    # (J, 3)
        vel_abs_mean = np.abs(velocity).mean(axis=0)  # 绝对速度
        
        # 全局速度
        speed = np.linalg.norm(velocity, axis=2)  # (T-1, J)
        avg_speed = speed.mean()
        max_speed = speed.max()
        
        features = np.concatenate([
            vel_mean.flatten(),
            vel_std.flatten(),
            vel_abs_mean.flatten(),
            [avg_speed, max_speed],
        ])
        
        return features
    
    def _extract_acceleration_features(self, motion: np.ndarray) -> np.ndarray:
        """提取加速度特征"""
        velocity = np.diff(motion, axis=0)
        acceleration = np.diff(velocity, axis=0)  # (T-2, J, 3)
        
        # 加速度统计
        acc_mean = acceleration.mean(axis=0)
        acc_std = acceleration.std(axis=0)
        acc_abs_mean = np.abs(acceleration).mean(axis=0)
        
        # 平滑度 (加速度的变化率)
        jerk = np.diff(acceleration, axis=0) if acceleration.shape[0] > 1 else np.zeros_like(acceleration)
        smoothness = np.abs(jerk).mean()
        
        features = np.concatenate([
            acc_mean.flatten(),
            acc_std.flatten(),
            acc_abs_mean.flatten(),
            [smoothness],
        ])
        
        return features
    
    def _extract_bone_features(self, motion: np.ndarray) -> np.ndarray:
        """提取骨骼特征"""
        T, J, _ = motion.shape
        
        bone_lengths = []
        bone_length_vars = []
        
        for parent, child in self.BONES:
            if parent < J and child < J:
                # 计算骨骼长度
                bone_vec = motion[:, child, :] - motion[:, parent, :]  # (T, 3)
                lengths = np.linalg.norm(bone_vec, axis=1)  # (T,)
                
                bone_lengths.append(lengths.mean())
                bone_length_vars.append(lengths.std())
        
        # 对称性特征 (左右肢体)
        # 左腿 vs 右腿
        left_leg_joints = [11, 12, 13, 14, 15]
        right_leg_joints = [16, 17, 18, 19, 20]
        
        symmetry_scores = []
        for l, r in zip(left_leg_joints, right_leg_joints):
            if l < J and r < J:
                # 计算左右对应关节的运动相似度
                left_motion = motion[:, l, :]
                right_motion = motion[:, r, :]
                # 镜像后的相关性
                right_mirrored = right_motion.copy()
                right_mirrored[:, 0] = -right_mirrored[:, 0]  # 镜像 X 轴
                
                corr = np.corrcoef(left_motion.flatten(), right_mirrored.flatten())[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                symmetry_scores.append(corr)
        
        features = np.concatenate([
            np.array(bone_lengths),
            np.array(bone_length_vars),
            np.array(symmetry_scores),
        ])
        
        return features
    
    def _extract_trajectory_features(self, motion: np.ndarray) -> np.ndarray:
        """提取轨迹特征 (根关节运动)"""
        root_traj = motion[:, 0, :]  # (T, 3)
        T = root_traj.shape[0]
        
        # 总位移
        total_displacement = np.linalg.norm(root_traj[-1] - root_traj[0])
        
        # 路径长度
        if T > 1:
            path_length = np.sum(np.linalg.norm(np.diff(root_traj, axis=0), axis=1))
        else:
            path_length = 0.0
        
        # 方向变化
        if T > 2:
            directions = np.diff(root_traj, axis=0)
            dir_norms = np.linalg.norm(directions, axis=1, keepdims=True)
            dir_norms = np.maximum(dir_norms, 1e-8)  # 避免除零
            directions = directions / dir_norms
            
            # 方向变化的角度
            dot_products = np.sum(directions[:-1] * directions[1:], axis=1)
            dot_products = np.clip(dot_products, -1, 1)
            angles = np.arccos(dot_products)
            mean_direction_change = angles.mean()
        else:
            mean_direction_change = 0.0
        
        # 垂直运动 (高度变化)
        height_range = root_traj[:, 1].max() - root_traj[:, 1].min()
        height_var = root_traj[:, 1].std()
        
        features = np.array([
            total_displacement,
            path_length,
            mean_direction_change,
            height_range,
            height_var,
            T,  # 序列长度也是一个特征
        ])
        
        return features


class SimpleTextFeatureExtractor:
    """
    简单的文本特征提取器
    
    使用词袋模型或简单的嵌入来提取文本特征。
    用于计算 Matching Score 和 R-precision。
    """
    
    # 常见的动作词汇表
    ACTION_VOCAB = [
        'walk', 'run', 'jump', 'sit', 'stand', 'turn', 'wave', 'kick',
        'punch', 'throw', 'catch', 'pick', 'put', 'push', 'pull',
        'bend', 'stretch', 'squat', 'kneel', 'crawl', 'climb', 'dance',
        'bow', 'clap', 'shake', 'nod', 'point', 'reach', 'grab', 'lift',
        'forward', 'backward', 'left', 'right', 'up', 'down', 'around',
        'slowly', 'quickly', 'fast', 'slow', 'twice', 'once', 'repeatedly',
        'person', 'man', 'woman', 'human', 'someone', 'figure',
        'arm', 'leg', 'hand', 'foot', 'head', 'body',
    ]
    
    def __init__(self, feature_dim: int = 256):
        self.feature_dim = feature_dim
        self.vocab = {word: i for i, word in enumerate(self.ACTION_VOCAB)}
    
    def extract_features(self, text: str) -> np.ndarray:
        """
        从文本提取特征
        
        Args:
            text: 文本描述
        
        Returns:
            features: (D,) 特征向量
        """
        text = text.lower()
        words = text.split()
        
        # 词袋特征
        bow = np.zeros(len(self.ACTION_VOCAB))
        for word in words:
            if word in self.vocab:
                bow[self.vocab[word]] = 1
        
        # 简单的 n-gram 特征 (bigram)
        bigrams = []
        for i in range(len(words) - 1):
            bigrams.append(f"{words[i]}_{words[i+1]}")
        
        # 文本长度特征
        length_features = np.array([
            len(words),
            len(text),
            len(set(words)),  # 唯一词数
        ])
        
        # 合并特征
        features = np.concatenate([bow, length_features])
        
        # 调整维度
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        elif len(features) > self.feature_dim:
            features = features[:self.feature_dim]
        
        # L2 归一化
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features = features / norm
        
        return features
    
    def extract_batch_features(self, texts: List[str]) -> np.ndarray:
        """批量提取文本特征"""
        features = []
        for text in texts:
            feat = self.extract_features(text)
            features.append(feat)
        return np.stack(features, axis=0)

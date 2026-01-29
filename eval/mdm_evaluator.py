"""
MDM 预训练 Evaluator 集成

使用 Motion Diffusion Model 的预训练 motion encoder 计算真正的 FID。

使用前需要：
1. 下载预训练模型: cd motion-diffusion-model && bash prepare/download_t2m_evaluators.sh
   或手动下载 kit.zip 并解压
2. 确保 kit/text_mot_match/model/finest.tar 存在
"""

import os
import sys
import numpy as np
import torch
from typing import List, Optional, Tuple

# 添加 MDM 路径
MDM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'motion-diffusion-model')
sys.path.insert(0, MDM_PATH)

# LLM_Fintune 根目录（kit 文件夹可以放在这里）
LLM_FINTUNE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from data_loaders.humanml.utils.metrics import (
    calculate_activation_statistics,
    calculate_frechet_distance,
    calculate_diversity,
    euclidean_distance_matrix,
    calculate_top_k,
)

# 导入正确的数据转换器
try:
    from eval.kit_data_converter import joints_to_hml_features
    HML_CONVERTER_AVAILABLE = True
except ImportError:
    HML_CONVERTER_AVAILABLE = False
    print("Warning: kit_data_converter not available, using simplified features")


def get_kit_model_path():
    """查找 kit 预训练模型的路径"""
    # 首先在 LLM_Fintune 目录找
    path1 = os.path.join(LLM_FINTUNE_PATH, 'kit', 'text_mot_match', 'model', 'finest.tar')
    if os.path.exists(path1):
        return os.path.join(LLM_FINTUNE_PATH, 'kit')
    
    # 然后在 motion-diffusion-model 目录找
    path2 = os.path.join(MDM_PATH, 'kit', 'text_mot_match', 'model', 'finest.tar')
    if os.path.exists(path2):
        return os.path.join(MDM_PATH, 'kit')
    
    return None


class MDMEvaluator:
    """
    使用 MDM 预训练模型的评估器
    """
    
    def __init__(self, dataset_name: str = 'kit', device: str = 'cuda'):
        """
        Args:
            dataset_name: 'kit' 或 'humanml'
            device: 'cuda' 或 'cpu'
        """
        self.dataset_name = dataset_name
        self.device = device
        
        # 查找预训练模型路径
        kit_path = get_kit_model_path()
        if kit_path is None:
            raise FileNotFoundError(
                f"预训练模型不存在!\n"
                f"请将 kit.zip 解压到以下任一位置:\n"
                f"  1. {LLM_FINTUNE_PATH}\n"
                f"  2. {MDM_PATH}"
            )
        
        self.kit_base_path = os.path.dirname(kit_path)  # kit 文件夹的父目录
        print(f"Found kit model at: {kit_path}")
        
        # 加载 mean 和 std（从 motion-diffusion-model/dataset）
        self.mean = np.load(os.path.join(MDM_PATH, 'dataset', f'{dataset_name}_mean.npy'))
        self.std = np.load(os.path.join(MDM_PATH, 'dataset', f'{dataset_name}_std.npy'))
        
        print(f"Loading MDM evaluator for {dataset_name}...")
        
        # 需要切换到 kit 所在目录，因为模型加载时使用相对路径
        original_cwd = os.getcwd()
        os.chdir(self.kit_base_path)
        
        try:
            self.eval_wrapper = EvaluatorMDMWrapper(dataset_name, device)
        finally:
            os.chdir(original_cwd)
        
        print("MDM evaluator loaded successfully!")
        
        # KIT 数据集参数
        self.joints_num = 21 if dataset_name == 'kit' else 22
        self.dim_pose = 251 if dataset_name == 'kit' else 263
    
    def joints_to_features(self, joints: np.ndarray) -> np.ndarray:
        """
        将关节位置转换为 MDM 期望的 251 维特征格式
        
        Args:
            joints: (T, 21, 3) 关节位置
        
        Returns:
            features: (T-1, 251) 特征向量
        """
        if HML_CONVERTER_AVAILABLE:
            # 使用完整的转换器
            try:
                return joints_to_hml_features(joints)
            except Exception as e:
                print(f"Warning: HML conversion failed: {e}, using simplified features")
        
        # 简化版后备方案
        T, J, _ = joints.shape
        
        # Root 数据 (简化)
        root_pos = joints[:, 0]  # (T, 3)
        root_vel = root_pos[1:] - root_pos[:-1]
        root_data = np.zeros((T-1, 4))
        root_data[:, 1:3] = root_vel[:, [0, 2]]  # xz velocity
        root_data[:, 3] = root_pos[:-1, 1]  # y position
        
        # Local joint positions (相对于 root)
        local_joints = joints[:, 1:] - joints[:, 0:1]  # (T, 20, 3)
        ric_data = local_joints[:-1].reshape(T-1, -1)  # (T-1, 60)
        
        # 关节旋转 (简化：使用零)
        rot_data = np.zeros((T-1, (J-1) * 6))  # (T-1, 120)
        
        # 关节速度
        joint_vel = joints[1:] - joints[:-1]  # (T-1, 21, 3)
        local_vel = joint_vel.reshape(T-1, -1)  # (T-1, 63)
        
        # 脚部接触 (简化：使用零)
        feet_contact = np.zeros((T-1, 4))
        
        # 组合
        features = np.concatenate([
            root_data,      # 4
            ric_data,       # 60
            rot_data,       # 120
            local_vel,      # 63
            feet_contact,   # 4
        ], axis=-1)  # Total: 251
        
        return features
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        """归一化"""
        return (features - self.mean) / self.std
    
    def get_motion_embeddings(
        self, 
        motions: List[np.ndarray],
        use_simple_features: bool = True,
    ) -> np.ndarray:
        """
        获取动作的嵌入向量
        
        Args:
            motions: 动作列表，每个为 (T, 21, 3) 关节位置
            use_simple_features: 使用简化的特征转换（True）或期望已经是 251 维（False）
        
        Returns:
            embeddings: (N, 512) 嵌入向量
        """
        all_embeddings = []
        
        for i, motion in enumerate(motions):
            if use_simple_features:
                # 转换为 251 维特征
                try:
                    features = self.joints_to_features(motion)
                except Exception as e:
                    print(f"Warning: Feature conversion failed for motion {i}: {e}")
                    # 使用零特征作为后备
                    features = np.zeros((motion.shape[0] - 1, 251), dtype=np.float32)
            else:
                features = motion
            
            # 检查 NaN/Inf
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                print(f"Warning: NaN/Inf detected in features for motion {i}, replacing with zeros")
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 归一化
            features_norm = self.normalize(features)
            
            # 再次检查 NaN/Inf（归一化后可能产生）
            if np.any(np.isnan(features_norm)) or np.any(np.isinf(features_norm)):
                print(f"Warning: NaN/Inf after normalization for motion {i}, replacing with zeros")
                features_norm = np.nan_to_num(features_norm, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 转为 tensor
            motion_tensor = torch.from_numpy(features_norm).unsqueeze(0).float().to(self.device)
            m_lens = torch.tensor([features_norm.shape[0]]).to(self.device)
            
            # 获取嵌入
            embedding = self.eval_wrapper.get_motion_embeddings(motion_tensor, m_lens)
            all_embeddings.append(embedding.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)
    
    def calculate_fid(
        self, 
        gt_motions: List[np.ndarray], 
        gen_motions: List[np.ndarray],
    ) -> float:
        """
        计算 FID
        
        Args:
            gt_motions: 真实动作列表
            gen_motions: 生成动作列表
        
        Returns:
            fid: FID 分数
        """
        print("Extracting GT embeddings...")
        gt_embeddings = self.get_motion_embeddings(gt_motions)
        
        print("Extracting generated embeddings...")
        gen_embeddings = self.get_motion_embeddings(gen_motions)
        
        print("Calculating FID...")
        gt_mu, gt_cov = calculate_activation_statistics(gt_embeddings)
        gen_mu, gen_cov = calculate_activation_statistics(gen_embeddings)
        
        # 检查 NaN/Inf
        if np.any(np.isnan(gt_cov)) or np.any(np.isnan(gen_cov)):
            print("Warning: NaN in covariance matrix, returning large FID")
            return 999999.0
        
        try:
            fid = calculate_frechet_distance(gt_mu, gt_cov, gen_mu, gen_cov)
        except ValueError as e:
            print(f"Warning: FID calculation failed: {e}, returning large FID")
            fid = 999999.0
        
        return fid
    
    def calculate_diversity(
        self, 
        motions: List[np.ndarray],
        diversity_times: int = 300,
    ) -> float:
        """
        计算多样性（与 MDM 一致）
        
        注意：MDM 的 calculate_diversity 要求 num_samples > diversity_times (严格大于)
        """
        embeddings = self.get_motion_embeddings(motions)
        num_samples = embeddings.shape[0]
        
        # MDM 要求 num_samples > diversity_times (严格大于)
        # 所以 diversity_times 最大为 num_samples - 1
        if num_samples <= 2:
            print(f"Warning: Only {num_samples} samples, need > 2 for diversity")
            return -1.0  # 返回 -1 表示无法计算
        
        actual_diversity_times = min(diversity_times, num_samples - 1)
        
        if actual_diversity_times < diversity_times:
            print(f"Warning: diversity_times reduced to {actual_diversity_times} (num_samples={num_samples})")
        
        return calculate_diversity(embeddings, actual_diversity_times)
    
    def evaluate_all(
        self,
        gt_motions: List[np.ndarray],
        gen_motions: List[np.ndarray],
    ) -> dict:
        """
        计算所有指标
        """
        results = {}
        
        print("=" * 60)
        print("MDM Evaluator - Using Pretrained Motion Encoder")
        print("=" * 60)
        
        # FID
        print("\n[1/3] Calculating FID...")
        fid = self.calculate_fid(gt_motions, gen_motions)
        results['FID'] = fid
        print(f"FID: {fid:.4f}")
        
        # GT Diversity
        print("\n[2/3] Calculating GT Diversity...")
        gt_diversity = self.calculate_diversity(gt_motions)
        results['GT_Diversity'] = gt_diversity
        print(f"GT Diversity: {gt_diversity:.4f}")
        
        # Gen Diversity
        print("\n[3/3] Calculating Generated Diversity...")
        gen_diversity = self.calculate_diversity(gen_motions)
        results['Gen_Diversity'] = gen_diversity
        print(f"Generated Diversity: {gen_diversity:.4f}")
        
        return results


def check_mdm_available() -> bool:
    """检查 MDM evaluator 是否可用"""
    return get_kit_model_path() is not None


if __name__ == "__main__":
    # 测试
    if not check_mdm_available():
        print("MDM evaluator 不可用，请先下载预训练模型:")
        print(f"  cd {MDM_PATH}")
        print("  bash prepare/download_t2m_evaluators.sh")
    else:
        print("MDM evaluator 可用!")
        evaluator = MDMEvaluator('kit', 'cuda')
        
        # 测试数据
        gt_motion = np.random.randn(20, 21, 3).astype(np.float32)
        gen_motion = np.random.randn(20, 21, 3).astype(np.float32)
        
        results = evaluator.evaluate_all([gt_motion], [gen_motion])
        print(results)

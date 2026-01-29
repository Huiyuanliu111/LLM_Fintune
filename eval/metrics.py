"""
Motion Evaluation Metrics
基于 Motion Diffusion Model (MDM) 的评估指标实现

包含以下指标:
- FID (Fréchet Inception Distance): 衡量生成动作与真实动作分布的距离
- Diversity: 衡量生成动作的多样性
- MultiModality: 衡量同一文本生成不同动作的能力
- Matching Score: 衡量文本与动作的匹配程度
- R-precision: 衡量文本-动作检索的精度
"""

import numpy as np
from scipy import linalg
from typing import Tuple, Optional


def euclidean_distance_matrix(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """
    计算两组特征之间的欧氏距离矩阵
    
    Args:
        matrix1: (N1, D) 第一组特征
        matrix2: (N2, D) 第二组特征
    
    Returns:
        dist: (N1, N2) 距离矩阵，dist[i,j] = distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1], "Feature dimensions must match"
    
    # (X - Y)^2 = X^2 + Y^2 - 2*X*Y
    d1 = -2 * np.dot(matrix1, matrix2.T)  # (N1, N2)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)  # (N1, 1)
    d3 = np.sum(np.square(matrix2), axis=1)  # (N2,)
    dists = np.sqrt(np.maximum(d1 + d2 + d3, 0))  # 防止数值误差导致负数
    return dists


def calculate_activation_statistics(activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算特征的均值和协方差 (用于 FID 计算)
    
    Args:
        activations: (N, D) N 个样本的 D 维特征
    
    Returns:
        mu: (D,) 均值向量
        sigma: (D, D) 协方差矩阵
    """
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(
    mu1: np.ndarray, 
    sigma1: np.ndarray, 
    mu2: np.ndarray, 
    sigma2: np.ndarray, 
    eps: float = 1e-6
) -> float:
    """
    计算 Fréchet 距离 (FID 的核心)
    
    两个多元高斯分布 X_1 ~ N(mu_1, C_1) 和 X_2 ~ N(mu_2, C_2) 之间的 Fréchet 距离:
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
    
    Args:
        mu1, sigma1: 生成样本的均值和协方差
        mu2, sigma2: 真实样本的均值和协方差
        eps: 数值稳定性的小量
    
    Returns:
        fid: Fréchet 距离
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, f"Mean shape mismatch: {mu1.shape} vs {mu2.shape}"
    assert sigma1.shape == sigma2.shape, f"Covariance shape mismatch: {sigma1.shape} vs {sigma2.shape}"
    
    diff = mu1 - mu2
    
    # 计算 sqrt(sigma1 * sigma2)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # 处理数值不稳定
    if not np.isfinite(covmean).all():
        print(f"Warning: FID calculation produces singular product, adding {eps} to diagonal")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # 处理虚数部分 (数值误差)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component too large: {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_fid(
    gt_features: np.ndarray, 
    gen_features: np.ndarray
) -> float:
    """
    计算 FID (Fréchet Inception Distance)
    
    FID 衡量生成样本与真实样本在特征空间中的分布距离。
    越低越好，表示生成质量越接近真实数据。
    
    Args:
        gt_features: (N1, D) 真实样本的特征
        gen_features: (N2, D) 生成样本的特征
    
    Returns:
        fid: FID 分数
    """
    gt_mu, gt_sigma = calculate_activation_statistics(gt_features)
    gen_mu, gen_sigma = calculate_activation_statistics(gen_features)
    return calculate_frechet_distance(gen_mu, gen_sigma, gt_mu, gt_sigma)


def calculate_diversity(
    features: np.ndarray, 
    diversity_times: int = 300,
    seed: Optional[int] = None
) -> float:
    """
    计算动作多样性
    
    随机采样配对，计算特征空间中的平均 L2 距离。
    越高越好，表示生成的动作越多样。
    
    Args:
        features: (N, D) N 个样本的 D 维特征
        diversity_times: 采样次数
        seed: 随机种子
    
    Returns:
        diversity: 多样性分数
    """
    assert len(features.shape) == 2, f"Expected 2D array, got {features.shape}"
    num_samples = features.shape[0]
    
    if num_samples < 2:
        print("Warning: Not enough samples for diversity calculation")
        return 0.0
    
    # 确保采样次数不超过可能的配对数
    max_pairs = num_samples * (num_samples - 1) // 2
    diversity_times = min(diversity_times, max_pairs)
    
    if seed is not None:
        np.random.seed(seed)
    
    first_indices = np.random.choice(num_samples, diversity_times, replace=True)
    second_indices = np.random.choice(num_samples, diversity_times, replace=True)
    
    # 计算配对距离
    distances = np.linalg.norm(
        features[first_indices] - features[second_indices], 
        axis=1
    )
    
    return float(distances.mean())


def calculate_multimodality(
    features: np.ndarray, 
    num_times: int = 10,
    seed: Optional[int] = None
) -> float:
    """
    计算多模态性
    
    对于同一文本 prompt 生成的多个动作，计算它们之间的平均距离。
    越高越好，表示模型能为相同输入生成不同的动作。
    
    Args:
        features: (num_prompts, num_repeats, D) 
                  每个 prompt 生成 num_repeats 个动作
        num_times: 每个 prompt 的采样次数
        seed: 随机种子
    
    Returns:
        multimodality: 多模态性分数
    """
    assert len(features.shape) == 3, f"Expected 3D array (prompts, repeats, features), got {features.shape}"
    
    num_prompts, num_repeats, _ = features.shape
    
    if num_repeats < 2:
        print("Warning: Need at least 2 repeats per prompt for multimodality")
        return 0.0
    
    if seed is not None:
        np.random.seed(seed)
    
    num_times = min(num_times, num_repeats * (num_repeats - 1) // 2)
    
    first_indices = np.random.choice(num_repeats, num_times, replace=True)
    second_indices = np.random.choice(num_repeats, num_times, replace=True)
    
    # 计算每个 prompt 内不同生成结果的距离
    distances = np.linalg.norm(
        features[:, first_indices] - features[:, second_indices],
        axis=2
    )
    
    return float(distances.mean())


def calculate_top_k(argsort_mat: np.ndarray, top_k: int = 3) -> np.ndarray:
    """
    计算 Top-K 匹配矩阵
    
    Args:
        argsort_mat: (N, N) 排序后的索引矩阵
        top_k: 计算前 K 个
    
    Returns:
        top_k_mat: (N, top_k) 布尔矩阵，表示每个样本是否在 top-k 中匹配
    """
    size = argsort_mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (argsort_mat == gt_mat)
    
    correct_vec = np.zeros(size, dtype=bool)
    top_k_list = []
    
    for i in range(top_k):
        correct_vec = correct_vec | bool_mat[:, i]
        top_k_list.append(correct_vec.copy())
    
    top_k_mat = np.stack(top_k_list, axis=1)
    return top_k_mat


def calculate_r_precision(
    text_features: np.ndarray, 
    motion_features: np.ndarray, 
    top_k: int = 3
) -> np.ndarray:
    """
    计算 R-precision (检索精度)
    
    给定文本特征和动作特征，计算文本检索到正确动作的 Top-K 精度。
    
    Args:
        text_features: (N, D) 文本特征
        motion_features: (N, D) 动作特征 (与文本一一对应)
        top_k: 计算 Top-1/2/3 精度
    
    Returns:
        r_precision: (top_k,) Top-1/2/3 精度
    """
    assert text_features.shape[0] == motion_features.shape[0], "Sample count must match"
    
    dist_mat = euclidean_distance_matrix(text_features, motion_features)
    argsort_mat = np.argsort(dist_mat, axis=1)
    
    top_k_mat = calculate_top_k(argsort_mat, top_k)
    r_precision = top_k_mat.sum(axis=0) / text_features.shape[0]
    
    return r_precision


def calculate_matching_score(
    text_features: np.ndarray, 
    motion_features: np.ndarray
) -> float:
    """
    计算匹配分数 (Matching Score)
    
    计算配对的文本和动作特征之间的平均欧氏距离。
    越低越好，表示文本和动作的匹配程度越高。
    
    Args:
        text_features: (N, D) 文本特征
        motion_features: (N, D) 动作特征 (与文本一一对应)
    
    Returns:
        matching_score: 匹配分数
    """
    assert text_features.shape == motion_features.shape, "Feature shapes must match"
    
    dist_mat = euclidean_distance_matrix(text_features, motion_features)
    matching_score = np.trace(dist_mat) / text_features.shape[0]
    
    return float(matching_score)


def calculate_motion_statistics(motion_data: np.ndarray) -> dict:
    """
    计算动作数据的基础统计信息
    
    Args:
        motion_data: (T, J, 3) 或 (N, T, J, 3) 动作数据
    
    Returns:
        stats: 统计信息字典
    """
    if len(motion_data.shape) == 3:
        motion_data = motion_data[np.newaxis, ...]
    
    # 计算速度 (帧间差分)
    velocity = np.diff(motion_data, axis=1)
    
    # 计算加速度 (速度差分)
    acceleration = np.diff(velocity, axis=1)
    
    stats = {
        'mean_position': float(motion_data.mean()),
        'std_position': float(motion_data.std()),
        'mean_velocity': float(np.abs(velocity).mean()),
        'std_velocity': float(velocity.std()),
        'mean_acceleration': float(np.abs(acceleration).mean()),
        'std_acceleration': float(acceleration.std()),
        'position_range': float(motion_data.max() - motion_data.min()),
    }
    
    return stats

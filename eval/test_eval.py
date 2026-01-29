"""
测试评估模块
用于验证评估功能是否正常工作

用法:
    python -m eval.test_eval
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import (
    calculate_fid,
    calculate_diversity,
    calculate_multimodality,
    calculate_matching_score,
    calculate_r_precision,
    calculate_motion_statistics,
    euclidean_distance_matrix,
)
from eval.feature_extractor import MotionFeatureExtractor, SimpleTextFeatureExtractor
from eval.evaluator import MotionEvaluator


def generate_random_motion(num_frames: int = 40, num_joints: int = 21) -> np.ndarray:
    """生成随机动作数据用于测试"""
    # 模拟真实动作的统计特性
    motion = np.random.randn(num_frames, num_joints, 3) * 500  # 标准差约 500mm
    
    # 添加一些时序相关性
    for i in range(1, num_frames):
        motion[i] = 0.8 * motion[i-1] + 0.2 * motion[i]
    
    return motion


def test_metrics():
    """测试基础 metrics 函数"""
    print("=" * 60)
    print("Testing Basic Metrics")
    print("=" * 60)
    
    # 生成测试数据
    np.random.seed(42)
    
    # 测试欧氏距离矩阵
    print("\n[1] Testing euclidean_distance_matrix...")
    mat1 = np.random.randn(10, 64)
    mat2 = np.random.randn(10, 64)
    dist_mat = euclidean_distance_matrix(mat1, mat2)
    assert dist_mat.shape == (10, 10), f"Expected (10, 10), got {dist_mat.shape}"
    print(f"  Distance matrix shape: {dist_mat.shape}")
    print(f"  Distance range: [{dist_mat.min():.2f}, {dist_mat.max():.2f}]")
    print("  PASSED ✓")
    
    # 测试 FID
    print("\n[2] Testing calculate_fid...")
    gt_features = np.random.randn(100, 64)
    gen_features = np.random.randn(100, 64) + 0.5  # 稍微偏移
    fid = calculate_fid(gt_features, gen_features)
    assert fid >= 0, f"FID should be non-negative, got {fid}"
    print(f"  FID: {fid:.4f}")
    print("  PASSED ✓")
    
    # 测试 Diversity
    print("\n[3] Testing calculate_diversity...")
    features = np.random.randn(100, 64)
    diversity = calculate_diversity(features, diversity_times=100)
    assert diversity >= 0, f"Diversity should be non-negative, got {diversity}"
    print(f"  Diversity: {diversity:.4f}")
    print("  PASSED ✓")
    
    # 测试 MultiModality
    print("\n[4] Testing calculate_multimodality...")
    features_3d = np.random.randn(20, 5, 64)  # 20 prompts, 5 repeats each
    mm = calculate_multimodality(features_3d, num_times=5)
    assert mm >= 0, f"MultiModality should be non-negative, got {mm}"
    print(f"  MultiModality: {mm:.4f}")
    print("  PASSED ✓")
    
    # 测试 Matching Score
    print("\n[5] Testing calculate_matching_score...")
    text_feat = np.random.randn(50, 64)
    motion_feat = np.random.randn(50, 64)
    ms = calculate_matching_score(text_feat, motion_feat)
    assert ms >= 0, f"Matching Score should be non-negative, got {ms}"
    print(f"  Matching Score: {ms:.4f}")
    print("  PASSED ✓")
    
    # 测试 R-precision
    print("\n[6] Testing calculate_r_precision...")
    r_prec = calculate_r_precision(text_feat, motion_feat, top_k=3)
    assert len(r_prec) == 3, f"Expected 3 values, got {len(r_prec)}"
    assert all(0 <= p <= 1 for p in r_prec), "R-precision values should be in [0, 1]"
    print(f"  R-precision: Top-1={r_prec[0]:.4f}, Top-2={r_prec[1]:.4f}, Top-3={r_prec[2]:.4f}")
    print("  PASSED ✓")
    
    # 测试动作统计
    print("\n[7] Testing calculate_motion_statistics...")
    motion = generate_random_motion()
    stats = calculate_motion_statistics(motion)
    print(f"  Stats keys: {list(stats.keys())}")
    print(f"  Mean velocity: {stats['mean_velocity']:.4f}")
    print("  PASSED ✓")


def test_feature_extractor():
    """测试特征提取器"""
    print("\n" + "=" * 60)
    print("Testing Feature Extractors")
    print("=" * 60)
    
    # 测试 Motion Feature Extractor
    print("\n[1] Testing MotionFeatureExtractor...")
    motion_extractor = MotionFeatureExtractor(num_joints=21, feature_dim=256)
    
    motion = generate_random_motion(num_frames=40, num_joints=21)
    features = motion_extractor.extract_features(motion)
    
    assert features.shape == (256,), f"Expected (256,), got {features.shape}"
    print(f"  Input motion shape: {motion.shape}")
    print(f"  Output feature shape: {features.shape}")
    print("  PASSED ✓")
    
    # 批量提取
    print("\n[2] Testing batch extraction...")
    motions = [generate_random_motion(np.random.randint(20, 60)) for _ in range(10)]
    batch_features = motion_extractor.extract_batch_features(motions)
    
    assert batch_features.shape == (10, 256), f"Expected (10, 256), got {batch_features.shape}"
    print(f"  Batch input: {len(motions)} motions")
    print(f"  Batch output shape: {batch_features.shape}")
    print("  PASSED ✓")
    
    # 测试 Text Feature Extractor
    print("\n[3] Testing SimpleTextFeatureExtractor...")
    text_extractor = SimpleTextFeatureExtractor(feature_dim=256)
    
    text = "A person walks forward slowly"
    features = text_extractor.extract_features(text)
    
    assert features.shape == (256,), f"Expected (256,), got {features.shape}"
    print(f"  Input text: '{text}'")
    print(f"  Output feature shape: {features.shape}")
    print("  PASSED ✓")
    
    # 批量提取
    print("\n[4] Testing batch text extraction...")
    texts = [
        "A person walks forward",
        "Someone jumps up and down",
        "A man waves his hand",
        "A woman sits down on a chair",
    ]
    batch_features = text_extractor.extract_batch_features(texts)
    
    assert batch_features.shape == (4, 256), f"Expected (4, 256), got {batch_features.shape}"
    print(f"  Batch input: {len(texts)} texts")
    print(f"  Batch output shape: {batch_features.shape}")
    print("  PASSED ✓")


def test_evaluator():
    """测试完整的评估器"""
    print("\n" + "=" * 60)
    print("Testing MotionEvaluator")
    print("=" * 60)
    
    # 初始化评估器
    evaluator = MotionEvaluator(
        num_joints=21,
        feature_dim=256,
        diversity_times=100,
        seed=42,
    )
    
    # 生成测试数据
    print("\n[1] Generating test data...")
    np.random.seed(42)
    
    # GT 动作
    gt_motions = [generate_random_motion(np.random.randint(30, 50)) for _ in range(30)]
    
    # 生成动作 (添加一些噪声来模拟生成质量差异)
    gen_motions = []
    for gt in gt_motions:
        # 模拟生成: GT + 噪声
        noise_level = np.random.uniform(0.1, 0.5)
        gen = gt + np.random.randn(*gt.shape) * noise_level * gt.std()
        gen_motions.append(gen)
    
    print(f"  GT motions: {len(gt_motions)}")
    print(f"  Generated motions: {len(gen_motions)}")
    
    # 文本
    texts = [
        f"A person performs action {i}" for i in range(len(gen_motions))
    ]
    
    # 测试各个评估函数
    print("\n[2] Testing FID...")
    fid = evaluator.evaluate_fid(gt_motions, gen_motions)
    print(f"  FID: {fid:.4f}")
    
    print("\n[3] Testing Diversity...")
    gt_div = evaluator.evaluate_diversity(gt_motions)
    gen_div = evaluator.evaluate_diversity(gen_motions)
    print(f"  GT Diversity: {gt_div:.4f}")
    print(f"  Gen Diversity: {gen_div:.4f}")
    
    print("\n[4] Testing Matching Score...")
    ms = evaluator.evaluate_matching_score(texts, gen_motions)
    print(f"  Matching Score: {ms:.4f}")
    
    print("\n[5] Testing R-precision...")
    r_prec = evaluator.evaluate_r_precision(texts, gen_motions)
    print(f"  R-precision: Top-1={r_prec[0]:.4f}, Top-2={r_prec[1]:.4f}, Top-3={r_prec[2]:.4f}")
    
    # 完整评估
    print("\n[6] Testing full evaluation...")
    results = evaluator.evaluate_all(
        gt_motions=gt_motions,
        gen_motions=gen_motions,
        texts=texts,
        compute_mm=False,
    )
    
    print("\n  Results summary:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"    {key}: <dict with {len(value)} keys>")
    
    print("\n  PASSED ✓")


def main():
    print("=" * 60)
    print("Motion Evaluation Module Tests")
    print("=" * 60)
    
    try:
        test_metrics()
        test_feature_extractor()
        test_evaluator()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

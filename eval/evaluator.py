"""
Motion Evaluator

综合评估器，统一管理所有评估指标的计算。
"""

import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
from datetime import datetime

from .metrics import (
    calculate_fid,
    calculate_diversity,
    calculate_multimodality,
    calculate_matching_score,
    calculate_r_precision,
    calculate_motion_statistics,
)
from .feature_extractor import MotionFeatureExtractor, SimpleTextFeatureExtractor


class MotionEvaluator:
    """
    动作生成模型评估器
    
    支持以下评估指标:
    - FID: 生成质量
    - Diversity: 生成多样性
    - MultiModality: 多模态性 (同一文本生成不同动作)
    - Matching Score: 文本-动作匹配度
    - R-precision: 文本-动作检索精度
    """
    
    def __init__(
        self,
        num_joints: int = 21,
        feature_dim: int = 256,
        diversity_times: int = 300,
        mm_num_times: int = 10,
        top_k: int = 3,
        seed: Optional[int] = None,
    ):
        """
        Args:
            num_joints: 关节数量 (KIT-ML 为 21)
            feature_dim: 特征维度
            diversity_times: 多样性计算的采样次数
            mm_num_times: 多模态性计算的采样次数
            top_k: R-precision 的 K 值
            seed: 随机种子
        """
        self.num_joints = num_joints
        self.feature_dim = feature_dim
        self.diversity_times = diversity_times
        self.mm_num_times = mm_num_times
        self.top_k = top_k
        self.seed = seed
        
        # 初始化特征提取器
        self.motion_extractor = MotionFeatureExtractor(
            num_joints=num_joints,
            feature_dim=feature_dim,
        )
        self.text_extractor = SimpleTextFeatureExtractor(feature_dim=feature_dim)
    
    def evaluate_fid(
        self, 
        gt_motions: List[np.ndarray], 
        gen_motions: List[np.ndarray]
    ) -> float:
        """
        计算 FID
        
        Args:
            gt_motions: 真实动作列表
            gen_motions: 生成动作列表
        
        Returns:
            fid: FID 分数
        """
        print("Extracting GT features...")
        gt_features = self.motion_extractor.extract_batch_features(gt_motions)
        
        print("Extracting generated features...")
        gen_features = self.motion_extractor.extract_batch_features(gen_motions)
        
        print("Calculating FID...")
        fid = calculate_fid(gt_features, gen_features)
        return fid
    
    def evaluate_diversity(self, motions: List[np.ndarray]) -> float:
        """
        计算多样性
        
        Args:
            motions: 动作列表
        
        Returns:
            diversity: 多样性分数
        """
        features = self.motion_extractor.extract_batch_features(motions)
        diversity = calculate_diversity(
            features, 
            diversity_times=self.diversity_times,
            seed=self.seed
        )
        return diversity
    
    def evaluate_multimodality(
        self, 
        motions_per_text: List[List[np.ndarray]]
    ) -> float:
        """
        计算多模态性
        
        Args:
            motions_per_text: 每个文本对应的多个生成动作
                              [[motion1, motion2, ...], [motion1, motion2, ...], ...]
        
        Returns:
            multimodality: 多模态性分数
        """
        # 提取每个动作的特征
        all_features = []
        for text_motions in motions_per_text:
            text_features = self.motion_extractor.extract_batch_features(text_motions)
            all_features.append(text_features)
        
        # 对齐维度 (取最小重复次数)
        min_repeats = min(len(f) for f in all_features)
        features_array = np.stack([f[:min_repeats] for f in all_features], axis=0)
        
        multimodality = calculate_multimodality(
            features_array,
            num_times=self.mm_num_times,
            seed=self.seed
        )
        return multimodality
    
    def evaluate_matching_score(
        self,
        texts: List[str],
        motions: List[np.ndarray]
    ) -> float:
        """
        计算匹配分数
        
        Args:
            texts: 文本列表
            motions: 对应的动作列表
        
        Returns:
            matching_score: 匹配分数
        """
        text_features = self.text_extractor.extract_batch_features(texts)
        motion_features = self.motion_extractor.extract_batch_features(motions)
        
        matching_score = calculate_matching_score(text_features, motion_features)
        return matching_score
    
    def evaluate_r_precision(
        self,
        texts: List[str],
        motions: List[np.ndarray]
    ) -> np.ndarray:
        """
        计算 R-precision
        
        Args:
            texts: 文本列表
            motions: 对应的动作列表
        
        Returns:
            r_precision: (top_k,) Top-1/2/3 精度
        """
        text_features = self.text_extractor.extract_batch_features(texts)
        motion_features = self.motion_extractor.extract_batch_features(motions)
        
        r_precision = calculate_r_precision(
            text_features, 
            motion_features, 
            top_k=self.top_k
        )
        return r_precision
    
    def evaluate_all(
        self,
        gt_motions: List[np.ndarray],
        gen_motions: List[np.ndarray],
        texts: Optional[List[str]] = None,
        motions_per_text: Optional[List[List[np.ndarray]]] = None,
        compute_mm: bool = True,
    ) -> Dict[str, Any]:
        """
        计算所有评估指标
        
        Args:
            gt_motions: 真实动作列表
            gen_motions: 生成动作列表
            texts: 文本列表 (用于 matching score 和 R-precision)
            motions_per_text: 每个文本的多个生成结果 (用于 multimodality)
            compute_mm: 是否计算 multimodality
        
        Returns:
            results: 包含所有指标的字典
        """
        results = OrderedDict()
        
        print("=" * 60)
        print("Motion Generation Evaluation")
        print("=" * 60)
        print(f"GT samples: {len(gt_motions)}")
        print(f"Generated samples: {len(gen_motions)}")
        print(f"Time: {datetime.now()}")
        print("=" * 60)
        
        # 1. FID
        print("\n[1/5] Calculating FID...")
        fid = self.evaluate_fid(gt_motions, gen_motions)
        results['FID'] = fid
        print(f"FID: {fid:.4f}")
        
        # 2. GT Diversity
        print("\n[2/5] Calculating GT Diversity...")
        gt_diversity = self.evaluate_diversity(gt_motions)
        results['GT_Diversity'] = gt_diversity
        print(f"GT Diversity: {gt_diversity:.4f}")
        
        # 3. Generated Diversity
        print("\n[3/5] Calculating Generated Diversity...")
        gen_diversity = self.evaluate_diversity(gen_motions)
        results['Gen_Diversity'] = gen_diversity
        print(f"Generated Diversity: {gen_diversity:.4f}")
        
        # 4. Matching Score & R-precision (需要文本)
        if texts is not None and len(texts) == len(gen_motions):
            print("\n[4/5] Calculating Matching Score & R-precision...")
            matching_score = self.evaluate_matching_score(texts, gen_motions)
            r_precision = self.evaluate_r_precision(texts, gen_motions)
            
            results['Matching_Score'] = matching_score
            results['R_precision_top1'] = float(r_precision[0])
            results['R_precision_top2'] = float(r_precision[1])
            results['R_precision_top3'] = float(r_precision[2])
            
            print(f"Matching Score: {matching_score:.4f}")
            print(f"R-precision (Top-1): {r_precision[0]:.4f}")
            print(f"R-precision (Top-2): {r_precision[1]:.4f}")
            print(f"R-precision (Top-3): {r_precision[2]:.4f}")
        else:
            print("\n[4/5] Skipping Matching Score & R-precision (no text provided)")
            results['Matching_Score'] = None
            results['R_precision_top1'] = None
            results['R_precision_top2'] = None
            results['R_precision_top3'] = None
        
        # 5. MultiModality (需要多次生成)
        if compute_mm and motions_per_text is not None and len(motions_per_text) > 0:
            print("\n[5/5] Calculating MultiModality...")
            multimodality = self.evaluate_multimodality(motions_per_text)
            results['MultiModality'] = multimodality
            print(f"MultiModality: {multimodality:.4f}")
        else:
            print("\n[5/5] Skipping MultiModality (no repeated generations)")
            results['MultiModality'] = None
        
        # 附加: 动作统计信息
        print("\n--- Motion Statistics ---")
        
        gt_stats = self._aggregate_motion_stats(gt_motions)
        gen_stats = self._aggregate_motion_stats(gen_motions)
        
        results['GT_Stats'] = gt_stats
        results['Gen_Stats'] = gen_stats
        
        print(f"GT  - Mean Velocity: {gt_stats['mean_velocity']:.4f}, Std Position: {gt_stats['std_position']:.4f}")
        print(f"Gen - Mean Velocity: {gen_stats['mean_velocity']:.4f}, Std Position: {gen_stats['std_position']:.4f}")
        
        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)
        
        return results
    
    def _aggregate_motion_stats(self, motions: List[np.ndarray]) -> Dict[str, float]:
        """聚合多个动作的统计信息"""
        all_stats = []
        for motion in motions:
            stats = calculate_motion_statistics(motion)
            all_stats.append(stats)
        
        # 取平均
        agg_stats = {}
        for key in all_stats[0].keys():
            values = [s[key] for s in all_stats]
            agg_stats[key] = float(np.mean(values))
        
        return agg_stats
    
    def save_results(self, results: Dict[str, Any], save_path: str):
        """
        保存评估结果
        
        Args:
            results: 评估结果字典
            save_path: 保存路径 (JSON)
        """
        # 转换为可序列化格式
        serializable = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                serializable[key] = float(value)
            elif isinstance(value, dict):
                serializable[key] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in value.items()
                }
            else:
                serializable[key] = value
        
        serializable['timestamp'] = datetime.now().isoformat()
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {save_path}")


def run_evaluation_with_replications(
    evaluator: MotionEvaluator,
    gt_motions: List[np.ndarray],
    gen_motions: List[np.ndarray],
    texts: Optional[List[str]] = None,
    replication_times: int = 5,
) -> Dict[str, Tuple[float, float]]:
    """
    运行多次评估并计算均值和置信区间
    
    Args:
        evaluator: 评估器实例
        gt_motions: 真实动作
        gen_motions: 生成动作
        texts: 文本描述
        replication_times: 重复次数
    
    Returns:
        results: {metric_name: (mean, confidence_interval)}
    """
    all_results = []
    
    for rep in range(replication_times):
        print(f"\n{'='*60}")
        print(f"Replication {rep + 1}/{replication_times}")
        print(f"{'='*60}")
        
        results = evaluator.evaluate_all(
            gt_motions=gt_motions,
            gen_motions=gen_motions,
            texts=texts,
            compute_mm=False,  # 多模态性通常只计算一次
        )
        all_results.append(results)
    
    # 聚合结果
    final_results = {}
    numeric_keys = ['FID', 'GT_Diversity', 'Gen_Diversity', 'Matching_Score',
                    'R_precision_top1', 'R_precision_top2', 'R_precision_top3']
    
    for key in numeric_keys:
        values = [r[key] for r in all_results if r[key] is not None]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            conf_interval = 1.96 * std / np.sqrt(len(values))
            final_results[key] = (float(mean), float(conf_interval))
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Mean ± 95% Confidence Interval)")
    print("=" * 60)
    for key, (mean, ci) in final_results.items():
        print(f"{key}: {mean:.4f} ± {ci:.4f}")
    
    return final_results

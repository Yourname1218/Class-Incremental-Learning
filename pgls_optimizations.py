# -*- coding: utf-8 -*-
"""
Progressive Learning Strategy for Few-Shot Class-Incremental Learning (PGLS) 實現
本模組實現了PGLS論文中的兩個核心方法：
1. 魯棒課程學習 (Robust Curriculum Learning, RCL)
2. 漸進式虛擬類別引入 (Progressive Virtual Class Introduction, IVC)

注意：此實現僅用於基礎模型訓練，與GAN訓練無關
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class RobustCurriculumLearner:
    """
    魯棒課程學習器（增強版）
    
    基於協方差噪聲擾動的樣本魯棒性評估，實現課程學習策略
    新增功能：多層次魯棒性評估（已啟用）、記憶體友好處理、自適應噪聲調整、對比學習增強
    
    多層次魯棒性評估：
    - 將特徵分為淺層（前40%）和深層（後60%）
    - 淺層特徵：捕捉邊緣、紋理等低層次視覺信息（權重0.3）
    - 深層特徵：捕捉語義、抽象等高層次概念信息（權重0.7）
    - 使用layer_weights融合不同層次的魯棒性評估結果
    """
    
    def __init__(self, num_classes: int = 1000, noise_scale: float = 0.12, robust_threshold: float = 0.65):
        """
        初始化魯棒課程學習器
        
        Args:
            num_classes: 總類別數
            noise_scale: 噪聲縮放係數 λ（提高以增強魯棒性）
            robust_threshold: 魯棒性閾值（降低以包含更多樣本）
        """
        self.num_classes = num_classes                    # 總類別數，用於計算類別統計信息
        self.noise_scale = noise_scale                    # 噪聲縮放係數，控制擾動強度
        self.robust_threshold = robust_threshold          # 魯棒性閾值，區分魯棒和弱魯棒樣本
        self.class_statistics = {}                        # 存儲每個類別的統計信息（均值、協方差）
        self.robust_weights = {"W1": 2.0, "W2": 1.0}    # 魯棒樣本權重W1=2，弱魯棒樣本權重W2=1
        
        # 優化：多層次魯棒性評估配置
        self.layer_weights = [0.4, 0.6]                  # 淺層/深層特徵權重 [淺層, 深層] - 更平衡
        self.adaptive_noise_enabled = True                # 啟用自適應噪聲調整
        self.difficulty_cache = {}                        # 樣本難度快取（避免重複計算）
        self.batch_processing_size = 64                   # 批次處理大小（記憶體友好）
        
        # 優化：對比學習增強配置（大幅提升權重）
        self.contrastive_enabled = True                   # 啟用對比學習增強
        self.contrastive_temperature = 0.05               # 對比學習溫度參數（更嚴格區分）
        self.negative_samples_ratio = 2.0                 # 負樣本比例
        self.contrastive_alpha = 0.25                     # 對比損失權重（大幅提升）
        self.similarity_threshold = 0.8                   # 相似度閾值
        
    def compute_class_statistics(self, features: torch.Tensor, labels: torch.Tensor) -> Dict:
        """
        計算每個類別的統計信息（均值和協方差）
        
        Args:
            features: 特徵張量 [batch_size, feature_dim]
            labels: 標籤張量 [batch_size]
            
        Returns:
            字典包含每個類別的均值和協方差矩陣
        """
        unique_labels = torch.unique(labels)              # 獲取批次中所有唯一的類別標籤
        statistics = {}                                    # 初始化統計信息字典
        
        for label in unique_labels:                        # 遍歷每個類別
            mask = labels == label                         # 創建該類別的樣本掩碼
            class_features = features[mask]                # 提取該類別的所有特徵
            
            if class_features.size(0) > 1:                # 確保有足夠樣本計算協方差
                mu = torch.mean(class_features, dim=0)    # 計算類別均值
                # 計算協方差矩陣，添加對角線正則化避免奇異性
                centered = class_features - mu            # 中心化特徵
                cov = torch.mm(centered.t(), centered) / (class_features.size(0) - 1)  # 計算協方差
                cov += torch.eye(cov.size(0), device=cov.device) * 1e-6  # 添加正則化項避免奇異性
                
                statistics[label.item()] = {"mean": mu, "cov": cov}  # 存儲該類別的統計信息
            else:
                # 如果只有一個樣本，使用該樣本作為均值，協方差為單位矩陣
                mu = class_features[0]
                cov = torch.eye(features.size(1), device=features.device) * 0.1
                statistics[label.item()] = {"mean": mu, "cov": cov}
                
        return statistics
    
    def compute_adaptive_noise_scale(self, epoch: int, total_epochs: int, sample_difficulty: float = 0.5) -> float:
        """
        自適應噪聲強度調整 - 根據訓練進度和樣本難度動態調整噪聲
        
        Args:
            epoch: 當前epoch
            total_epochs: 總epoch數  
            sample_difficulty: 樣本難度 [0-1]，0=簡單，1=困難
            
        Returns:
            調整後的噪聲縮放係數
        """
        if not self.adaptive_noise_enabled:
            return self.noise_scale
        
        # 訓練進度影響：早期大噪聲，後期小噪聲
        progress_factor = 1.0 - (epoch / total_epochs) * 0.5
        
        # 樣本難度影響：困難樣本小噪聲，簡單樣本大噪聲
        difficulty_factor = 1.0 - sample_difficulty * 0.3
        
        # 組合調整係數
        adaptive_scale = self.noise_scale * progress_factor * difficulty_factor
        
        # 確保在合理範圍內 [0.05, 0.2]
        return max(0.05, min(0.2, adaptive_scale))
    
    def evaluate_sample_difficulty(self, model: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        評估樣本學習難度（輕量級實現）
        
        Args:
            model: 當前模型
            features: 特徵張量
            labels: 標籤張量
            
        Returns:
            樣本難度分數 [batch_size]，範圍 [0-1]
        """
        model.eval()
        difficulties = []
        
        with torch.no_grad():
            # 分批處理，避免記憶體問題
            for i in range(0, features.size(0), self.batch_processing_size):
                batch_features = features[i:i+self.batch_processing_size]
                batch_labels = labels[i:i+self.batch_processing_size]
                
                # 計算預測置信度（低置信度 = 高難度）
                logits = model.embed(batch_features)
                probs = F.softmax(logits, dim=1)
                
                # 獲取正確類別的概率
                correct_probs = probs.gather(1, batch_labels.unsqueeze(1)).squeeze()
                
                # 難度 = 1 - 正確預測的置信度
                batch_difficulty = 1.0 - correct_probs
                difficulties.append(batch_difficulty)
        
        model.train()
        return torch.cat(difficulties) if difficulties else torch.zeros(features.size(0), device=features.device)
    
    def multi_layer_robustness_evaluation(self, model: nn.Module, features: torch.Tensor, 
                                        labels: torch.Tensor, statistics: Dict, 
                                        epoch: int, total_epochs: int) -> Dict:
        """
        多層次魯棒性評估 - 在不同特徵層次進行擾動和評估
        
        基於layer_weights在淺層和深層特徵上分別進行魯棒性測試，
        提供更全面的樣本難度和魯棒性評估
        
        Args:
            model: 當前訓練的模型
            features: 輸入特徵 [batch_size, feature_dim]
            labels: 對應標籤 [batch_size]
            statistics: 類別統計信息
            epoch: 當前epoch
            total_epochs: 總epoch數
            
        Returns:
            包含多層次魯棒性信息的字典
        """
        # 模擬淺層和深層特徵（通過特徵分割）
        feature_dim = features.size(1)
        shallow_dim = int(feature_dim * 0.4)  # 前40%作為淺層特徵
        deep_dim = feature_dim - shallow_dim   # 後60%作為深層特徵
        
        shallow_features = features[:, :shallow_dim]    # 淺層特徵（邊緣、紋理等）
        deep_features = features[:, shallow_dim:]       # 深層特徵（語義、抽象等）
        
        # 對淺層特徵進行擾動評估
        shallow_perturbations = self._generate_layer_perturbation(
            shallow_features, labels, statistics, epoch, total_epochs, layer_type="shallow"
        )
        shallow_robustness = self._evaluate_layer_robustness(
            model, features, shallow_perturbations, labels, "shallow"
        )
        
        # 對深層特徵進行擾動評估  
        deep_perturbations = self._generate_layer_perturbation(
            deep_features, labels, statistics, epoch, total_epochs, layer_type="deep"
        )
        deep_robustness = self._evaluate_layer_robustness(
            model, features, deep_perturbations, labels, "deep"
        )
        
        # 使用layer_weights融合多層次評估結果
        shallow_weight, deep_weight = self.layer_weights
        
        # 融合魯棒性評估結果
        combined_robust_indices = []
        combined_weak_indices = []
        layer_robustness_scores = torch.zeros(features.size(0), device=features.device)
        
        for i in range(features.size(0)):
            # 計算加權魯棒性分數
            shallow_robust = 1.0 if i in shallow_robustness["robust_indices"] else 0.0
            deep_robust = 1.0 if i in deep_robustness["robust_indices"] else 0.0
            
            weighted_robustness = shallow_weight * shallow_robust + deep_weight * deep_robust
            layer_robustness_scores[i] = weighted_robustness
            
            # 基於加權分數分類樣本
            if weighted_robustness > 0.6:  # 高閾值表示魯棒
                combined_robust_indices.append(i)
            elif weighted_robustness > 0.2:  # 中等閾值表示弱魯棒
                combined_weak_indices.append(i)
            # 低於0.2的樣本被忽略（過於困難）
        
        return {
            "robust_indices": combined_robust_indices,
            "weak_robust_indices": combined_weak_indices,
            "layer_robustness_scores": layer_robustness_scores,
            "shallow_robustness": shallow_robustness,
            "deep_robustness": deep_robustness,
            "layer_weights_used": self.layer_weights,
            "multi_layer_enabled": True
        }
    
    def _generate_layer_perturbation(self, layer_features: torch.Tensor, labels: torch.Tensor,
                                   statistics: Dict, epoch: int, total_epochs: int, 
                                   layer_type: str = "shallow") -> torch.Tensor:
        """
        為特定層次的特徵生成擾動
        
        Args:
            layer_features: 特定層次的特徵
            labels: 標籤
            statistics: 統計信息
            epoch: 當前epoch
            total_epochs: 總epoch數
            layer_type: 層次類型 ("shallow" or "deep")
            
        Returns:
            擾動後的特徵
        """
        perturbed_features = []
        
        # 根據層次類型調整擾動強度
        if layer_type == "shallow":
            # 淺層特徵：較大的擾動（模擬視覺噪聲、光照變化等）
            layer_noise_scale = self.noise_scale * 1.2
        else:
            # 深層特徵：較小的擾動（保持語義一致性）
            layer_noise_scale = self.noise_scale * 0.8
        
        for feature, label in zip(layer_features, labels):
            label_item = label.item()
            sample_difficulty = 0.5  # 簡化版
            adaptive_scale = self.compute_adaptive_noise_scale(epoch, total_epochs, sample_difficulty)
            final_scale = layer_noise_scale * adaptive_scale
            
            if label_item in statistics:
                # 使用類別統計信息生成擾動
                stat = statistics[label_item]
                mu = stat["mean"][:layer_features.size(1)] if stat["mean"].size(0) > layer_features.size(1) else stat["mean"]
                
                # 簡化的協方差擾動（適應層次特徵維度）
                try:
                    noise = torch.randn_like(feature) * final_scale
                    perturbed_feature = feature + noise
                except:
                    perturbed_feature = feature + torch.randn_like(feature) * final_scale
            else:
                # 使用標準高斯噪聲
                perturbed_feature = feature + torch.randn_like(feature) * final_scale
            
            perturbed_features.append(perturbed_feature)
        
        return torch.stack(perturbed_features)
    
    def _evaluate_layer_robustness(self, model: nn.Module, original_full_features: torch.Tensor,
                                 layer_perturbations: torch.Tensor, labels: torch.Tensor,
                                 layer_type: str) -> Dict:
        """
        評估特定層次擾動對整體預測的影響
        
        Args:
            model: 模型
            original_full_features: 完整的原始特徵
            layer_perturbations: 特定層次的擾動特徵
            labels: 標籤
            layer_type: 層次類型
            
        Returns:
            該層次的魯棒性評估結果
        """
        model.eval()
        robust_indices = []
        weak_robust_indices = []
        
        feature_dim = original_full_features.size(1)
        shallow_dim = int(feature_dim * 0.4)
        
        with torch.no_grad():
            for i, (perturbed_layer, original_full, true_label) in enumerate(
                zip(layer_perturbations, original_full_features, labels)):
                
                # 重構完整特徵：將擾動層次與未擾動層次組合
                if layer_type == "shallow":
                    # 淺層擾動：替換前40%特徵
                    reconstructed_feature = torch.cat([perturbed_layer, original_full[shallow_dim:]])
                else:
                    # 深層擾動：替換後60%特徵  
                    reconstructed_feature = torch.cat([original_full[:shallow_dim], perturbed_layer])
                
                # 預測擾動後的結果
                perturbed_logits = model.embed(reconstructed_feature.unsqueeze(0))
                perturbed_probs = F.softmax(perturbed_logits, dim=1)
                perturbed_pred = torch.argmax(perturbed_probs, dim=1).item()
                max_confidence = torch.max(perturbed_probs, dim=1)[0].item()
                
                # 評估魯棒性
                if perturbed_pred == true_label.item():
                    robust_indices.append(i)
                elif max_confidence < 0.5:
                    weak_robust_indices.append(i)
        
        model.train()
        
        return {
            "robust_indices": robust_indices,
            "weak_robust_indices": weak_robust_indices,
            "layer_type": layer_type,
            "total_samples": len(layer_perturbations)
        }
    
    def memory_efficient_perturbation(self, features: torch.Tensor, labels: torch.Tensor, 
                                    statistics: Dict, epoch: int, total_epochs: int) -> torch.Tensor:
        """
        記憶體友好的特徵擾動生成 - 分批處理避免記憶體溢出
        
        Args:
            features: 原始特徵 
            labels: 對應標籤
            statistics: 類別統計信息
            epoch: 當前epoch
            total_epochs: 總epoch數
            
        Returns:
            擾動後的特徵
        """
        perturbed_features = []
        
        # 分批處理，每次處理一定數量的樣本
        for i in range(0, features.size(0), self.batch_processing_size):
            batch_features = features[i:i+self.batch_processing_size]
            batch_labels = labels[i:i+self.batch_processing_size]
            
            batch_perturbed = []
            
            for j, (feature, label) in enumerate(zip(batch_features, batch_labels)):
                label_item = label.item()
                
                # 計算當前樣本的自適應噪聲強度
                sample_difficulty = 0.5  # 簡化版，實際可以從快取中獲取
                adaptive_noise_scale = self.compute_adaptive_noise_scale(epoch, total_epochs, sample_difficulty)
                
                if label_item in statistics:
                    stat = statistics[label_item]
                    mu, cov = stat["mean"], stat["cov"]
                    
                    try:
                        # 生成較少的噪聲樣本以節省記憶體
                        noise_dist = torch.distributions.MultivariateNormal(mu, cov)
                        noise_samples = [noise_dist.sample() - mu for _ in range(3)]  # 減少到3個樣本
                        
                        # 選擇干擾最大的噪聲
                        best_noise = min(noise_samples, 
                                       key=lambda n: F.cosine_similarity(feature.unsqueeze(0), 
                                                                        (feature + adaptive_noise_scale * n).unsqueeze(0)).item())
                        
                        perturbed_feature = feature + adaptive_noise_scale * best_noise
                    except:
                        # 簡化處理
                        noise = torch.randn_like(feature) * 0.1
                        perturbed_feature = feature + adaptive_noise_scale * noise
                else:
                    noise = torch.randn_like(feature) * 0.1
                    perturbed_feature = feature + adaptive_noise_scale * noise
                
                batch_perturbed.append(perturbed_feature)
            
            if batch_perturbed:
                perturbed_features.extend(batch_perturbed)
        
        return torch.stack(perturbed_features) if perturbed_features else features
    
    def generate_covariance_perturbation(self, features: torch.Tensor, labels: torch.Tensor, 
                                       statistics: Dict, n_samples: int = 5) -> torch.Tensor:
        """
        基於協方差統計信息生成噪聲擾動
        
        Args:
            features: 原始特徵 [batch_size, feature_dim]
            labels: 對應標籤 [batch_size]
            statistics: 類別統計信息
            n_samples: 每個樣本生成的擾動數量
            
        Returns:
            擾動後的特徵，選擇干擾最大的樣本
        """
        perturbed_features_list = []                       # 存儲所有擾動後的特徵
        
        for i, (feature, label) in enumerate(zip(features, labels)):  # 遍歷每個樣本
            label_item = label.item()                      # 獲取標籤值
            
            if label_item in statistics:                   # 如果該類別有統計信息
                stat = statistics[label_item]             # 獲取該類別的統計信息
                mu, cov = stat["mean"], stat["cov"]       # 提取均值和協方差
                
                # 生成多元高斯分佈樣本作為噪聲
                try:
                    noise_dist = torch.distributions.MultivariateNormal(mu, cov)  # 創建多元高斯分佈
                    noise_samples = []                     # 存儲噪聲樣本
                    
                    for _ in range(n_samples):             # 生成n_samples個噪聲樣本
                        noise = noise_dist.sample() - mu   # 生成噪聲（減去均值確保是純噪聲）
                        noise_samples.append(noise)
                    
                    # 選擇與原特徵餘弦相似度最小的噪聲（即干擾最大的）
                    best_noise = None
                    min_similarity = float('inf')         # 初始化最小相似度為正無窮
                    
                    for noise in noise_samples:           # 遍歷所有噪聲樣本
                        perturbed = feature + self.noise_scale * noise  # 生成擾動後的特徵
                        # 計算原特徵與擾動特徵的餘弦相似度
                        similarity = F.cosine_similarity(feature.unsqueeze(0), 
                                                        perturbed.unsqueeze(0)).item()
                        if similarity < min_similarity:    # 選擇相似度最小的（干擾最大的）
                            min_similarity = similarity
                            best_noise = noise
                    
                    # 使用最佳噪聲生成最終擾動特徵
                    perturbed_feature = feature + self.noise_scale * best_noise
                    
                except:
                    # 如果協方差矩陣有問題，使用簡單的高斯噪聲
                    noise = torch.randn_like(feature) * 0.1  # 標準高斯噪聲
                    perturbed_feature = feature + self.noise_scale * noise
            else:
                # 如果沒有統計信息，使用標準高斯噪聲
                noise = torch.randn_like(feature) * 0.1
                perturbed_feature = feature + self.noise_scale * noise
            
            perturbed_features_list.append(perturbed_feature)  # 添加到結果列表
        
        return torch.stack(perturbed_features_list)        # 將列表轉換為張量並返回
    
    def evaluate_sample_robustness(self, model: nn.Module, original_features: torch.Tensor, 
                                  perturbed_features: torch.Tensor, original_labels: torch.Tensor) -> Dict:
        """
        評估樣本的魯棒性
        
        Args:
            model: 訓練中的模型
            original_features: 原始特徵
            perturbed_features: 擾動後的特徵  
            original_labels: 原始標籤
            
        Returns:
            包含魯棒樣本和弱魯棒樣本索引的字典
        """
        model.eval()                                       # 設置模型為評估模式
        robust_indices = []                                # 魯棒樣本索引列表
        weak_robust_indices = []                           # 弱魯棒樣本索引列表
        
        with torch.no_grad():                              # 不計算梯度，節省內存和計算
            # 對擾動後的特徵進行前向傳播
            perturbed_logits = model.embed(perturbed_features)  # 獲取擾動特徵的logits
            perturbed_probs = F.softmax(perturbed_logits, dim=1)  # 轉換為概率分佈
            perturbed_preds = torch.argmax(perturbed_probs, dim=1)  # 獲取預測結果
            max_probs = torch.max(perturbed_probs, dim=1)[0]  # 獲取最大概率值（置信度）
        
        for i, (pred, true_label, confidence) in enumerate(zip(perturbed_preds, original_labels, max_probs)):
            if pred == true_label:                         # 如果擾動後預測仍然正確
                robust_indices.append(i)                  # 標記為魯棒樣本（W1類別）
            elif confidence < 0.5:                        # 如果預測錯誤且置信度低
                weak_robust_indices.append(i)             # 標記為弱魯棒樣本（W2類別）
            # 注意：不滿足上述條件的樣本將被忽略（過於難以處理）
        
        model.train()                                      # 恢復模型的訓練模式
        
        return {
            "robust_indices": robust_indices,              # 返回魯棒樣本索引
            "weak_robust_indices": weak_robust_indices     # 返回弱魯棒樣本索引
        }
    
    def compute_contrastive_curriculum_loss(self, features: torch.Tensor, labels: torch.Tensor, 
                                          robustness_info: Dict, epoch: int, total_epochs: int) -> torch.Tensor:
        """
        對比學習增強的課程學習損失
        
        基於樣本魯棒性構建正負樣本對，實現對比學習增強的課程學習
        
        Args:
            features: 特徵張量 [batch_size, feature_dim]
            labels: 標籤張量 [batch_size]
            robustness_info: 魯棒性評估結果
            epoch: 當前epoch
            total_epochs: 總epoch數
            
        Returns:
            對比學習損失
        """
        if not self.contrastive_enabled or features.size(0) < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # 動態調整對比學習溫度（訓練後期降低溫度）
        progress = epoch / total_epochs
        dynamic_temperature = self.contrastive_temperature * (1.0 + progress * 0.5)
        
        robust_indices = robustness_info.get("robust_indices", [])
        weak_robust_indices = robustness_info.get("weak_robust_indices", [])
        
        if len(robust_indices) < 2:  # 至少需要2個魯棒樣本
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        contrastive_losses = []
        
        # 為每個魯棒樣本構建對比學習損失
        for anchor_idx in robust_indices:
            anchor_feature = features[anchor_idx]  # 錨點特徵（魯棒樣本）
            anchor_label = labels[anchor_idx]
            
            # 尋找正樣本（同類別的魯棒樣本）
            positive_candidates = [idx for idx in robust_indices 
                                 if idx != anchor_idx and labels[idx] == anchor_label]
            
            # 尋找負樣本（不同類別的樣本，優先選擇弱魯棒樣本）
            negative_candidates = [idx for idx in range(features.size(0)) 
                                 if labels[idx] != anchor_label]
            
            # 優先選擇弱魯棒樣本作為困難負樣本
            hard_negatives = [idx for idx in negative_candidates if idx in weak_robust_indices]
            if hard_negatives:
                negative_candidates = hard_negatives[:int(self.negative_samples_ratio * len(positive_candidates))]
            else:
                negative_candidates = negative_candidates[:int(self.negative_samples_ratio * max(1, len(positive_candidates)))]
            
            if positive_candidates and negative_candidates:
                # 計算正樣本相似度
                positive_sims = []
                for pos_idx in positive_candidates:
                    sim = F.cosine_similarity(anchor_feature.unsqueeze(0), 
                                            features[pos_idx].unsqueeze(0))
                    positive_sims.append(torch.exp(sim / dynamic_temperature))
                
                # 計算負樣本相似度
                negative_sims = []
                for neg_idx in negative_candidates:
                    sim = F.cosine_similarity(anchor_feature.unsqueeze(0), 
                                            features[neg_idx].unsqueeze(0))
                    negative_sims.append(torch.exp(sim / dynamic_temperature))
                
                # InfoNCE損失計算
                positive_sum = sum(positive_sims)
                negative_sum = sum(negative_sims)
                
                if positive_sum > 0 and (positive_sum + negative_sum) > 0:
                    contrastive_loss = -torch.log(positive_sum / (positive_sum + negative_sum))
                    contrastive_losses.append(contrastive_loss)
        
        if contrastive_losses:
            return torch.stack(contrastive_losses).mean() * self.contrastive_alpha
        else:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
    
    def select_contrastive_samples(self, features: torch.Tensor, labels: torch.Tensor, 
                                  robustness_info: Dict) -> Dict[str, List[Tuple[int, int]]]:
        """
        智能選擇對比學習樣本對
        
        基於樣本魯棒性和特徵相似度選擇高質量的正負樣本對
        
        Args:
            features: 特徵張量
            labels: 標籤張量  
            robustness_info: 魯棒性評估結果
            
        Returns:
            包含正樣本對和負樣本對索引的字典
        """
        robust_indices = set(robustness_info.get("robust_indices", []))
        weak_robust_indices = set(robustness_info.get("weak_robust_indices", []))
        
        positive_pairs = []  # 正樣本對 (同類別)
        negative_pairs = []  # 負樣本對 (不同類別)
        
        # 構建正樣本對：優先選擇魯棒樣本之間的配對
        for i in range(features.size(0)):
            for j in range(i + 1, features.size(0)):
                if labels[i] == labels[j]:  # 同類別
                    # 魯棒樣本對優先級最高
                    if i in robust_indices and j in robust_indices:
                        similarity = F.cosine_similarity(features[i].unsqueeze(0), features[j].unsqueeze(0)).item()
                        if similarity > self.similarity_threshold:
                            positive_pairs.append((i, j))
                    # 魯棒-弱魯棒樣本對次優先級
                    elif (i in robust_indices and j in weak_robust_indices) or \
                         (i in weak_robust_indices and j in robust_indices):
                        positive_pairs.append((i, j))
        
        # 構建困難負樣本對：魯棒樣本 vs 弱魯棒樣本（不同類別）
        for robust_idx in robust_indices:
            for weak_idx in weak_robust_indices:
                if labels[robust_idx] != labels[weak_idx]:
                    # 選擇相似度較高的困難負樣本
                    similarity = F.cosine_similarity(features[robust_idx].unsqueeze(0), 
                                                   features[weak_idx].unsqueeze(0)).item()
                    if similarity > 0.3:  # 困難負樣本閾值
                        negative_pairs.append((robust_idx, weak_idx))
        
        return {
            "positive_pairs": positive_pairs,
            "negative_pairs": negative_pairs,
            "stats": {
                "num_positive_pairs": len(positive_pairs),
                "num_negative_pairs": len(negative_pairs),
                "robust_samples": len(robust_indices),
                "weak_robust_samples": len(weak_robust_indices)
            }
        }
    
    def compute_curriculum_loss(self, model: nn.Module, features: torch.Tensor, 
                               labels: torch.Tensor, epoch: int = 0, total_epochs: int = 100) -> Tuple[torch.Tensor, Dict]:
        """
        計算增強版課程學習損失
        
        Args:
            model: 當前訓練的模型
            features: 輸入特徵
            labels: 對應標籤
            epoch: 當前epoch（新增，用於自適應調整）
            total_epochs: 總epoch數（新增）
            
        Returns:
            (課程學習損失, 統計信息字典)
        """
        # 步驟1：計算類別統計信息
        statistics = self.compute_class_statistics(features, labels)
        
        # 步驟2：評估樣本難度（輕量級，用於自適應噪聲調整）
        sample_difficulties = self.evaluate_sample_difficulty(model, features, labels)
        
        # 步驟3：執行多層次魯棒性評估（使用layer_weights）
        multi_layer_robustness = self.multi_layer_robustness_evaluation(model, features, labels, statistics, epoch, total_epochs)
        
        # 步驟4：提取多層次魯棒性結果（保持兼容性）
        robustness_info = {
            "robust_indices": multi_layer_robustness["robust_indices"],
            "weak_robust_indices": multi_layer_robustness["weak_robust_indices"],
            "multi_layer_scores": multi_layer_robustness["layer_robustness_scores"],
            "shallow_robustness": multi_layer_robustness["shallow_robustness"],
            "deep_robustness": multi_layer_robustness["deep_robustness"],
            "layer_weights": multi_layer_robustness["layer_weights_used"]
        }
        
        # 步驟5：計算動態加權交叉熵損失
        logits = model.embed(features)                     # 獲取原始特徵的logits
        
        # 創建樣本權重張量，基於多層次魯棒性分數
        sample_weights = torch.ones(features.size(0), device=features.device)
        multi_layer_scores = robustness_info["multi_layer_scores"]
        
        # 使用多層次魯棒性分數進行更精細的權重分配
        for i in range(features.size(0)):
            layer_robustness = multi_layer_scores[i].item()
            difficulty_factor = sample_difficulties[i].item()
            
            # 基於多層次魯棒性和難度計算權重
            if i in robustness_info["robust_indices"]:
                # 魯棒樣本：基礎權重 + 難度獎勵 + 層次獎勵
                difficulty_bonus = 1.0 + difficulty_factor * 0.5
                layer_bonus = 1.0 + layer_robustness * 0.3  # 多層次魯棒性獎勵
                sample_weights[i] = self.robust_weights["W1"] * difficulty_bonus * layer_bonus
            elif i in robustness_info["weak_robust_indices"]:
                # 弱魯棒樣本：基礎權重 + 層次調整
                layer_adjustment = 1.0 + layer_robustness * 0.2
                sample_weights[i] = self.robust_weights["W2"] * layer_adjustment
            else:
                # 其他樣本：基於層次魯棒性分數調整
                sample_weights[i] = 0.5 + layer_robustness * 0.5
        
        # 計算加權交叉熵損失
        ce_loss = F.cross_entropy(logits, labels, reduction='none')  # 不進行reduce，保持每個樣本的損失
        weighted_loss = (sample_weights * ce_loss).mean()  # 應用權重並計算平均值
        
        # 步驟6：計算對比學習增強損失
        contrastive_loss = self.compute_contrastive_curriculum_loss(features, labels, robustness_info, epoch, total_epochs)
        
        # 整合所有損失
        total_loss = weighted_loss + contrastive_loss
        
        # 智能選擇對比樣本對（用於統計分析）
        contrastive_samples = self.select_contrastive_samples(features, labels, robustness_info)
        
        # 增強統計信息用於監控和調試（包含多層次魯棒性信息）
        stats = {
            "num_robust_samples": len(robustness_info["robust_indices"]),     # 魯棒樣本數量
            "num_weak_robust_samples": len(robustness_info["weak_robust_indices"]),  # 弱魯棒樣本數量
            "total_samples": features.size(0),                              # 總樣本數量
            "robust_ratio": len(robustness_info["robust_indices"]) / features.size(0),  # 魯棒樣本比例
            "average_weight": sample_weights.mean().item(),                 # 平均權重
            "average_difficulty": sample_difficulties.mean().item(),        # 平均樣本難度
            "adaptive_noise_scale": self.compute_adaptive_noise_scale(epoch, total_epochs),  # 當前自適應噪聲強度
            "contrastive_loss": contrastive_loss.item(),                    # 對比學習損失值
            "contrastive_pairs": contrastive_samples["stats"],             # 對比樣本對統計
            # 新增：多層次魯棒性統計
            "multi_layer_robustness": {
                "average_layer_score": multi_layer_scores.mean().item(),   # 平均多層次魯棒性分數
                "layer_weights_used": robustness_info["layer_weights"],    # 使用的層次權重
                "shallow_robust_count": len(robustness_info["shallow_robustness"]["robust_indices"]),  # 淺層魯棒樣本數
                "deep_robust_count": len(robustness_info["deep_robustness"]["robust_indices"]),        # 深層魯棒樣本數
                "multi_layer_enabled": True                                # 多層次評估啟用狀態
            },
            "enhancement_enabled": True                                     # 標記使用增強版本
        }
        
        return total_loss, stats


class ProgressiveVirtualClassGenerator:
    """
    漸進式虛擬類別生成器（增強版）
    
    實現粗粒度和細粒度虛擬類別的漸進式引入
    新增功能：多樣性增強、記憶體優化、不確定性引導、注意力增強
    """
    
    def __init__(self, num_classes: int = 1000, coarse_dropout_rate: float = 0.5, 
                 fine_noise_std: float = 0.05, min_virtual_ratio: float = 0.2):
        """
        初始化虛擬類別生成器
        
        Args:
            num_classes: 總類別數
            coarse_dropout_rate: 粗粒度虛擬類別的dropout比率
            fine_noise_std: 細粒度虛擬類別的噪聲標準差
            min_virtual_ratio: 最小虛擬類別比例
        """
        self.num_classes = num_classes                     # 總類別數
        self.coarse_dropout_rate = coarse_dropout_rate     # 粗粒度dropout比率，用於模糊圖像細節
        self.fine_noise_std = fine_noise_std               # 細粒度噪聲標準差，用於生成高真實感虛擬樣本
        self.min_virtual_ratio = min_virtual_ratio         # 最小虛擬類別比例（20%）
        self.virtual_class_id_offset = num_classes         # 虛擬類別ID偏移量，避免與真實類別衝突
        
        # 新增：增強功能配置
        self.diversity_factor = 0.8                        # 多樣性增強係數
        self.feature_cache = {}                            # 特徵快取，提升效率
        self.max_cache_size = 200                          # 最大快取大小
        
        # 優化：不確定性引導配置（降低閾值以處理更多樣本）
        self.uncertainty_enabled = True                    # 啟用不確定性引導樣本選擇
        self.uncertainty_threshold = 0.5                   # 不確定性閾值（降低以包含更多困難樣本）
        self.entropy_weight = 0.4                          # 熵值權重（提高重要性）
        self.confidence_weight = 0.6                       # 置信度權重（適當降低）
        
        # 優化：注意力機制配置（增加頭數和調整溫度）
        self.attention_enabled = True                      # 啟用注意力機制增強
        self.attention_heads = 6                           # 注意力頭數（增加以提升表現）
        self.attention_temperature = 0.08                  # 注意力溫度參數（更集中）
        self.feature_importance_cache = {}                 # 特徵重要性快取
        
    def compute_virtual_class_count(self, batch_size: int, epoch: int, total_epochs: int) -> int:
        """
        根據訓練進度動態計算虛擬類別數量
        
        Args:
            batch_size: 當前批次大小
            epoch: 當前epoch
            total_epochs: 總epoch數
            
        Returns:
            虛擬類別數量
        """
        # 實現論文中的公式：N = Batch × max([epoch/total_epoch], 0.2)
        progress_ratio = epoch / total_epochs              # 計算訓練進度比例
        virtual_ratio = max(progress_ratio, self.min_virtual_ratio)  # 確保不低於最小比例
        virtual_count = int(batch_size * virtual_ratio)   # 計算虛擬樣本數量
        return virtual_count
    
    def generate_diversity_enhanced_virtual_classes(self, features: torch.Tensor, count: int) -> torch.Tensor:
        """
        多樣性增強的虛擬類別生成 - 使用多種策略增加樣本多樣性
        
        Args:
            features: 真實特徵
            count: 需要生成的虛擬樣本數量
            
        Returns:
            多樣性增強虛擬特徵
        """
        if count == 0:
            return torch.empty(0, features.size(1), device=features.device)
        
        virtual_features = []
        
        for i in range(count):
            # 使用多種不同的策略來增加多樣性
            strategy = i % 4  # 循環使用4種策略
            
            if strategy == 0:
                # 策略1：特徵插值 + 隨機旋轉（模擬視角變化）
                if features.size(0) >= 2:
                    indices = torch.randperm(features.size(0))[:2]
                    f1, f2 = features[indices[0]], features[indices[1]]
                    alpha = torch.rand(1).item() * 0.6 + 0.2  # [0.2, 0.8]
                    interpolated = alpha * f1 + (1 - alpha) * f2
                    
                    # 添加旋轉變換（在特徵空間中模擬）
                    rotation_noise = torch.randn_like(interpolated) * 0.02
                    virtual_feature = interpolated + rotation_noise
                else:
                    virtual_feature = features[0] + torch.randn_like(features[0]) * self.fine_noise_std
                    
            elif strategy == 1:
                # 策略2：特徵縮放變化（模擬尺度變化）
                base_feature = features[torch.randint(0, features.size(0), (1,))][0]
                scale_factor = torch.rand(1).item() * 0.4 + 0.8  # [0.8, 1.2]
                scaled = base_feature * scale_factor
                scale_noise = torch.randn_like(scaled) * self.fine_noise_std * 0.5
                virtual_feature = scaled + scale_noise
                
            elif strategy == 2:
                # 策略3：局部特徵替換（模擬局部病變變化）
                base_feature = features[torch.randint(0, features.size(0), (1,))][0].clone()
                if features.size(0) > 1:
                    replacement_feature = features[torch.randint(0, features.size(0), (1,))][0]
                    
                    # 隨機選擇30%的特徵維度進行替換
                    replacement_mask = torch.rand(base_feature.shape) < 0.3
                    base_feature[replacement_mask] = replacement_feature[replacement_mask]
                
                local_noise = torch.randn_like(base_feature) * self.fine_noise_std * 0.3
                virtual_feature = base_feature + local_noise
                
            else:
                # 策略4：多尺度特徵融合（模擬不同解析度影響）
                base_feature = features[torch.randint(0, features.size(0), (1,))][0]
                
                # 生成多個尺度的噪聲並融合
                fine_noise = torch.randn_like(base_feature) * self.fine_noise_std * 0.2    # 細尺度
                medium_noise = torch.randn_like(base_feature) * self.fine_noise_std * 0.5  # 中尺度
                coarse_noise = torch.randn_like(base_feature) * self.fine_noise_std * 0.3  # 粗尺度
                
                multiscale_noise = 0.5 * fine_noise + 0.3 * medium_noise + 0.2 * coarse_noise
                virtual_feature = base_feature + multiscale_noise
            
            virtual_features.append(virtual_feature)
        
        return torch.stack(virtual_features)
    
    def generate_coarse_virtual_classes(self, features: torch.Tensor, count: int) -> torch.Tensor:
        """
        生成粗粒度虛擬類別
        
        使用dropout操作來模糊特徵，模擬語義細節缺失的樣本
        
        Args:
            features: 真實特徵 [batch_size, feature_dim]
            count: 需要生成的粗粒度虛擬樣本數量
            
        Returns:
            粗粒度虛擬特徵 [count, feature_dim]
        """
        virtual_features = []                              # 存儲生成的虛擬特徵
        
        for _ in range(count):                             # 生成指定數量的虛擬樣本
            # 隨機選擇一個真實特徵作為基礎
            base_idx = torch.randint(0, features.size(0), (1,)).item()  # 隨機選擇基礎特徵索引
            base_feature = features[base_idx].clone()      # 複製基礎特徵
            
            # 應用dropout來模糊特徵（模擬語義細節缺失）
            # 這相當於論文中提到的"drop"操作
            dropout_mask = torch.rand_like(base_feature) > self.coarse_dropout_rate  # 創建dropout掩碼
            coarse_virtual = base_feature * dropout_mask.float()  # 應用掩碼，部分特徵置零
            
            virtual_features.append(coarse_virtual)        # 添加到結果列表
        
        return torch.stack(virtual_features) if virtual_features else torch.empty(0, features.size(1), device=features.device)
    
    def generate_fine_virtual_classes(self, features: torch.Tensor, count: int) -> torch.Tensor:
        """
        生成細粒度虛擬類別
        
        通過添加高斯噪聲和特徵混合來生成高真實感的虛擬樣本
        
        Args:
            features: 真實特徵 [batch_size, feature_dim]
            count: 需要生成的細粒度虛擬樣本數量
            
        Returns:
            細粒度虛擬特徵 [count, feature_dim]
        """
        virtual_features = []                              # 存儲生成的虛擬特徵
        
        for _ in range(count):                             # 生成指定數量的虛擬樣本
            if features.size(0) >= 2:                      # 確保有足夠的樣本進行混合
                # 隨機選擇兩個不同的特徵進行混合
                indices = torch.randperm(features.size(0))[:2]  # 隨機排列並選擇前兩個
                feature1, feature2 = features[indices[0]], features[indices[1]]  # 獲取兩個特徵
                
                # 線性混合兩個特徵（增加樣本多樣性）
                mixing_ratio = torch.rand(1).item() * 0.3 + 0.35  # 混合比例在[0.35, 0.65]範圍內
                mixed_feature = mixing_ratio * feature1 + (1 - mixing_ratio) * feature2  # 線性混合
            else:
                # 如果樣本不足，直接複製現有特徵
                mixed_feature = features[0].clone()
            
            # 添加高斯噪聲來增加真實感和多樣性
            noise = torch.randn_like(mixed_feature) * self.fine_noise_std  # 生成高斯噪聲
            fine_virtual = mixed_feature + noise           # 添加噪聲到混合特徵
            
            virtual_features.append(fine_virtual)          # 添加到結果列表
        
        return torch.stack(virtual_features) if virtual_features else torch.empty(0, features.size(1), device=features.device)
    
    def compute_sample_uncertainty(self, model: nn.Module, features: torch.Tensor) -> torch.Tensor:
        """
        不確定性引導的樣本選擇機制
        
        基於預測熵值和置信度計算樣本的不確定性，用於智能選擇虛擬樣本生成的基礎樣本
        
        Args:
            model: 當前訓練的模型
            features: 特徵張量 [batch_size, feature_dim]
            
        Returns:
            樣本不確定性分數 [batch_size]
        """
        if not self.uncertainty_enabled:
            return torch.ones(features.size(0), device=features.device)
        
        model.eval()
        with torch.no_grad():
            # 獲取預測logits和概率
            logits = model.embed(features)
            probs = F.softmax(logits, dim=1)
            
            # 計算預測熵值（不確定性指標）
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # 添加小值避免log(0)
            
            # 計算最大置信度（確定性指標）
            max_confidence, _ = torch.max(probs, dim=1)
            
            # 組合不確定性分數：高熵值 + 低置信度 = 高不確定性
            uncertainty_score = (self.entropy_weight * entropy + 
                               self.confidence_weight * (1.0 - max_confidence))
            
            # 正規化到[0,1]範圍
            if uncertainty_score.std() > 0:
                uncertainty_score = (uncertainty_score - uncertainty_score.min()) / (uncertainty_score.max() - uncertainty_score.min())
            
        model.train()
        return uncertainty_score
    
    def uncertainty_guided_sample_selection(self, features: torch.Tensor, labels: torch.Tensor, 
                                          model: nn.Module, selection_count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基於不確定性進行智能樣本選擇
        
        優先選擇高不確定性樣本作為虛擬樣本生成的基礎，提升生成質量
        
        Args:
            features: 特徵張量
            labels: 標籤張量
            model: 當前模型
            selection_count: 需要選擇的樣本數量
            
        Returns:
            選擇的特徵和標籤
        """
        if selection_count >= features.size(0):
            return features, labels
        
        # 計算樣本不確定性
        uncertainty_scores = self.compute_sample_uncertainty(model, features)
        
        # 基於不確定性進行加權隨機選擇（高不確定性樣本被選中概率更高）
        selection_probs = uncertainty_scores / (uncertainty_scores.sum() + 1e-8)
        
        # 使用多項式採樣進行選擇
        try:
            selected_indices = torch.multinomial(selection_probs, selection_count, replacement=False)
        except:
            # 如果採樣失敗，使用確定性選擇（選擇不確定性最高的樣本）
            _, selected_indices = torch.topk(uncertainty_scores, selection_count)
        
        return features[selected_indices], labels[selected_indices]
    
    def compute_attention_weights(self, features: torch.Tensor, query_feature: torch.Tensor) -> torch.Tensor:
        """
        注意力機制增強的特徵生成
        
        使用多頭注意力機制計算特徵重要性，指導虛擬特徵生成
        
        Args:
            features: 特徵張量 [batch_size, feature_dim]  
            query_feature: 查詢特徵 [feature_dim]
            
        Returns:
            注意力權重 [batch_size]
        """
        if not self.attention_enabled or features.size(0) == 0:
            return torch.ones(features.size(0), device=features.device) / features.size(0)
        
        feature_dim = features.size(1)
        head_dim = feature_dim // self.attention_heads
        
        if head_dim == 0:  # 處理特徵維度小於注意力頭數的情況
            head_dim = 1
            actual_heads = feature_dim
        else:
            actual_heads = self.attention_heads
        
        attention_weights_list = []
        
        # 多頭注意力計算
        for head in range(actual_heads):
            start_idx = head * head_dim
            end_idx = min((head + 1) * head_dim, feature_dim)
            
            # 提取當前頭的特徵
            head_features = features[:, start_idx:end_idx]  # [batch_size, head_dim]
            head_query = query_feature[start_idx:end_idx]   # [head_dim]
            
            # 計算注意力分數（點積注意力）
            attention_scores = torch.matmul(head_features, head_query.unsqueeze(1)).squeeze(1)  # [batch_size]
            
            # 應用溫度縮放
            scaled_scores = attention_scores / self.attention_temperature
            
            # Softmax正規化
            head_weights = F.softmax(scaled_scores, dim=0)
            attention_weights_list.append(head_weights)
        
        # 平均多頭注意力權重
        if attention_weights_list:
            final_weights = torch.stack(attention_weights_list).mean(dim=0)
        else:
            final_weights = torch.ones(features.size(0), device=features.device) / features.size(0)
        
        return final_weights
    
    def generate_attention_enhanced_virtual_features(self, features: torch.Tensor, labels: torch.Tensor, 
                                                   target_feature: torch.Tensor, count: int) -> torch.Tensor:
        """
        注意力機制增強的虛擬特徵生成
        
        使用注意力權重指導特徵融合，生成高質量虛擬特徵
        
        Args:
            features: 候選特徵 [batch_size, feature_dim]
            labels: 對應標籤 [batch_size]  
            target_feature: 目標特徵 [feature_dim]
            count: 需要生成的虛擬特徵數量
            
        Returns:
            注意力增強的虛擬特徵 [count, feature_dim]
        """
        if not self.attention_enabled or count == 0 or features.size(0) == 0:
            return torch.empty(0, features.size(1), device=features.device)
        
        virtual_features = []
        
        for _ in range(count):
            # 計算注意力權重
            attention_weights = self.compute_attention_weights(features, target_feature)
            
            # 基於注意力權重進行加權特徵融合
            weighted_features = features * attention_weights.unsqueeze(1)  # [batch_size, feature_dim]
            attended_feature = weighted_features.sum(dim=0)  # [feature_dim]
            
            # 與目標特徵進行自適應混合
            mixing_ratio = torch.sigmoid(torch.randn(1)).item()  # 動態混合比例
            mixed_feature = mixing_ratio * attended_feature + (1 - mixing_ratio) * target_feature
            
            # 添加注意力引導的噪聲
            attention_guided_noise = torch.randn_like(mixed_feature) * self.fine_noise_std
            # 根據注意力權重的方差調整噪聲強度（高方差=更多噪聲）
            noise_scale = 1.0 + attention_weights.var().item()
            
            virtual_feature = mixed_feature + attention_guided_noise * noise_scale
            virtual_features.append(virtual_feature)
        
        return torch.stack(virtual_features) if virtual_features else torch.empty(0, features.size(1), device=features.device)
    
    def generate_virtual_classes(self, features: torch.Tensor, epoch: int, 
                                total_epochs: int, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成增強版虛擬類別（整合多種策略）
        
        Args:
            features: 真實特徵
            epoch: 當前epoch
            total_epochs: 總epoch數
            labels: 真實標籤（可選）
            
        Returns:
            (虛擬特徵, 虛擬標籤)
        """
        batch_size = features.size(0)                      # 獲取批次大小
        
        # 計算總虛擬樣本數量
        total_virtual_count = self.compute_virtual_class_count(batch_size, epoch, total_epochs)
        
        if total_virtual_count == 0:                       # 如果不需要生成虛擬樣本
            empty_features = torch.empty(0, features.size(1), device=features.device)
            empty_labels = torch.empty(0, dtype=torch.long, device=features.device)
            return empty_features, empty_labels
        
        # 動態分配不同策略的樣本數量
        # 後期訓練更多使用多樣性增強和高級功能策略
        progress = epoch / total_epochs
        
        # 策略分配（根據訓練進度調整）
        diversity_ratio = min(0.4, progress * 0.6)                   # 多樣性增強
        uncertainty_ratio = min(0.25, progress * 0.5) if self.uncertainty_enabled else 0  # 不確定性引導
        attention_ratio = min(0.2, progress * 0.4) if self.attention_enabled else 0        # 注意力增強
        coarse_ratio = max(0.15, 0.35 - progress * 0.2)             # 早期較多粗粒度，後期較少
        fine_ratio = 1.0 - diversity_ratio - uncertainty_ratio - attention_ratio - coarse_ratio
        
        # 計算各策略的樣本數量
        diversity_count = int(total_virtual_count * diversity_ratio)
        uncertainty_count = int(total_virtual_count * uncertainty_ratio)
        attention_count = int(total_virtual_count * attention_ratio)
        coarse_count = int(total_virtual_count * coarse_ratio)
        fine_count = total_virtual_count - diversity_count - uncertainty_count - attention_count - coarse_count
        
        virtual_features_list = []
        
        # 1. 生成多樣性增強虛擬類別
        if diversity_count > 0:
            diversity_virtual = self.generate_diversity_enhanced_virtual_classes(features, diversity_count)
            if diversity_virtual.size(0) > 0:
                virtual_features_list.append(diversity_virtual)
        
        # 2. 生成粗粒度虛擬類別
        if coarse_count > 0:
            coarse_virtual = self.generate_coarse_virtual_classes(features, coarse_count)
            if coarse_virtual.size(0) > 0:
                virtual_features_list.append(coarse_virtual)
        
        # 3. 生成不確定性引導的虛擬類別
        if uncertainty_count > 0 and hasattr(self, 'current_model'):  # 需要模型進行不確定性計算
            try:
                # 選擇高不確定性樣本作為基礎
                uncertain_features, uncertain_labels = self.uncertainty_guided_sample_selection(
                    features, labels if labels is not None else torch.zeros(features.size(0), dtype=torch.long, device=features.device), 
                    self.current_model, min(uncertainty_count, features.size(0))
                )
                if uncertain_features.size(0) > 0:
                    # 基於不確定性樣本生成虛擬特徵
                    uncertainty_virtual = self.generate_fine_virtual_classes(uncertain_features, uncertainty_count)
                    if uncertainty_virtual.size(0) > 0:
                        virtual_features_list.append(uncertainty_virtual)
            except:
                # 如果不確定性引導失敗，降級為普通細粒度生成
                uncertainty_virtual = self.generate_fine_virtual_classes(features, uncertainty_count)
                if uncertainty_virtual.size(0) > 0:
                    virtual_features_list.append(uncertainty_virtual)
        
        # 4. 生成注意力增強的虛擬類別
        if attention_count > 0 and features.size(0) > 1:
            # 隨機選擇目標特徵
            target_idx = torch.randint(0, features.size(0), (1,)).item()
            target_feature = features[target_idx]
            
            # 生成注意力增強的虛擬特徵
            attention_virtual = self.generate_attention_enhanced_virtual_features(
                features, labels if labels is not None else torch.zeros(features.size(0), dtype=torch.long, device=features.device), 
                target_feature, attention_count
            )
            if attention_virtual.size(0) > 0:
                virtual_features_list.append(attention_virtual)
        
        # 5. 生成細粒度虛擬類別
        if fine_count > 0:
            fine_virtual = self.generate_fine_virtual_classes(features, fine_count)
            if fine_virtual.size(0) > 0:
                virtual_features_list.append(fine_virtual)
        
        # 合併所有虛擬特徵
        if virtual_features_list:
            all_virtual_features = torch.cat(virtual_features_list, dim=0)
        else:
            all_virtual_features = torch.empty(0, features.size(1), device=features.device)
        
        # 生成虛擬標籤（使用偏移量避免與真實類別衝突）
        if all_virtual_features.size(0) > 0:
            virtual_labels = torch.arange(
                self.virtual_class_id_offset,              # 起始ID
                self.virtual_class_id_offset + all_virtual_features.size(0),  # 結束ID
                device=features.device, dtype=torch.long
            )
        else:
            virtual_labels = torch.empty(0, dtype=torch.long, device=features.device)
        
        return all_virtual_features, virtual_labels
    
    def compute_virtual_class_loss(self, model: nn.Module, virtual_features: torch.Tensor, 
                                  virtual_labels: torch.Tensor) -> torch.Tensor:
        """
        計算虛擬類別的監督損失
        
        Args:
            model: 當前模型
            virtual_features: 虛擬特徵
            virtual_labels: 虛擬標籤
            
        Returns:
            虛擬類別損失
        """
        if virtual_features.size(0) == 0:                 # 如果沒有虛擬樣本
            return torch.tensor(0.0, device=virtual_features.device, requires_grad=True)
        
        # 獲取虛擬特徵的logits
        virtual_logits = model.embed(virtual_features)     # 前向傳播獲取logits
        
        # 為虛擬類別創建one-hot標籤（因為虛擬類別不在原始分類器中）
        # 這裡我們使用一種簡化的方法：讓虛擬樣本的預測盡可能均勻分佈
        num_real_classes = virtual_logits.size(1)         # 真實類別數量
        target_probs = torch.ones_like(virtual_logits) / num_real_classes  # 均勻分佈目標
        
        # 使用KL散度損失來讓虛擬樣本的預測更加均勻
        virtual_probs = F.log_softmax(virtual_logits, dim=1)  # 獲取log概率
        virtual_loss = F.kl_div(virtual_probs, target_probs, reduction='batchmean')  # KL散度損失
        
        return virtual_loss


class PGLSOptimizationManager:
    """
    PGLS優化管理器
    
    統一管理魯棒課程學習和漸進式虛擬類別兩個組件
    """
    
    def __init__(self, num_classes: int = 1000, rcl_alpha: float = 0.2, ivc_alpha: float = 0.1):
        """
        初始化PGLS優化管理器
        
        Args:
            num_classes: 總類別數
            rcl_alpha: 魯棒課程學習的損失權重
            ivc_alpha: 虛擬類別損失的權重
        """
        self.num_classes = num_classes                     # 總類別數
        self.rcl_alpha = rcl_alpha                         # RCL損失權重（相比原論文0.5，調低以平衡整體損失）
        self.ivc_alpha = ivc_alpha                         # IVC損失權重
        
        # 初始化兩個核心組件
        self.rcl_learner = RobustCurriculumLearner(num_classes)  # 魯棒課程學習器
        self.ivc_generator = ProgressiveVirtualClassGenerator(num_classes)  # 虛擬類別生成器
        
        # 統計信息記錄
        self.optimization_stats = {
            "rcl_stats": [],                               # RCL統計信息列表
            "ivc_stats": [],                               # IVC統計信息列表
            "total_pgls_loss": [],                         # 總PGLS損失記錄
            "epochs_processed": 0                          # 已處理的epoch數量
        }
        
    def compute_pgls_loss(self, model: nn.Module, features: torch.Tensor, labels: torch.Tensor, 
                         epoch: int, total_epochs: int) -> Tuple[torch.Tensor, Dict]:
        """
        計算增強版完整PGLS損失（RCL + IVC）
        
        Args:
            model: 當前訓練的模型
            features: 輸入特徵
            labels: 對應標籤
            epoch: 當前epoch
            total_epochs: 總epoch數
            
        Returns:
            (PGLS總損失, 詳細統計信息)
        """
        # 計算增強版魯棒課程學習損失（現在包含對比學習）
        rcl_loss, rcl_stats = self.rcl_learner.compute_curriculum_loss(model, features, labels, epoch, total_epochs)
        
        # 為虛擬類別生成器提供當前模型（用於不確定性引導）
        self.ivc_generator.current_model = model
        
        # 生成增強版虛擬類別（包含不確定性引導、注意力增強）
        virtual_features, virtual_labels = self.ivc_generator.generate_virtual_classes(
            features, epoch, total_epochs, labels  # 新增labels參數
        )
        
        # 計算虛擬類別損失
        ivc_loss = self.ivc_generator.compute_virtual_class_loss(model, virtual_features, virtual_labels)
        
        # 動態調整損失權重（根據訓練進度）
        progress = epoch / total_epochs
        dynamic_rcl_alpha = self.rcl_alpha * (1.0 + progress * 0.3)  # 後期稍微增加RCL權重
        dynamic_ivc_alpha = self.ivc_alpha * (1.0 + progress * 0.5)  # 後期較多增加IVC權重
        
        # 組合總損失
        total_pgls_loss = dynamic_rcl_alpha * rcl_loss + dynamic_ivc_alpha * ivc_loss
        
        # 增強統計信息
        combined_stats = {
            "rcl_loss": rcl_loss.item(),                   # RCL損失值
            "ivc_loss": ivc_loss.item(),                   # IVC損失值
            "total_pgls_loss": total_pgls_loss.item(),     # 總PGLS損失值
            "dynamic_rcl_alpha": dynamic_rcl_alpha,        # 動態RCL權重
            "dynamic_ivc_alpha": dynamic_ivc_alpha,        # 動態IVC權重
            "rcl_stats": rcl_stats,                        # RCL詳細統計
            "ivc_stats": {
                "num_virtual_samples": virtual_features.size(0),     # 虛擬樣本數量
                "virtual_ratio": virtual_features.size(0) / features.size(0) if features.size(0) > 0 else 0,  # 虛擬樣本比例
                "diversity_enhancement_enabled": True,       # 多樣性增強啟用狀態
                "uncertainty_guided_enabled": self.ivc_generator.uncertainty_enabled,  # 不確定性引導啟用狀態
                "attention_enhanced_enabled": self.ivc_generator.attention_enabled,    # 注意力增強啟用狀態
            },
            "enhancement_info": {
                "adaptive_noise_enabled": self.rcl_learner.adaptive_noise_enabled,  # 自適應噪聲狀態
                "memory_efficient_processing": True,        # 記憶體友好處理狀態
                "multi_strategy_virtual_generation": True,  # 多策略虛擬生成狀態
                "contrastive_learning_enabled": self.rcl_learner.contrastive_enabled,  # 對比學習啟用狀態
                "uncertainty_guided_selection": self.ivc_generator.uncertainty_enabled,  # 不確定性引導選擇
                "attention_enhanced_generation": self.ivc_generator.attention_enabled,   # 注意力增強生成
            },
            "epoch": epoch,                                # 當前epoch
            "progress": progress                           # 訓練進度
        }
        
        # 記錄增強統計信息
        self.optimization_stats["rcl_stats"].append(rcl_stats)
        self.optimization_stats["ivc_stats"].append(combined_stats["ivc_stats"])
        self.optimization_stats["total_pgls_loss"].append(total_pgls_loss.item())
        self.optimization_stats["epochs_processed"] = epoch + 1
        
        return total_pgls_loss, combined_stats
    
    def get_optimization_summary(self) -> Dict:
        """
        獲取優化過程的總結統計信息
        
        Returns:
            優化總結字典
        """
        if not self.optimization_stats["total_pgls_loss"]:  # 如果沒有記錄
            return {"status": "未開始訓練"}
        
        # 計算統計摘要
        total_losses = self.optimization_stats["total_pgls_loss"]
        
        summary = {
            "epochs_processed": self.optimization_stats["epochs_processed"],     # 已處理epoch數
            "average_pgls_loss": sum(total_losses) / len(total_losses),         # 平均PGLS損失
            "min_pgls_loss": min(total_losses),                                 # 最小PGLS損失
            "max_pgls_loss": max(total_losses),                                 # 最大PGLS損失
            "loss_trend": "下降" if len(total_losses) > 1 and total_losses[-1] < total_losses[0] else "其他",  # 損失趨勢
            "rcl_alpha": self.rcl_alpha,                                        # RCL權重
            "ivc_alpha": self.ivc_alpha,                                        # IVC權重
            "total_batches_processed": len(total_losses)                        # 總處理批次數
        }
        
        return summary


def create_pgls_manager(num_classes: int = 1000, rcl_alpha: float = 0.2, ivc_alpha: float = 0.1) -> PGLSOptimizationManager:
    """
    創建PGLS優化管理器的工廠函數
    
    Args:
        num_classes: 總類別數
        rcl_alpha: RCL損失權重
        ivc_alpha: IVC損失權重
        
    Returns:
        配置好的PGLS優化管理器
    """
    return PGLSOptimizationManager(num_classes, rcl_alpha, ivc_alpha)


# 用於兼容性的簡化接口
def compute_robust_curriculum_loss(model: nn.Module, features: torch.Tensor, 
                                  labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    """
    計算魯棒課程學習損失的簡化接口
    
    Args:
        model: 訓練中的模型
        features: 輸入特徵
        labels: 對應標籤
        
    Returns:
        (RCL損失, 統計信息)
    """
    rcl_learner = RobustCurriculumLearner()               # 創建RCL學習器實例
    return rcl_learner.compute_curriculum_loss(model, features, labels)  # 計算並返回損失


def generate_progressive_virtual_classes(features: torch.Tensor, epoch: int, 
                                       total_epochs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成漸進式虛擬類別的簡化接口
    
    Args:
        features: 真實特徵
        epoch: 當前epoch
        total_epochs: 總epoch數
        
    Returns:
        (虛擬特徵, 虛擬標籤)
    """
    ivc_generator = ProgressiveVirtualClassGenerator()    # 創建IVC生成器實例
    return ivc_generator.generate_virtual_classes(features, epoch, total_epochs)  # 生成並返回虛擬類別
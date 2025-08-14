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
    魯棒課程學習器
    
    基於協方差噪聲擾動的樣本魯棒性評估，實現課程學習策略
    """
    
    def __init__(self, num_classes: int = 1000, noise_scale: float = 0.1, robust_threshold: float = 0.7):
        """
        初始化魯棒課程學習器
        
        Args:
            num_classes: 總類別數
            noise_scale: 噪聲縮放係數 λ
            robust_threshold: 魯棒性閾值，用於分類魯棒/弱魯棒樣本
        """
        self.num_classes = num_classes                    # 總類別數，用於計算類別統計信息
        self.noise_scale = noise_scale                    # 噪聲縮放係數，控制擾動強度
        self.robust_threshold = robust_threshold          # 魯棒性閾值，區分魯棒和弱魯棒樣本
        self.class_statistics = {}                        # 存儲每個類別的統計信息（均值、協方差）
        self.robust_weights = {"W1": 2.0, "W2": 1.0}    # 魯棒樣本權重W1=2，弱魯棒樣本權重W2=1
        
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
    
    def compute_curriculum_loss(self, model: nn.Module, features: torch.Tensor, 
                               labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        計算課程學習損失
        
        Args:
            model: 當前訓練的模型
            features: 輸入特徵
            labels: 對應標籤
            
        Returns:
            (課程學習損失, 統計信息字典)
        """
        # 步驟1：計算類別統計信息
        statistics = self.compute_class_statistics(features, labels)
        
        # 步驟2：生成協方差噪聲擾動
        perturbed_features = self.generate_covariance_perturbation(features, labels, statistics)
        
        # 步驟3：評估樣本魯棒性
        robustness_info = self.evaluate_sample_robustness(model, features, perturbed_features, labels)
        
        # 步驟4：計算加權交叉熵損失
        logits = model.embed(features)                     # 獲取原始特徵的logits
        
        # 創建樣本權重張量，默認權重為1
        sample_weights = torch.ones(features.size(0), device=features.device)
        
        # 為魯棒樣本分配較高權重
        for idx in robustness_info["robust_indices"]:
            sample_weights[idx] = self.robust_weights["W1"]  # W1 = 2.0
        
        # 為弱魯棒樣本分配標準權重
        for idx in robustness_info["weak_robust_indices"]:
            sample_weights[idx] = self.robust_weights["W2"]  # W2 = 1.0
        
        # 計算加權交叉熵損失
        ce_loss = F.cross_entropy(logits, labels, reduction='none')  # 不進行reduce，保持每個樣本的損失
        weighted_loss = (sample_weights * ce_loss).mean()  # 應用權重並計算平均值
        
        # 統計信息用於監控和調試
        stats = {
            "num_robust_samples": len(robustness_info["robust_indices"]),     # 魯棒樣本數量
            "num_weak_robust_samples": len(robustness_info["weak_robust_indices"]),  # 弱魯棒樣本數量
            "total_samples": features.size(0),                              # 總樣本數量
            "robust_ratio": len(robustness_info["robust_indices"]) / features.size(0),  # 魯棒樣本比例
            "average_weight": sample_weights.mean().item()                  # 平均權重
        }
        
        return weighted_loss, stats


class ProgressiveVirtualClassGenerator:
    """
    漸進式虛擬類別生成器
    
    實現粗粒度和細粒度虛擬類別的漸進式引入
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
                
                # 線性混合兩個特徵（模擬病理變異或成像條件變化）
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
    
    def generate_virtual_classes(self, features: torch.Tensor, epoch: int, 
                                total_epochs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成虛擬類別（粗粒度 + 細粒度）
        
        Args:
            features: 真實特徵
            epoch: 當前epoch
            total_epochs: 總epoch數
            
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
        
        # 分配粗粒度和細粒度樣本數量（各佔一半）
        coarse_count = total_virtual_count // 2           # 粗粒度樣本數量
        fine_count = total_virtual_count - coarse_count   # 細粒度樣本數量（處理奇數情況）
        
        # 生成粗粒度虛擬類別
        coarse_virtual = self.generate_coarse_virtual_classes(features, coarse_count)
        
        # 生成細粒度虛擬類別
        fine_virtual = self.generate_fine_virtual_classes(features, fine_count)
        
        # 合併所有虛擬特徵
        if coarse_virtual.size(0) > 0 and fine_virtual.size(0) > 0:
            all_virtual_features = torch.cat([coarse_virtual, fine_virtual], dim=0)  # 連接兩種虛擬特徵
        elif coarse_virtual.size(0) > 0:
            all_virtual_features = coarse_virtual          # 只有粗粒度特徵
        elif fine_virtual.size(0) > 0:
            all_virtual_features = fine_virtual            # 只有細粒度特徵
        else:
            all_virtual_features = torch.empty(0, features.size(1), device=features.device)  # 空張量
        
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
        self.rcl_alpha = rcl_alpha                         # RCL損失權重（相比原論文0.5，調低避免與醫學損失衝突）
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
        計算完整的PGLS損失（RCL + IVC）
        
        Args:
            model: 當前訓練的模型
            features: 輸入特徵
            labels: 對應標籤
            epoch: 當前epoch
            total_epochs: 總epoch數
            
        Returns:
            (PGLS總損失, 詳細統計信息)
        """
        # 計算魯棒課程學習損失
        rcl_loss, rcl_stats = self.rcl_learner.compute_curriculum_loss(model, features, labels)
        
        # 生成虛擬類別
        virtual_features, virtual_labels = self.ivc_generator.generate_virtual_classes(
            features, epoch, total_epochs
        )
        
        # 計算虛擬類別損失
        ivc_loss = self.ivc_generator.compute_virtual_class_loss(model, virtual_features, virtual_labels)
        
        # 組合總損失
        total_pgls_loss = self.rcl_alpha * rcl_loss + self.ivc_alpha * ivc_loss
        
        # 整合統計信息
        combined_stats = {
            "rcl_loss": rcl_loss.item(),                   # RCL損失值
            "ivc_loss": ivc_loss.item(),                   # IVC損失值
            "total_pgls_loss": total_pgls_loss.item(),     # 總PGLS損失值
            "rcl_stats": rcl_stats,                        # RCL詳細統計
            "ivc_stats": {
                "num_virtual_samples": virtual_features.size(0),     # 虛擬樣本數量
                "virtual_ratio": virtual_features.size(0) / features.size(0) if features.size(0) > 0 else 0  # 虛擬樣本比例
            },
            "epoch": epoch,                                # 當前epoch
            "progress": epoch / total_epochs               # 訓練進度
        }
        
        # 記錄統計信息
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
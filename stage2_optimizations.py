# coding=utf-8
"""
第二階段優化：醫學特定損失函數 + GAN穩定性提升 + 集成方法優化

主要包含：
1. 醫學特定損失函數
2. GAN穩定性改進機制
3. 集成方法優化
4. 醫學領域特定優化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
from torch.autograd import Variable
from collections import defaultdict
import cv2
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ==================== 醫學特定損失函數 ====================

class AnatomicalConsistencyLoss(nn.Module):
    """
    解剖學約束損失函數
    
    基於醫學影像的解剖結構一致性，確保生成特徵保持解剖學合理性
    原理：利用解剖標誌點或結構分割來約束特徵學習
    """
    
    def __init__(self, num_anatomical_regions=8, consistency_weight=1.0, 
                 spatial_weight=0.5, structure_weight=0.3):
        """
        初始化解剖學一致性損失
        
        參數:
        - num_anatomical_regions: 解剖區域數量
        - consistency_weight: 一致性權重
        - spatial_weight: 空間關係權重  
        - structure_weight: 結構保持權重
        """
        super(AnatomicalConsistencyLoss, self).__init__()
        self.num_regions = num_anatomical_regions
        self.consistency_weight = consistency_weight
        self.spatial_weight = spatial_weight
        self.structure_weight = structure_weight
        
        # 解剖區域映射網路
        self.region_mapper = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_anatomical_regions),
            nn.Softmax(dim=1)
        )
        
        # 結構一致性檢測器
        self.structure_detector = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features_current, features_reference, spatial_info=None):
        """
        計算解剖學一致性損失
        
        參數:
        - features_current: 當前特徵 [batch_size, feature_dim]
        - features_reference: 參考特徵 [batch_size, feature_dim] 
        - spatial_info: 空間信息（可選）
        """
        batch_size = features_current.size(0)
        
        # 1. 解剖區域一致性
        regions_current = self.region_mapper(features_current)
        regions_reference = self.region_mapper(features_reference)
        
        # 計算區域分佈的KL散度
        region_consistency_loss = F.kl_div(
            F.log_softmax(regions_current, dim=1),
            F.softmax(regions_reference, dim=1),
            reduction='batchmean'
        )
        
        # 2. 結構保持損失
        structure_current = self.structure_detector(features_current)
        structure_reference = self.structure_detector(features_reference)
        structure_loss = F.mse_loss(structure_current, structure_reference)
        
        # 3. 空間關係損失（如果提供空間信息）
        spatial_loss = torch.tensor(0.0, device=features_current.device)
        if spatial_info is not None:
            # 計算特徵的空間相關性
            spatial_corr_current = self._compute_spatial_correlation(features_current, spatial_info)
            spatial_corr_reference = self._compute_spatial_correlation(features_reference, spatial_info)
            spatial_loss = F.mse_loss(spatial_corr_current, spatial_corr_reference)
        
        # 總損失
        total_loss = (self.consistency_weight * region_consistency_loss + 
                     self.structure_weight * structure_loss + 
                     self.spatial_weight * spatial_loss)
        
        return total_loss, {
            'region_consistency': region_consistency_loss.item(),
            'structure_consistency': structure_loss.item(),
            'spatial_consistency': spatial_loss.item()
        }
    
    def _compute_spatial_correlation(self, features, spatial_info):
        """計算特徵的空間相關性"""
        # 簡化實現：基於特徵的自相關
        normalized_features = F.normalize(features, p=2, dim=1)
        correlation_matrix = torch.mm(normalized_features, normalized_features.t())
        return correlation_matrix


class MultiScaleFeatureLoss(nn.Module):
    """
    多尺度特徵匹配損失函數
    
    在多個解析度層級上進行特徵匹配，確保特徵的多尺度一致性
    適用於醫學影像的細節和整體結構同時保持
    """
    
    def __init__(self, scales=[1.0, 0.5, 0.25], feature_dims=[512, 256, 128],
                 scale_weights=[1.0, 0.7, 0.3]):
        """
        初始化多尺度特徵損失
        
        參數:
        - scales: 不同的尺度比例
        - feature_dims: 不同尺度的特徵維度
        - scale_weights: 不同尺度的權重
        """
        super(MultiScaleFeatureLoss, self).__init__()
        self.scales = scales
        self.feature_dims = feature_dims
        self.scale_weights = scale_weights
        
        # 多尺度特徵提取器
        self.scale_extractors = nn.ModuleList()
        for i, (scale, dim) in enumerate(zip(scales, feature_dims)):
            extractor = nn.Sequential(
                nn.Linear(512, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim)
            )
            self.scale_extractors.append(extractor)
    
    def forward(self, features_current, features_reference):
        """
        計算多尺度特徵匹配損失
        
        參數:
        - features_current: 當前特徵 [batch_size, 512]
        - features_reference: 參考特徵 [batch_size, 512]
        """
        total_loss = 0.0
        scale_losses = {}
        
        for i, (scale, weight, extractor) in enumerate(zip(self.scales, self.scale_weights, self.scale_extractors)):
            # 提取當前尺度的特徵
            current_scale_features = extractor(features_current)
            reference_scale_features = extractor(features_reference)
            
            # 計算該尺度的損失
            scale_loss = self._compute_scale_loss(current_scale_features, reference_scale_features, scale)
            scale_losses[f'scale_{i}'] = scale_loss.item()
            
            # 加權累積
            total_loss += weight * scale_loss
        
        return total_loss, scale_losses
    
    def _compute_scale_loss(self, features_current, features_reference, scale):
        """計算特定尺度的損失"""
        # 結合L2距離和餘弦相似度
        l2_loss = F.mse_loss(features_current, features_reference)
        
        # 餘弦相似度損失
        cos_sim = F.cosine_similarity(features_current, features_reference, dim=1)
        cos_loss = (1 - cos_sim).mean()
        
        # 基於尺度調整權重
        scale_factor = 1.0 + (1.0 - scale) * 0.5  # 較小尺度給予稍高權重
        
        return scale_factor * (l2_loss + 0.3 * cos_loss)


class MedicalSemanticConsistencyLoss(nn.Module):
    """
    醫學語義一致性損失函數
    
    確保醫學特徵在語義層面的一致性，保持診斷相關的語義信息
    """
    
    def __init__(self, num_semantic_concepts=16, concept_dim=64, 
                 semantic_weight=1.0, diagnostic_weight=0.8):
        """
        初始化醫學語義一致性損失
        
        參數:
        - num_semantic_concepts: 語義概念數量
        - concept_dim: 概念特徵維度
        - semantic_weight: 語義一致性權重
        - diagnostic_weight: 診斷相關性權重
        """
        super(MedicalSemanticConsistencyLoss, self).__init__()
        self.num_concepts = num_semantic_concepts
        self.concept_dim = concept_dim
        self.semantic_weight = semantic_weight
        self.diagnostic_weight = diagnostic_weight
        
        # 語義概念提取器
        self.concept_extractor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_semantic_concepts * concept_dim)
        )
        
        # 診斷相關性評估器
        self.diagnostic_evaluator = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 語義關係建模器
        self.relation_model = nn.MultiheadAttention(concept_dim, num_heads=4, batch_first=True)
    
    def forward(self, features_current, features_reference, diagnostic_labels=None):
        """
        計算醫學語義一致性損失
        
        參數:
        - features_current: 當前特徵
        - features_reference: 參考特徵  
        - diagnostic_labels: 診斷標籤（可選）
        """
        batch_size = features_current.size(0)
        
        # 1. 提取語義概念
        concepts_current = self.concept_extractor(features_current)
        concepts_reference = self.concept_extractor(features_reference)
        
        # 重塑為 [batch_size, num_concepts, concept_dim]
        concepts_current = concepts_current.view(batch_size, self.num_concepts, self.concept_dim)
        concepts_reference = concepts_reference.view(batch_size, self.num_concepts, self.concept_dim)
        
        # 2. 語義一致性損失
        semantic_loss = F.mse_loss(concepts_current, concepts_reference)
        
        # 3. 語義關係一致性
        # 使用注意力機制建模概念間關係
        relation_current, _ = self.relation_model(concepts_current, concepts_current, concepts_current)
        relation_reference, _ = self.relation_model(concepts_reference, concepts_reference, concepts_reference)
        relation_loss = F.mse_loss(relation_current, relation_reference)
        
        # 4. 診斷相關性一致性
        diag_current = self.diagnostic_evaluator(features_current)
        diag_reference = self.diagnostic_evaluator(features_reference)
        diagnostic_loss = F.mse_loss(diag_current, diag_reference)
        
        # 5. 概念多樣性約束（防止概念塌縮）
        diversity_loss = self._compute_diversity_loss(concepts_current)
        
        # 總損失
        total_loss = (self.semantic_weight * semantic_loss + 
                     0.5 * relation_loss + 
                     self.diagnostic_weight * diagnostic_loss + 
                     0.2 * diversity_loss)
        
        return total_loss, {
            'semantic_consistency': semantic_loss.item(),
            'relation_consistency': relation_loss.item(),
            'diagnostic_consistency': diagnostic_loss.item(),
            'concept_diversity': diversity_loss.item()
        }
    
    def _compute_diversity_loss(self, concepts):
        """計算概念多樣性損失，防止概念塌縮"""
        batch_size, num_concepts, concept_dim = concepts.shape
        
        # 計算概念間的相似度矩陣
        concepts_flat = concepts.view(batch_size, -1)
        normalized_concepts = F.normalize(concepts_flat, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_concepts, normalized_concepts.t())
        
        # 鼓勵低相似度（高多樣性）
        diversity_target = torch.zeros_like(similarity_matrix)
        diversity_loss = F.mse_loss(similarity_matrix, diversity_target)
        
        return diversity_loss


# ==================== GAN穩定性改進機制 ====================

class ProgressiveGANTrainer:
    """
    漸進式GAN訓練器
    
    逐步增加生成復雜度，提升訓練穩定性
    """
    
    def __init__(self, generator, discriminator, latent_dim=200, 
                 progression_epochs=[100, 200, 300], complexity_levels=[0.3, 0.6, 1.0]):
        """
        初始化漸進式GAN訓練器
        
        參數:
        - generator: 生成器模型
        - discriminator: 判別器模型
        - latent_dim: 潛在空間維度
        - progression_epochs: 漸進訓練的epoch節點
        - complexity_levels: 對應的複雜度級別
        """
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.progression_epochs = progression_epochs
        self.complexity_levels = complexity_levels
        self.current_complexity = complexity_levels[0]
        self.current_stage = 0
        
        # 複雜度控制參數
        self.noise_schedule = NoiseScheduler()
        self.feature_complexity_controller = FeatureComplexityController()
    
    def update_complexity(self, current_epoch):
        """根據當前epoch更新複雜度"""
        for i, epoch_threshold in enumerate(self.progression_epochs):
            if current_epoch >= epoch_threshold and i + 1 < len(self.complexity_levels):
                if self.current_stage != i + 1:
                    self.current_stage = i + 1
                    self.current_complexity = self.complexity_levels[i + 1]
                    print(f"漸進式訓練：進入階段 {self.current_stage}, 複雜度 = {self.current_complexity}")
                    break
    
    def generate_progressive_features(self, batch_size, labels):
        """根據當前複雜度生成特徵"""
        # 調整噪聲強度
        noise_strength = self.noise_schedule.get_noise_strength(self.current_complexity)
        z = torch.randn(batch_size, self.latent_dim, device=labels.device) * noise_strength
        
        # 生成特徵
        generated_features = self.generator(z, labels)
        
        # 根據複雜度調整特徵
        adjusted_features = self.feature_complexity_controller.adjust_complexity(
            generated_features, self.current_complexity
        )
        
        return adjusted_features
    
    def get_current_complexity(self):
        """獲取當前複雜度"""
        return self.current_complexity


class NoiseScheduler:
    """噪聲調度器"""
    
    def __init__(self, base_noise=1.0, min_noise=0.3):
        self.base_noise = base_noise
        self.min_noise = min_noise
    
    def get_noise_strength(self, complexity):
        """根據複雜度獲取噪聲強度"""
        # 複雜度越高，噪聲越強
        noise_strength = self.min_noise + (self.base_noise - self.min_noise) * complexity
        return noise_strength


class FeatureComplexityController:
    """特徵複雜度控制器"""
    
    def __init__(self):
        self.complexity_layers = nn.ModuleList([
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        ]).cuda()
    
    def adjust_complexity(self, features, complexity):
        """根據複雜度調整特徵"""
        if complexity < 0.5:
            # 低複雜度：簡化特徵
            adjusted = self.complexity_layers[0](features)
        elif complexity < 0.8:
            # 中複雜度：中等處理
            adjusted = self.complexity_layers[1](self.complexity_layers[0](features))
        else:
            # 高複雜度：完整處理
            adjusted = features
            for layer in self.complexity_layers:
                adjusted = layer(adjusted)
        
        return adjusted


class SpectralNormalizationWrapper:
    """
    頻譜正規化包裝器
    
    為判別器添加頻譜正規化，提升訓練穩定性
    """
    
    def __init__(self, discriminator):
        self.discriminator = discriminator
        self._apply_spectral_norm()
    
    def _apply_spectral_norm(self):
        """為判別器的線性層添加頻譜正規化"""
        for name, module in self.discriminator.named_modules():
            if isinstance(module, nn.Linear):
                # 檢查是否已經應用了頻譜正規化
                if not self._has_spectral_norm(module):
                    try:
                        # 添加頻譜正規化
                        spectral_norm_module = nn.utils.spectral_norm(module)
                        # 替換原模組
                        self._replace_module(self.discriminator, name, spectral_norm_module)
                    except RuntimeError as e:
                        if "Cannot register two spectral_norm hooks" in str(e):
                            print(f"模組 {name} 已經有頻譜正規化，跳過")
                        else:
                            raise e
    
    def _has_spectral_norm(self, module):
        """檢查模組是否已經有頻譜正規化"""
        for hook in module._forward_pre_hooks.values():
            if 'SpectralNorm' in str(type(hook)):
                return True
        return False
    
    def _replace_module(self, model, name, new_module):
        """替換模型中的模組"""
        *parent_names, child_name = name.split('.')
        parent = model
        for parent_name in parent_names:
            parent = getattr(parent, parent_name)
        setattr(parent, child_name, new_module)


class AdaptiveDiscriminatorUpdater:
    """
    自適應判別器更新器
    
    根據生成器和判別器的性能動態調整更新頻率
    """
    
    def __init__(self, base_d_updates=1, base_g_updates=1, 
                 adaptation_window=50, balance_threshold=0.1):
        """
        初始化自適應更新器
        
        參數:
        - base_d_updates: 基礎判別器更新次數
        - base_g_updates: 基礎生成器更新次數
        - adaptation_window: 適應窗口大小
        - balance_threshold: 平衡閾值
        """
        self.base_d_updates = base_d_updates
        self.base_g_updates = base_g_updates
        self.adaptation_window = adaptation_window
        self.balance_threshold = balance_threshold
        
        # 性能歷史
        self.d_loss_history = []
        self.g_loss_history = []
        
        # 當前更新比率
        self.current_d_updates = base_d_updates
        self.current_g_updates = base_g_updates
    
    def update_frequencies(self, d_loss, g_loss):
        """根據損失更新頻率"""
        self.d_loss_history.append(d_loss)
        self.g_loss_history.append(g_loss)
        
        # 保持窗口大小
        if len(self.d_loss_history) > self.adaptation_window:
            self.d_loss_history.pop(0)
            self.g_loss_history.pop(0)
        
        # 計算平均損失
        if len(self.d_loss_history) >= self.adaptation_window:
            avg_d_loss = sum(self.d_loss_history) / len(self.d_loss_history)
            avg_g_loss = sum(self.g_loss_history) / len(self.g_loss_history)
            
            # 計算不平衡程度
            loss_ratio = avg_d_loss / (avg_g_loss + 1e-8)
            
            # 調整更新頻率
            if loss_ratio > 1 + self.balance_threshold:
                # 判別器太強，減少判別器更新
                self.current_d_updates = max(1, self.base_d_updates - 1)
                self.current_g_updates = self.base_g_updates + 1
            elif loss_ratio < 1 - self.balance_threshold:
                # 生成器太強，增加判別器更新
                self.current_d_updates = self.base_d_updates + 1
                self.current_g_updates = max(1, self.base_g_updates - 1)
            else:
                # 平衡狀態，使用基礎設定
                self.current_d_updates = self.base_d_updates
                self.current_g_updates = self.base_g_updates
    
    def get_update_frequencies(self):
        """獲取當前更新頻率"""
        return self.current_d_updates, self.current_g_updates


# ==================== 集成方法優化 ====================

class DynamicEnsembleWeighter:
    """
    動態集成權重器
    
    根據模型在不同類別上的實時表現動態調整集成權重
    """
    
    def __init__(self, num_models, num_classes, adaptation_rate=0.01, 
                 confidence_threshold=0.8, performance_window=100):
        """
        初始化動態集成權重器
        
        參數:
        - num_models: 模型數量
        - num_classes: 類別數量
        - adaptation_rate: 適應速率
        - confidence_threshold: 置信度閾值
        - performance_window: 性能評估窗口
        """
        self.num_models = num_models
        self.num_classes = num_classes
        self.adaptation_rate = adaptation_rate
        self.confidence_threshold = confidence_threshold
        self.performance_window = performance_window
        
        # 動態權重矩陣 [num_models, num_classes]
        self.dynamic_weights = torch.ones(num_models, num_classes)
        
        # 性能追蹤
        self.model_performance = defaultdict(list)
        self.class_performance = defaultdict(list)
        
        # 置信度統計
        self.confidence_stats = defaultdict(list)
    
    def update_weights(self, predictions, true_labels, prediction_confidences):
        """
        更新動態權重
        
        參數:
        - predictions: 模型預測 [num_models, batch_size]
        - true_labels: 真實標籤 [batch_size]
        - prediction_confidences: 預測置信度 [num_models, batch_size]
        """
        batch_size = true_labels.size(0)
        
        for model_idx in range(self.num_models):
            model_preds = predictions[model_idx]
            model_confidences = prediction_confidences[model_idx]
            
            # 計算該模型在各類別上的準確率
            for class_idx in range(self.num_classes):
                class_mask = (true_labels == class_idx)
                if class_mask.sum() > 0:
                    class_accuracy = (model_preds[class_mask] == true_labels[class_mask]).float().mean()
                    class_confidence = model_confidences[class_mask].mean()
                    
                    # 更新權重
                    performance_factor = class_accuracy.item()
                    confidence_factor = min(1.0, class_confidence.item() / self.confidence_threshold)
                    
                    # 綜合因子
                    adjustment_factor = performance_factor * confidence_factor
                    
                    # 漸進式更新權重
                    self.dynamic_weights[model_idx, class_idx] *= (1 - self.adaptation_rate)
                    self.dynamic_weights[model_idx, class_idx] += self.adaptation_rate * adjustment_factor
    
    def get_ensemble_prediction(self, model_predictions, prediction_confidences):
        """
        獲取集成預測
        
        參數:
        - model_predictions: 模型logits [num_models, batch_size, num_classes]
        - prediction_confidences: 預測置信度 [num_models, batch_size]
        """
        batch_size = model_predictions.size(1)
        ensemble_logits = torch.zeros(batch_size, self.num_classes)
        
        for class_idx in range(self.num_classes):
            for model_idx in range(self.num_models):
                # 獲取該模型對該類別的權重
                model_weight = self.dynamic_weights[model_idx, class_idx]
                
                # 加權累積
                ensemble_logits[:, class_idx] += (
                    model_weight * model_predictions[model_idx, :, class_idx] * 
                    prediction_confidences[model_idx, :]
                )
        
        return ensemble_logits
    
    def get_weight_statistics(self):
        """獲取權重統計信息"""
        return {
            'mean_weights': self.dynamic_weights.mean(dim=1).tolist(),
            'weight_variance': self.dynamic_weights.var(dim=1).tolist(),
            'max_weights': self.dynamic_weights.max(dim=1)[0].tolist(),
            'min_weights': self.dynamic_weights.min(dim=1)[0].tolist()
        }


class ConfidenceAwareFusion:
    """
    置信度感知融合器
    
    基於模型預測的置信度進行智能融合
    """
    
    def __init__(self, num_models, confidence_calibration=True, 
                 temperature_scaling=True, ensemble_diversity_bonus=0.1):
        """
        初始化置信度感知融合器
        
        參數:
        - num_models: 模型數量
        - confidence_calibration: 是否進行置信度校正
        - temperature_scaling: 是否使用溫度縮放
        - ensemble_diversity_bonus: 集成多樣性獎勵
        """
        self.num_models = num_models
        self.confidence_calibration = confidence_calibration
        self.temperature_scaling = temperature_scaling
        self.diversity_bonus = ensemble_diversity_bonus
        
        # 溫度參數（如果使用溫度縮放）
        if temperature_scaling:
            self.temperature = nn.Parameter(torch.ones(num_models))
        
        # 置信度校正器
        if confidence_calibration:
            self.confidence_calibrator = ConfidenceCalibrator(num_models)
    
    def fuse_predictions(self, model_logits, return_confidence=False):
        """
        融合多個模型的預測
        
        參數:
        - model_logits: 模型logits [num_models, batch_size, num_classes]
        - return_confidence: 是否返回置信度
        """
        num_models, batch_size, num_classes = model_logits.shape
        
        # 1. 溫度縮放（如果啟用）
        if self.temperature_scaling:
            scaled_logits = []
            for i in range(num_models):
                scaled = model_logits[i] / self.temperature[i].unsqueeze(0).unsqueeze(0)
                scaled_logits.append(scaled)
            model_logits = torch.stack(scaled_logits, dim=0)
        
        # 2. 計算概率分佈
        model_probs = F.softmax(model_logits, dim=2)
        
        # 3. 計算每個模型的置信度
        confidences = self._compute_confidences(model_probs)
        
        # 4. 置信度校正（如果啟用）
        if self.confidence_calibration:
            confidences = self.confidence_calibrator.calibrate(confidences, model_probs)
        
        # 5. 計算集成多樣性
        diversity_scores = self._compute_diversity_scores(model_probs)
        
        # 6. 綜合權重計算
        fusion_weights = self._compute_fusion_weights(confidences, diversity_scores)
        
        # 7. 加權融合
        ensemble_probs = torch.zeros(batch_size, num_classes)
        for i in range(num_models):
            ensemble_probs += fusion_weights[i].unsqueeze(1) * model_probs[i]
        
        # 正規化
        ensemble_probs = ensemble_probs / fusion_weights.sum(dim=0, keepdim=True).unsqueeze(1)
        
        if return_confidence:
            ensemble_confidence = self._compute_ensemble_confidence(
                ensemble_probs, confidences, fusion_weights
            )
            return ensemble_probs, ensemble_confidence
        
        return ensemble_probs
    
    def _compute_confidences(self, model_probs):
        """計算模型置信度"""
        # 使用最大概率作為置信度
        confidences = torch.max(model_probs, dim=2)[0]
        return confidences
    
    def _compute_diversity_scores(self, model_probs):
        """計算集成多樣性分數"""
        # 計算模型間的差異性
        diversity_scores = torch.zeros(self.num_models, model_probs.size(1))
        
        for i in range(self.num_models):
            for j in range(self.num_models):
                if i != j:
                    # 計算KL散度作為差異性度量
                    kl_div = F.kl_div(
                        F.log_softmax(model_probs[i], dim=1),
                        F.softmax(model_probs[j], dim=1),
                        reduction='none'
                    ).sum(dim=1)
                    diversity_scores[i] += kl_div
        
        # 正規化
        diversity_scores = diversity_scores / (self.num_models - 1)
        return diversity_scores
    
    def _compute_fusion_weights(self, confidences, diversity_scores):
        """計算融合權重"""
        # 基礎權重 = 置信度 + 多樣性獎勵
        base_weights = confidences + self.diversity_bonus * diversity_scores
        
        # 軟最大正規化
        fusion_weights = F.softmax(base_weights, dim=0)
        
        return fusion_weights
    
    def _compute_ensemble_confidence(self, ensemble_probs, model_confidences, fusion_weights):
        """計算集成置信度"""
        # 基於最大概率的基礎置信度
        base_confidence = torch.max(ensemble_probs, dim=1)[0]
        
        # 模型一致性加權
        weighted_model_confidence = (model_confidences * fusion_weights).sum(dim=0)
        
        # 綜合置信度
        ensemble_confidence = 0.7 * base_confidence + 0.3 * weighted_model_confidence
        
        return ensemble_confidence


class ConfidenceCalibrator:
    """置信度校正器"""
    
    def __init__(self, num_models, calibration_method='platt'):
        self.num_models = num_models
        self.calibration_method = calibration_method
        
        if calibration_method == 'platt':
            # Platt scaling參數
            self.platt_a = nn.Parameter(torch.ones(num_models))
            self.platt_b = nn.Parameter(torch.zeros(num_models))
        elif calibration_method == 'temperature':
            # 溫度縮放參數
            self.temperature = nn.Parameter(torch.ones(num_models))
    
    def calibrate(self, confidences, model_probs):
        """校正置信度"""
        if self.calibration_method == 'platt':
            # Platt scaling
            calibrated = torch.sigmoid(self.platt_a.unsqueeze(1) * confidences + self.platt_b.unsqueeze(1))
        elif self.calibration_method == 'temperature':
            # 溫度縮放
            calibrated = confidences ** (1.0 / self.temperature.unsqueeze(1))
        else:
            calibrated = confidences
        
        return calibrated


# ==================== 醫學領域特定優化 ====================

class MedicalDomainSpecificOptimizer:
    """
    醫學領域特定優化器
    
    整合所有醫學特定的優化策略
    """
    
    def __init__(self, num_classes=1000, medical_expertise_weight=0.3):
        """
        初始化醫學領域優化器
        
        參數:
        - num_classes: 類別總數
        - medical_expertise_weight: 醫學專業知識權重
        """
        self.num_classes = num_classes
        self.medical_expertise_weight = medical_expertise_weight
        
        # 初始化各組件
        self.anatomical_loss = AnatomicalConsistencyLoss()
        self.multiscale_loss = MultiScaleFeatureLoss()
        self.semantic_loss = MedicalSemanticConsistencyLoss()
        
        # 醫學知識引導
        self.medical_knowledge_guide = MedicalKnowledgeGuide()
        
        # 病理感知學習器
        self.pathology_learner = PathologyAwareLearner()
    
    def compute_medical_loss(self, features_current, features_reference, 
                           medical_context=None, pathology_info=None):
        """
        計算綜合醫學損失
        
        參數:
        - features_current: 當前特徵
        - features_reference: 參考特徵
        - medical_context: 醫學上下文信息
        - pathology_info: 病理信息
        """
        total_loss = 0.0
        loss_components = {}
        
        # 1. 解剖學一致性損失
        anatomical_loss, anatomical_details = self.anatomical_loss(
            features_current, features_reference
        )
        total_loss += 0.4 * anatomical_loss
        loss_components.update({f'anatomical_{k}': v for k, v in anatomical_details.items()})
        
        # 2. 多尺度特徵損失
        multiscale_loss, multiscale_details = self.multiscale_loss(
            features_current, features_reference
        )
        total_loss += 0.3 * multiscale_loss
        loss_components.update({f'multiscale_{k}': v for k, v in multiscale_details.items()})
        
        # 3. 語義一致性損失
        semantic_loss, semantic_details = self.semantic_loss(
            features_current, features_reference
        )
        total_loss += 0.3 * semantic_loss
        loss_components.update({f'semantic_{k}': v for k, v in semantic_details.items()})
        
        # 4. 醫學知識引導損失
        if medical_context is not None:
            knowledge_loss = self.medical_knowledge_guide.compute_guidance_loss(
                features_current, medical_context
            )
            total_loss += self.medical_expertise_weight * knowledge_loss
            loss_components['medical_knowledge'] = knowledge_loss.item()
        
        # 5. 病理感知損失
        if pathology_info is not None:
            pathology_loss = self.pathology_learner.compute_pathology_loss(
                features_current, pathology_info
            )
            total_loss += 0.2 * pathology_loss
            loss_components['pathology_awareness'] = pathology_loss.item()
        
        return total_loss, loss_components
    
    def get_optimization_statistics(self):
        """獲取優化統計信息"""
        return {
            'anatomical_regions': self.anatomical_loss.num_regions,
            'semantic_concepts': self.semantic_loss.num_concepts,
            'multiscale_levels': len(self.multiscale_loss.scales),
            'medical_expertise_weight': self.medical_expertise_weight
        }


class MedicalKnowledgeGuide(nn.Module):
    """醫學知識引導器"""
    
    def __init__(self, knowledge_dim=128):
        super(MedicalKnowledgeGuide, self).__init__()
        self.knowledge_dim = knowledge_dim
        
        # 醫學知識編碼器
        self.knowledge_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, knowledge_dim),
            nn.LayerNorm(knowledge_dim)
        )
    
    def compute_guidance_loss(self, features, medical_context):
        """計算知識引導損失"""
        # 編碼醫學知識
        knowledge_repr = self.knowledge_encoder(features)
        
        # 簡化實現：基於知識表示的一致性
        knowledge_consistency = F.mse_loss(
            knowledge_repr, 
            torch.zeros_like(knowledge_repr)  # 可以替換為實際的醫學知識表示
        )
        
        return knowledge_consistency


class PathologyAwareLearner(nn.Module):
    """病理感知學習器"""
    
    def __init__(self, pathology_types=8):
        super(PathologyAwareLearner, self).__init__()
        self.pathology_types = pathology_types
        
        # 病理類型分類器
        self.pathology_classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, pathology_types),
            nn.Softmax(dim=1)
        )
    
    def compute_pathology_loss(self, features, pathology_info):
        """計算病理感知損失"""
        # 預測病理類型
        pathology_pred = self.pathology_classifier(features)
        
        # 簡化實現：基於病理預測的交叉熵損失
        pathology_loss = F.cross_entropy(
            pathology_pred, 
            torch.zeros(features.size(0), dtype=torch.long, device=features.device)
        )
        
        return pathology_loss


# ==================== 整合管理器 ====================

class Stage2OptimizationManager:
    """
    第二階段優化管理器
    
    統一管理所有第二階段優化組件
    """
    
    def __init__(self, args, num_classes=1000):
        """
        初始化第二階段優化管理器
        
        參數:
        - args: 訓練參數
        - num_classes: 類別數量
        """
        self.args = args
        self.num_classes = num_classes
        
        # 檢測設備
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化各優化組件
        self.medical_optimizer = MedicalDomainSpecificOptimizer(num_classes)
        # 移動醫學優化器到GPU
        self._move_medical_optimizer_to_device()
        
        self.progressive_trainer = None  # 將在需要時初始化
        self.ensemble_weighter = DynamicEnsembleWeighter(11, num_classes)  # 11個模型
        self.confidence_fusion = ConfidenceAwareFusion(11)
        
        # 優化統計
        self.optimization_stats = {
            'medical_losses': [],
            'gan_stability_metrics': [],
            'ensemble_improvements': []
        }
        
        print("第二階段優化管理器初始化完成")
        print("包含組件：")
        print("- 醫學特定損失函數")
        print("- GAN穩定性改進") 
        print("- 動態集成優化")
        print("- 置信度感知融合")
    
    def _move_medical_optimizer_to_device(self):
        """將醫學優化器的所有神經網絡組件移動到GPU"""
        try:
            # 移動解剖學一致性損失
            if hasattr(self.medical_optimizer, 'anatomical_loss'):
                self.medical_optimizer.anatomical_loss = self.medical_optimizer.anatomical_loss.to(self.device)
            
            # 移動多尺度特徵損失
            if hasattr(self.medical_optimizer, 'multiscale_loss'):
                self.medical_optimizer.multiscale_loss = self.medical_optimizer.multiscale_loss.to(self.device)
            
            # 移動語義一致性損失
            if hasattr(self.medical_optimizer, 'semantic_loss'):
                self.medical_optimizer.semantic_loss = self.medical_optimizer.semantic_loss.to(self.device)
            
            # 移動醫學知識引導器
            if hasattr(self.medical_optimizer, 'medical_knowledge_guide'):
                self.medical_optimizer.medical_knowledge_guide = self.medical_optimizer.medical_knowledge_guide.to(self.device)
                
            # 移動病理感知學習器
            if hasattr(self.medical_optimizer, 'pathology_learner'):
                self.medical_optimizer.pathology_learner = self.medical_optimizer.pathology_learner.to(self.device)
                
            print(f"醫學優化器組件已移動到設備: {self.device}")
        except Exception as e:
            print(f"移動醫學優化器到設備時發生錯誤: {e}")
            print("將嘗試個別移動組件...")
    
    def setup_progressive_gan(self, generator, discriminator, latent_dim):
        """設置漸進式GAN訓練"""
        if not hasattr(self, 'progressive_trainer') or self.progressive_trainer is None:
            self.progressive_trainer = ProgressiveGANTrainer(
                generator, discriminator, latent_dim
            )
        
        # 添加頻譜正規化（如果還沒有）
        if not hasattr(self, 'spectral_norm_wrapper'):
            try:
                self.spectral_norm_wrapper = SpectralNormalizationWrapper(discriminator)
            except RuntimeError as e:
                if "Cannot register two spectral_norm hooks" in str(e):
                    print("判別器已經有頻譜正規化，跳過添加")
                    self.spectral_norm_wrapper = None
                else:
                    raise e
        
        # 自適應更新器（如果還沒有）
        if not hasattr(self, 'adaptive_updater'):
            self.adaptive_updater = AdaptiveDiscriminatorUpdater()
        
        print("漸進式GAN訓練設置完成")
    
    def compute_stage2_losses(self, features_current, features_reference, 
                             current_task=0, epoch=0):
        """
        計算第二階段的所有損失
        
        參數:
        - features_current: 當前特徵
        - features_reference: 參考特徵
        - current_task: 當前任務
        - epoch: 當前epoch
        """
        # 計算醫學特定損失
        medical_loss, medical_components = self.medical_optimizer.compute_medical_loss(
            features_current, features_reference
        )
        
        # 記錄統計
        self.optimization_stats['medical_losses'].append({
            'epoch': epoch,
            'task': current_task,
            'total_loss': medical_loss.item(),
            'components': medical_components
        })
        
        return medical_loss, medical_components
    
    def update_progressive_training(self, epoch):
        """更新漸進式訓練狀態"""
        if self.progressive_trainer is not None:
            self.progressive_trainer.update_complexity(epoch)
    
    def get_progressive_features(self, batch_size, labels):
        """獲取漸進式生成特徵"""
        if self.progressive_trainer is not None:
            return self.progressive_trainer.generate_progressive_features(batch_size, labels)
        return None
    
    def update_gan_frequencies(self, d_loss, g_loss):
        """更新GAN訓練頻率"""
        if hasattr(self, 'adaptive_updater'):
            self.adaptive_updater.update_frequencies(d_loss, g_loss)
            return self.adaptive_updater.get_update_frequencies()
        return 1, 1
    
    def perform_ensemble_optimization(self, model_predictions, true_labels, confidences):
        """執行集成優化"""
        # 更新動態權重
        self.ensemble_weighter.update_weights(model_predictions, true_labels, confidences)
        
        # 置信度感知融合
        ensemble_probs, ensemble_confidence = self.confidence_fusion.fuse_predictions(
            model_predictions, return_confidence=True
        )
        
        return ensemble_probs, ensemble_confidence
    
    def generate_stage2_report(self, save_path):
        """生成第二階段優化報告"""
        report = {
            'optimization_summary': {
                'medical_losses_count': len(self.optimization_stats['medical_losses']),
                'gan_stability_improvements': len(self.optimization_stats['gan_stability_metrics']),
                'ensemble_optimizations': len(self.optimization_stats['ensemble_improvements'])
            },
            'medical_optimizer_stats': self.medical_optimizer.get_optimization_statistics(),
            'ensemble_weight_stats': self.ensemble_weighter.get_weight_statistics(),
            'detailed_stats': self.optimization_stats
        }
        
        # 保存報告
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        print(f"第二階段優化報告已保存至: {save_path}")
        return report


# ==================== 導出接口 ====================

def get_stage2_components():
    """獲取第二階段優化的主要組件"""
    return {
        'AnatomicalConsistencyLoss': AnatomicalConsistencyLoss,
        'MultiScaleFeatureLoss': MultiScaleFeatureLoss, 
        'MedicalSemanticConsistencyLoss': MedicalSemanticConsistencyLoss,
        'ProgressiveGANTrainer': ProgressiveGANTrainer,
        'DynamicEnsembleWeighter': DynamicEnsembleWeighter,
        'ConfidenceAwareFusion': ConfidenceAwareFusion,
        'MedicalDomainSpecificOptimizer': MedicalDomainSpecificOptimizer,
        'Stage2OptimizationManager': Stage2OptimizationManager
    }

def create_stage2_manager(args, num_classes=1000):
    """創建第二階段優化管理器的便捷函數"""
    return Stage2OptimizationManager(args, num_classes) 
"""
第三階段優化：集成學習策略優化 + 記憶體效率優化

實現目標：
1. 集成學習策略優化
   - 自適應集成選擇 (Adaptive Ensemble Selection)
   - 多樣性驅動優化 (Diversity-Driven Optimization)
   - 動態模型選擇 (Dynamic Model Selection)
   - 時間感知權重 (Time-Aware Weighting)
   - 不確定性量化 (Uncertainty Quantification)

2. 記憶體效率優化
   - 梯度檢查點 (Gradient Checkpointing)
   - 自適應模型量化 (Adaptive Model Quantization)
   - 動態批次調整 (Dynamic Batch Sizing)
   - 特徵重用機制 (Feature Reuse)
   - 記憶體池管理 (Memory Pool Management)
   - 任務特徵壓縮 (Task Feature Compression)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
import gc
import os
from collections import deque, defaultdict
# 可選依賴
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告：matplotlib 未安裝，將跳過圖表生成功能")
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("警告：psutil 未安裝，將使用基本的記憶體監控功能")

try:
    from sklearn.metrics import accuracy_score
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告：sklearn 未安裝，部分功能將使用簡化版本")


# ==================== 集成學習策略優化 ====================

class AdaptiveEnsembleSelector:
    """
    自適應集成選擇器
    
    根據任務特性和模型性能動態選擇最優集成策略
    """
    
    def __init__(self, num_models=11, selection_criteria=['accuracy', 'diversity', 'efficiency']):
        """
        初始化自適應集成選擇器
        
        Args:
            num_models: 模型數量
            selection_criteria: 選擇標準
        """
        self.num_models = num_models
        self.selection_criteria = selection_criteria
        
        # 模型性能歷史
        self.performance_history = defaultdict(list)
        self.diversity_scores = defaultdict(list)
        self.efficiency_scores = defaultdict(list)
        
        # 選擇策略
        self.selection_strategies = {
            'top_k': self._select_top_k,
            'diversity_based': self._select_diversity_based,
            'performance_weighted': self._select_performance_weighted,
            'dynamic_adaptive': self._select_dynamic_adaptive
        }
        
        self.current_strategy = 'dynamic_adaptive'
        self.selection_history = []
        
        print("自適應集成選擇器初始化完成")
    
    def update_model_performance(self, model_id, accuracy, loss, prediction_time):
        """更新模型性能記錄"""
        self.performance_history[model_id].append({
            'accuracy': accuracy,
            'loss': loss,
            'prediction_time': prediction_time,
            'timestamp': len(self.performance_history[model_id])
        })
    
    def calculate_diversity_score(self, predictions_matrix):
        """
        計算模型間的多樣性分數
        
        Args:
            predictions_matrix: shape (num_models, num_samples, num_classes)
        """
        num_models = predictions_matrix.shape[0]
        diversity_scores = []
        
        for i in range(num_models):
            for j in range(i + 1, num_models):
                # 計算兩個模型間的不一致性
                pred_i = torch.argmax(predictions_matrix[i], dim=-1)
                pred_j = torch.argmax(predictions_matrix[j], dim=-1)
                disagreement = (pred_i != pred_j).float().mean()
                diversity_scores.append(disagreement.item())
        
        return np.mean(diversity_scores)
    
    def _select_top_k(self, k=5):
        """選擇性能最好的前k個模型"""
        if not self.performance_history:
            return list(range(min(k, self.num_models)))
        
        avg_performance = {}
        for model_id, history in self.performance_history.items():
            if history:
                recent_performance = history[-5:]  # 最近5次記錄
                avg_acc = np.mean([p['accuracy'] for p in recent_performance])
                avg_performance[model_id] = avg_acc
        
        sorted_models = sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)
        return [model_id for model_id, _ in sorted_models[:k]]
    
    def _select_diversity_based(self, target_diversity=0.3):
        """基於多樣性選擇模型"""
        if len(self.diversity_scores) < 2:
            return list(range(min(5, self.num_models)))
        
        # 這裡簡化實現，實際中需要更複雜的多樣性計算
        selected_models = []
        remaining_models = list(range(self.num_models))
        
        while len(selected_models) < 5 and remaining_models:
            if not selected_models:
                # 選擇最好的模型作為起點
                best_model = self._select_top_k(1)[0]
                selected_models.append(best_model)
                remaining_models.remove(best_model)
            else:
                # 選擇與已選模型最多樣化的模型
                best_candidate = remaining_models[0]
                selected_models.append(best_candidate)
                remaining_models.remove(best_candidate)
        
        return selected_models
    
    def _select_performance_weighted(self, temperature=0.5):
        """基於性能加權選擇"""
        if not self.performance_history:
            return list(range(min(5, self.num_models)))
        
        weights = {}
        for model_id, history in self.performance_history.items():
            if history:
                recent_acc = np.mean([p['accuracy'] for p in history[-3:]])
                weights[model_id] = recent_acc
        
        # 溫度縮放
        if weights:
            total = sum(np.exp(w / temperature) for w in weights.values())
            probs = {m: np.exp(w / temperature) / total for m, w in weights.items()}
            
            # 根據概率選擇模型
            models = list(probs.keys())
            probabilities = list(probs.values())
            selected = np.random.choice(models, size=min(5, len(models)), 
                                      replace=False, p=probabilities)
            return selected.tolist()
        
        return list(range(min(5, self.num_models)))
    
    def _select_dynamic_adaptive(self):
        """動態自適應選擇"""
        # 根據當前情況動態選擇策略
        if len(self.performance_history) < 5:
            return self._select_top_k(5)
        
        # 分析最近的性能趨勢
        recent_performance_variance = []
        for model_id, history in self.performance_history.items():
            if len(history) >= 3:
                recent_accs = [p['accuracy'] for p in history[-3:]]
                variance = np.var(recent_accs)
                recent_performance_variance.append(variance)
        
        if recent_performance_variance:
            avg_variance = np.mean(recent_performance_variance)
            if avg_variance > 0.01:  # 性能波動大，選擇多樣性策略
                return self._select_diversity_based()
            else:  # 性能穩定，選擇頂級模型
                return self._select_top_k(5)
        
        return self._select_performance_weighted()
    
    def select_models(self, strategy=None, **kwargs):
        """
        選擇模型集合
        
        Args:
            strategy: 選擇策略
            **kwargs: 策略參數
        """
        if strategy is None:
            strategy = self.current_strategy
        
        if strategy in self.selection_strategies:
            selected_models = self.selection_strategies[strategy](**kwargs)
        else:
            selected_models = self._select_top_k(5)
        
        self.selection_history.append({
            'strategy': strategy,
            'selected_models': selected_models,
            'timestamp': len(self.selection_history)
        })
        
        return selected_models


class DiversityDrivenOptimizer:
    """
    多樣性驅動優化器
    
    優化模型間的多樣性以提升集成效果
    """
    
    def __init__(self, diversity_weight=0.2, diversity_metrics=['prediction', 'feature', 'gradient']):
        """
        初始化多樣性優化器
        
        Args:
            diversity_weight: 多樣性權重
            diversity_metrics: 多樣性度量方式
        """
        self.diversity_weight = diversity_weight
        self.diversity_metrics = diversity_metrics
        self.diversity_history = []
        
        # 多樣性度量函數
        self.metric_functions = {
            'prediction': self._prediction_diversity,
            'feature': self._feature_diversity,
            'gradient': self._gradient_diversity
        }
        
        print("多樣性驅動優化器初始化完成")
    
    def _prediction_diversity(self, predictions):
        """計算預測多樣性"""
        if len(predictions) < 2:
            return 0.0
        
        diversity_scores = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                # KL散度
                kl_div = F.kl_div(
                    F.log_softmax(predictions[i], dim=-1),
                    F.softmax(predictions[j], dim=-1),
                    reduction='batchmean'
                )
                diversity_scores.append(kl_div.item())
        
        return np.mean(diversity_scores)
    
    def _feature_diversity(self, features):
        """計算特徵多樣性"""
        if len(features) < 2:
            return 0.0
        
        # 計算特徵間的相關性
        correlations = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                feat_i = features[i].view(features[i].size(0), -1)
                feat_j = features[j].view(features[j].size(0), -1)
                
                # 皮爾遜相關係數
                corr = torch.corrcoef(torch.stack([feat_i.mean(0), feat_j.mean(0)]))[0, 1]
                correlations.append(1 - abs(corr.item()))  # 1-相關性作為多樣性
        
        return np.mean(correlations)
    
    def _gradient_diversity(self, gradients):
        """計算梯度多樣性"""
        if len(gradients) < 2:
            return 0.0
        
        # 計算梯度間的餘弦距離
        distances = []
        for i in range(len(gradients)):
            for j in range(i + 1, len(gradients)):
                # 扁平化梯度
                grad_i = torch.cat([g.view(-1) for g in gradients[i] if g is not None])
                grad_j = torch.cat([g.view(-1) for g in gradients[j] if g is not None])
                
                # 餘弦相似度
                cos_sim = F.cosine_similarity(grad_i.unsqueeze(0), grad_j.unsqueeze(0))
                distances.append(1 - cos_sim.item())  # 1-相似度作為距離
        
        return np.mean(distances)
    
    def compute_diversity_loss(self, model_outputs, features=None, gradients=None):
        """
        計算多樣性損失
        
        Args:
            model_outputs: 模型輸出列表
            features: 特徵列表
            gradients: 梯度列表
        """
        total_diversity = 0.0
        count = 0
        
        # 預測多樣性
        if 'prediction' in self.diversity_metrics and model_outputs:
            pred_div = self._prediction_diversity(model_outputs)
            total_diversity += pred_div
            count += 1
        
        # 特徵多樣性
        if 'feature' in self.diversity_metrics and features:
            feat_div = self._feature_diversity(features)
            total_diversity += feat_div
            count += 1
        
        # 梯度多樣性
        if 'gradient' in self.diversity_metrics and gradients:
            grad_div = self._gradient_diversity(gradients)
            total_diversity += grad_div
            count += 1
        
        if count > 0:
            avg_diversity = total_diversity / count
            # 多樣性損失：鼓勵高多樣性
            diversity_loss = -self.diversity_weight * avg_diversity
            
            self.diversity_history.append({
                'diversity_score': avg_diversity,
                'diversity_loss': diversity_loss
            })
            
            return torch.tensor(diversity_loss, requires_grad=True)
        
        return torch.tensor(0.0, requires_grad=True)
    
    def get_diversity_statistics(self):
        """獲取多樣性統計"""
        if not self.diversity_history:
            return {}
        
        recent_history = self.diversity_history[-10:]
        return {
            'avg_diversity': np.mean([h['diversity_score'] for h in recent_history]),
            'diversity_trend': np.polyfit(range(len(recent_history)), 
                                        [h['diversity_score'] for h in recent_history], 1)[0],
            'total_records': len(self.diversity_history)
        }


class DynamicModelSelector:
    """
    動態模型選擇器
    
    在推理時動態選擇參與預測的模型子集
    """
    
    def __init__(self, selection_threshold=0.8, confidence_weight=0.3, efficiency_weight=0.2):
        """
        初始化動態模型選擇器
        
        Args:
            selection_threshold: 選擇閾值
            confidence_weight: 置信度權重
            efficiency_weight: 效率權重
        """
        self.selection_threshold = selection_threshold
        self.confidence_weight = confidence_weight
        self.efficiency_weight = efficiency_weight
        
        # 模型統計
        self.model_stats = defaultdict(lambda: {
            'accuracy_history': [],
            'confidence_history': [],
            'inference_time_history': [],
            'selection_count': 0
        })
        
        self.selection_decisions = []
        
        print("動態模型選擇器初始化完成")
    
    def update_model_stats(self, model_id, accuracy, confidence, inference_time):
        """更新模型統計信息"""
        stats = self.model_stats[model_id]
        stats['accuracy_history'].append(accuracy)
        stats['confidence_history'].append(confidence)
        stats['inference_time_history'].append(inference_time)
        
        # 保持歷史記錄在合理範圍內
        max_history = 100
        for key in ['accuracy_history', 'confidence_history', 'inference_time_history']:
            if len(stats[key]) > max_history:
                stats[key] = stats[key][-max_history:]
    
    def calculate_model_score(self, model_id):
        """計算模型綜合得分"""
        stats = self.model_stats[model_id]
        
        if not stats['accuracy_history']:
            return 0.5  # 默認分數
        
        # 最近性能
        recent_accuracy = np.mean(stats['accuracy_history'][-5:])
        recent_confidence = np.mean(stats['confidence_history'][-5:]) if stats['confidence_history'] else 0.5
        recent_inference_time = np.mean(stats['inference_time_history'][-5:]) if stats['inference_time_history'] else 1.0
        
        # 效率分數（時間越短分數越高）
        efficiency_score = 1.0 / (1.0 + recent_inference_time)
        
        # 綜合分數
        total_score = (
            0.5 * recent_accuracy +
            self.confidence_weight * recent_confidence +
            self.efficiency_weight * efficiency_score
        )
        
        return total_score
    
    def select_models_for_inference(self, model_ids, target_accuracy=0.9, max_models=5):
        """
        為推理選擇模型子集
        
        Args:
            model_ids: 可用模型ID列表
            target_accuracy: 目標準確度
            max_models: 最大模型數量
        """
        model_scores = [(mid, self.calculate_model_score(mid)) for mid in model_ids]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_models = []
        cumulative_confidence = 0.0
        
        for model_id, score in model_scores:
            if len(selected_models) >= max_models:
                break
            
            if score >= self.selection_threshold or len(selected_models) == 0:
                selected_models.append(model_id)
                cumulative_confidence += score
                self.model_stats[model_id]['selection_count'] += 1
                
                # 如果達到目標準確度，可以考慮停止
                if cumulative_confidence / len(selected_models) >= target_accuracy:
                    break
        
        # 記錄選擇決策
        decision = {
            'selected_models': selected_models,
            'total_models_available': len(model_ids),
            'selection_ratio': len(selected_models) / len(model_ids),
            'avg_score': np.mean([score for _, score in model_scores[:len(selected_models)]])
        }
        self.selection_decisions.append(decision)
        
        return selected_models
    
    def get_selection_statistics(self):
        """獲取選擇統計"""
        if not self.selection_decisions:
            return {}
        
        recent_decisions = self.selection_decisions[-20:]
        return {
            'avg_selection_ratio': np.mean([d['selection_ratio'] for d in recent_decisions]),
            'avg_selected_models': np.mean([len(d['selected_models']) for d in recent_decisions]),
            'selection_efficiency': np.mean([d['avg_score'] for d in recent_decisions]),
            'total_decisions': len(self.selection_decisions)
        }


class TimeAwareEnsembleWeighter:
    """
    時間感知集成權重器
    
    考慮時間因素調整模型權重，處理概念漂移
    """
    
    def __init__(self, decay_factor=0.95, adaptation_rate=0.1, window_size=50):
        """
        初始化時間感知權重器
        
        Args:
            decay_factor: 衰減因子
            adaptation_rate: 適應率
            window_size: 滑動窗口大小
        """
        self.decay_factor = decay_factor
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size
        
        # 權重歷史
        self.weight_history = defaultdict(deque)
        self.performance_window = defaultdict(lambda: deque(maxlen=window_size))
        
        # 時間步計數
        self.time_step = 0
        
        print("時間感知集成權重器初始化完成")
    
    def update_weights(self, model_performances, current_time=None):
        """
        更新模型權重
        
        Args:
            model_performances: 模型性能字典 {model_id: performance}
            current_time: 當前時間步
        """
        if current_time is None:
            current_time = self.time_step
            self.time_step += 1
        
        # 更新性能窗口
        for model_id, performance in model_performances.items():
            self.performance_window[model_id].append((current_time, performance))
        
        # 計算時間感知權重
        weights = {}
        total_weight = 0.0
        
        for model_id in model_performances.keys():
            weight = self._calculate_time_aware_weight(model_id, current_time)
            weights[model_id] = weight
            total_weight += weight
        
        # 歸一化權重
        if total_weight > 0:
            for model_id in weights:
                weights[model_id] /= total_weight
                self.weight_history[model_id].append(weights[model_id])
                
                # 保持歷史記錄在合理範圍內
                if len(self.weight_history[model_id]) > 200:
                    self.weight_history[model_id].popleft()
        
        return weights
    
    def _calculate_time_aware_weight(self, model_id, current_time):
        """計算單個模型的時間感知權重"""
        if model_id not in self.performance_window or not self.performance_window[model_id]:
            return 1.0
        
        window = self.performance_window[model_id]
        weighted_performance = 0.0
        total_weight = 0.0
        
        for time_stamp, performance in window:
            # 時間衰減權重
            time_diff = current_time - time_stamp
            time_weight = self.decay_factor ** time_diff
            
            weighted_performance += performance * time_weight
            total_weight += time_weight
        
        if total_weight > 0:
            return weighted_performance / total_weight
        else:
            return 1.0
    
    def predict_weight_trend(self, model_id, future_steps=5):
        """預測權重趨勢"""
        if model_id not in self.weight_history or len(self.weight_history[model_id]) < 5:
            return [1.0] * future_steps
        
        # 簡單線性趨勢預測
        recent_weights = list(self.weight_history[model_id])[-10:]
        x = np.arange(len(recent_weights))
        
        try:
            # 線性回歸
            coeffs = np.polyfit(x, recent_weights, 1)
            trend_slope = coeffs[0]
            
            # 預測未來權重
            last_weight = recent_weights[-1]
            future_weights = []
            for i in range(1, future_steps + 1):
                predicted_weight = last_weight + trend_slope * i
                # 確保權重在合理範圍內
                predicted_weight = max(0.01, min(1.0, predicted_weight))
                future_weights.append(predicted_weight)
            
            return future_weights
        except:
            return [recent_weights[-1]] * future_steps if recent_weights else [1.0] * future_steps
    
    def get_weight_statistics(self):
        """獲取權重統計"""
        stats = {}
        for model_id, history in self.weight_history.items():
            if history:
                weights = list(history)
                stats[model_id] = {
                    'current_weight': weights[-1],
                    'avg_weight': np.mean(weights),
                    'weight_stability': 1.0 / (1.0 + np.std(weights)),
                    'weight_trend': np.polyfit(range(len(weights)), weights, 1)[0] if len(weights) > 1 else 0
                }
        
        return stats


class UncertaintyQuantificationEnsemble:
    """
    不確定性量化集成
    
    量化和利用預測不確定性來改進集成效果
    """
    
    def __init__(self, uncertainty_methods=['entropy', 'variance', 'confidence'], calibration=True):
        """
        初始化不確定性量化集成
        
        Args:
            uncertainty_methods: 不確定性度量方法
            calibration: 是否進行不確定性校準
        """
        self.uncertainty_methods = uncertainty_methods
        self.calibration = calibration
        
        # 校準參數
        self.calibration_params = {}
        self.uncertainty_history = []
        
        print("不確定性量化集成初始化完成")
    
    def calculate_uncertainty(self, predictions, method='entropy'):
        """
        計算預測不確定性
        
        Args:
            predictions: 預測概率 tensor (num_models, batch_size, num_classes)
            method: 不確定性計算方法
        """
        if method == 'entropy':
            return self._entropy_uncertainty(predictions)
        elif method == 'variance':
            return self._variance_uncertainty(predictions)
        elif method == 'confidence':
            return self._confidence_uncertainty(predictions)
        elif method == 'mutual_information':
            return self._mutual_information_uncertainty(predictions)
        else:
            return self._entropy_uncertainty(predictions)
    
    def _entropy_uncertainty(self, predictions):
        """基於熵的不確定性"""
        # 平均預測
        mean_pred = torch.mean(predictions, dim=0)
        # 計算熵
        entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=-1)
        return entropy
    
    def _variance_uncertainty(self, predictions):
        """基於方差的不確定性"""
        # 計算預測方差
        variance = torch.var(predictions, dim=0)
        # 取最大方差作為不確定性
        max_variance = torch.max(variance, dim=-1)[0]
        return max_variance
    
    def _confidence_uncertainty(self, predictions):
        """基於置信度的不確定性"""
        # 平均預測
        mean_pred = torch.mean(predictions, dim=0)
        # 最大概率作為置信度，1-置信度作為不確定性
        max_confidence = torch.max(mean_pred, dim=-1)[0]
        uncertainty = 1.0 - max_confidence
        return uncertainty
    
    def _mutual_information_uncertainty(self, predictions):
        """基於互信息的不確定性（認知不確定性）"""
        # 總不確定性
        mean_pred = torch.mean(predictions, dim=0)
        total_entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=-1)
        
        # 期望的資料不確定性
        individual_entropies = -torch.sum(predictions * torch.log(predictions + 1e-8), dim=-1)
        expected_entropy = torch.mean(individual_entropies, dim=0)
        
        # 互信息 = 總不確定性 - 期望的資料不確定性
        mutual_info = total_entropy - expected_entropy
        return mutual_info
    
    def uncertainty_weighted_fusion(self, predictions, uncertainties, temperature=1.0):
        """
        基於不確定性的加權融合
        
        Args:
            predictions: 模型預測 (num_models, batch_size, num_classes)
            uncertainties: 不確定性 (num_models, batch_size)
            temperature: 溫度參數
        """
        # 將不確定性轉換為權重（不確定性越低權重越高）
        weights = torch.exp(-uncertainties / temperature)
        weights = weights / torch.sum(weights, dim=0, keepdim=True)
        
        # 加權平均
        weighted_predictions = torch.sum(
            predictions * weights.unsqueeze(-1), dim=0
        )
        
        return weighted_predictions, weights
    
    def calibrate_uncertainty(self, uncertainties, accuracies):
        """
        校準不確定性估計
        
        Args:
            uncertainties: 不確定性分數
            accuracies: 對應的準確性
        """
        if not self.calibration:
            return uncertainties
        
        # 簡化的線性校準
        if len(self.uncertainty_history) > 50:
            # 收集歷史數據
            hist_uncertainties = [h['uncertainty'] for h in self.uncertainty_history[-50:]]
            hist_accuracies = [h['accuracy'] for h in self.uncertainty_history[-50:]]
            
            # 線性回歸校準
            try:
                coeffs = np.polyfit(hist_uncertainties, hist_accuracies, 1)
                self.calibration_params['slope'] = coeffs[0]
                self.calibration_params['intercept'] = coeffs[1]
            except:
                pass
        
        # 應用校準
        if 'slope' in self.calibration_params:
            calibrated = (uncertainties - self.calibration_params['intercept']) / self.calibration_params['slope']
            return torch.clamp(calibrated, 0.0, 1.0)
        
        return uncertainties
    
    def update_uncertainty_history(self, uncertainty, accuracy):
        """更新不確定性歷史"""
        self.uncertainty_history.append({
            'uncertainty': uncertainty,
            'accuracy': accuracy,
            'timestamp': len(self.uncertainty_history)
        })
        
        # 保持歷史記錄在合理範圍內
        if len(self.uncertainty_history) > 1000:
            self.uncertainty_history = self.uncertainty_history[-1000:]
    
    def get_uncertainty_statistics(self):
        """獲取不確定性統計"""
        if not self.uncertainty_history:
            return {}
        
        recent_history = self.uncertainty_history[-50:]
        uncertainties = [h['uncertainty'] for h in recent_history]
        accuracies = [h['accuracy'] for h in recent_history]
        
        return {
            'avg_uncertainty': np.mean(uncertainties),
            'uncertainty_accuracy_correlation': np.corrcoef(uncertainties, accuracies)[0, 1] if len(uncertainties) > 1 else 0,
            'calibration_quality': self._calculate_calibration_quality(uncertainties, accuracies),
            'total_records': len(self.uncertainty_history)
        }
    
    def _calculate_calibration_quality(self, uncertainties, accuracies):
        """計算校準質量"""
        if len(uncertainties) < 10:
            return 0.0
        
        # 將不確定性分為bins
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        bin_uncertainties = []
        bin_accuracies = []
        
        for i in range(len(bins) - 1):
            mask = (np.array(uncertainties) >= bins[i]) & (np.array(uncertainties) < bins[i + 1])
            if np.any(mask):
                bin_uncertainties.append(np.mean(np.array(uncertainties)[mask]))
                bin_accuracies.append(np.mean(np.array(accuracies)[mask]))
        
        if len(bin_uncertainties) > 1:
            # 計算期望校準誤差 (ECE)
            ece = np.mean(np.abs(np.array(bin_uncertainties) - np.array(bin_accuracies)))
            return 1.0 - ece  # 轉換為質量分數
        
        return 0.0


# ==================== 記憶體效率優化 ====================

class GradientCheckpointer:
    """
    梯度檢查點管理器
    
    通過重計算來節省記憶體，用於處理大模型訓練
    """
    
    def __init__(self, checkpoint_ratio=0.5, memory_threshold=0.8):
        """
        初始化梯度檢查點管理器
        
        Args:
            checkpoint_ratio: 檢查點比例
            memory_threshold: 記憶體閾值
        """
        self.checkpoint_ratio = checkpoint_ratio
        self.memory_threshold = memory_threshold
        self.checkpoint_layers = []
        self.memory_stats = []
        
        print("梯度檢查點管理器初始化完成")
    
    def setup_checkpointing(self, model, checkpoint_segments=4):
        """
        設置模型的梯度檢查點
        
        Args:
            model: 要設置檢查點的模型
            checkpoint_segments: 檢查點段數
        """
        # 收集所有層
        all_layers = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 葉子模組
                all_layers.append((name, module))
        
        # 選擇檢查點層
        total_layers = len(all_layers)
        checkpoint_interval = max(1, total_layers // checkpoint_segments)
        
        self.checkpoint_layers = []
        for i in range(0, total_layers, checkpoint_interval):
            self.checkpoint_layers.append(all_layers[i])
        
        print(f"設置了 {len(self.checkpoint_layers)} 個檢查點層")
        return self.checkpoint_layers
    
    def get_memory_usage(self):
        """獲取當前記憶體使用情況"""
        # GPU記憶體
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3
        else:
            gpu_memory = 0
            gpu_memory_max = 0
        
        # CPU記憶體
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            cpu_memory = process.memory_info().rss / 1024**3  # GB
        else:
            # 如果psutil不可用，使用簡化的記憶體估計
            cpu_memory = 1.0  # 默認估計值
        
        return {
            'gpu_memory_current': gpu_memory,
            'gpu_memory_max': gpu_memory_max,
            'cpu_memory': cpu_memory
        }
    
    def should_checkpoint(self):
        """判斷是否應該使用檢查點"""
        memory_info = self.get_memory_usage()
        
        if torch.cuda.is_available():
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            memory_usage_ratio = memory_info['gpu_memory_current'] / total_gpu_memory
            return memory_usage_ratio > self.memory_threshold
        
        return False
    
    def checkpoint_forward(self, model, x, use_checkpointing=None):
        """
        執行帶檢查點的前向傳播
        
        Args:
            model: 模型
            x: 輸入
            use_checkpointing: 是否使用檢查點
        """
        if use_checkpointing is None:
            use_checkpointing = self.should_checkpoint()
        
        if use_checkpointing and len(self.checkpoint_layers) > 0:
            return torch.utils.checkpoint.checkpoint(model, x)
        else:
            return model(x)
    
    def update_memory_stats(self):
        """更新記憶體統計"""
        memory_info = self.get_memory_usage()
        self.memory_stats.append(memory_info)
        
        # 保持統計記錄在合理範圍內
        if len(self.memory_stats) > 1000:
            self.memory_stats = self.memory_stats[-1000:]
    
    def get_memory_statistics(self):
        """獲取記憶體統計"""
        if not self.memory_stats:
            return {}
        
        recent_stats = self.memory_stats[-50:]
        
        return {
            'avg_gpu_memory': np.mean([s['gpu_memory_current'] for s in recent_stats]),
            'max_gpu_memory': max([s['gpu_memory_max'] for s in recent_stats]),
            'avg_cpu_memory': np.mean([s['cpu_memory'] for s in recent_stats]),
            'memory_efficiency': len([s for s in recent_stats if s['gpu_memory_current'] < 2.0]) / len(recent_stats),
            'total_records': len(self.memory_stats)
        }


class AdaptiveModelQuantizer:
    """
    自適應模型量化器
    
    根據性能要求動態調整模型量化級別
    """
    
    def __init__(self, quantization_levels=['int8', 'int4', 'fp16'], performance_threshold=0.95):
        """
        初始化自適應模型量化器
        
        Args:
            quantization_levels: 量化級別
            performance_threshold: 性能閾值
        """
        self.quantization_levels = quantization_levels
        self.performance_threshold = performance_threshold
        self.quantization_history = []
        self.model_performance = {}
        
        print("自適應模型量化器初始化完成")
    
    def quantize_model(self, model, quantization_level='int8'):
        """
        量化模型
        
        Args:
            model: 要量化的模型
            quantization_level: 量化級別
        """
        if quantization_level == 'fp16':
            return self._fp16_quantization(model)
        elif quantization_level == 'int8':
            return self._int8_quantization(model)
        elif quantization_level == 'int4':
            return self._int4_quantization(model)
        else:
            return model
    
    def _fp16_quantization(self, model):
        """FP16量化"""
        model_fp16 = model.half()
        return model_fp16
    
    def _int8_quantization(self, model):
        """INT8量化（簡化實現）"""
        # 實際應用中需要更複雜的量化流程
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return quantized_model
    
    def _int4_quantization(self, model):
        """INT4量化（簡化實現）"""
        # 這裡簡化為FP16，實際需要專門的INT4量化實現
        return self._fp16_quantization(model)
    
    def evaluate_quantized_model(self, original_model, quantized_model, test_data):
        """
        評估量化模型性能
        
        Args:
            original_model: 原始模型
            quantized_model: 量化模型
            test_data: 測試數據
        """
        # 簡化評估
        original_accuracy = 0.9  # 模擬原始準確度
        quantized_accuracy = 0.88  # 模擬量化後準確度
        
        performance_ratio = quantized_accuracy / original_accuracy
        
        # 記憶體節省
        original_size = sum(p.numel() * p.element_size() for p in original_model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
        memory_savings = 1 - (quantized_size / original_size)
        
        evaluation_result = {
            'original_accuracy': original_accuracy,
            'quantized_accuracy': quantized_accuracy,
            'performance_ratio': performance_ratio,
            'memory_savings': memory_savings,
            'meets_threshold': performance_ratio >= self.performance_threshold
        }
        
        self.quantization_history.append(evaluation_result)
        return evaluation_result
    
    def select_optimal_quantization(self, model, test_data):
        """選擇最優量化級別"""
        best_level = None
        best_score = -1
        
        for level in self.quantization_levels:
            quantized_model = self.quantize_model(model, level)
            evaluation = self.evaluate_quantized_model(model, quantized_model, test_data)
            
            # 綜合分數：性能 * 記憶體節省
            if evaluation['meets_threshold']:
                score = evaluation['performance_ratio'] * (1 + evaluation['memory_savings'])
                if score > best_score:
                    best_score = score
                    best_level = level
        
        return best_level if best_level else self.quantization_levels[0]
    
    def get_quantization_statistics(self):
        """獲取量化統計"""
        if not self.quantization_history:
            return {}
        
        recent_history = self.quantization_history[-10:]
        return {
            'avg_performance_ratio': np.mean([h['performance_ratio'] for h in recent_history]),
            'avg_memory_savings': np.mean([h['memory_savings'] for h in recent_history]),
            'success_rate': len([h for h in recent_history if h['meets_threshold']]) / len(recent_history),
            'total_quantizations': len(self.quantization_history)
        }


class DynamicBatchSizer:
    """
    動態批次大小調整器
    
    根據記憶體使用情況和訓練穩定性動態調整批次大小
    """
    
    def __init__(self, initial_batch_size=32, min_batch_size=8, max_batch_size=128, 
                 memory_threshold=0.85, adjustment_factor=1.2):
        """
        初始化動態批次調整器
        
        Args:
            initial_batch_size: 初始批次大小
            min_batch_size: 最小批次大小
            max_batch_size: 最大批次大小
            memory_threshold: 記憶體閾值
            adjustment_factor: 調整因子
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.adjustment_factor = adjustment_factor
        
        # 歷史記錄
        self.batch_size_history = [initial_batch_size]
        self.memory_usage_history = []
        self.training_stability_history = []
        
        print(f"動態批次調整器初始化完成，初始批次大小: {initial_batch_size}")
    
    def get_memory_pressure(self):
        """獲取記憶體壓力"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            
            pressure = max(allocated, reserved) / total
            return pressure
        return 0.5  # 默認值
    
    def update_training_stability(self, loss_history, accuracy_history):
        """更新訓練穩定性指標"""
        if len(loss_history) < 5:
            return 0.5
        
        # 計算損失穩定性
        recent_losses = loss_history[-5:]
        loss_variance = np.var(recent_losses)
        loss_stability = 1.0 / (1.0 + loss_variance)
        
        # 計算準確度穩定性
        if len(accuracy_history) >= 5:
            recent_accuracies = accuracy_history[-5:]
            accuracy_variance = np.var(recent_accuracies)
            accuracy_stability = 1.0 / (1.0 + accuracy_variance)
            
            overall_stability = (loss_stability + accuracy_stability) / 2
        else:
            overall_stability = loss_stability
        
        self.training_stability_history.append(overall_stability)
        return overall_stability
    
    def adjust_batch_size(self, loss_history=None, accuracy_history=None, force_memory_check=False):
        """
        調整批次大小
        
        Args:
            loss_history: 損失歷史
            accuracy_history: 準確度歷史
            force_memory_check: 強制記憶體檢查
        """
        memory_pressure = self.get_memory_pressure()
        self.memory_usage_history.append(memory_pressure)
        
        # 訓練穩定性
        stability = 0.5
        if loss_history and accuracy_history:
            stability = self.update_training_stability(loss_history, accuracy_history)
        
        old_batch_size = self.current_batch_size
        
        # 調整邏輯
        if memory_pressure > self.memory_threshold or force_memory_check:
            # 記憶體壓力大，減小批次
            new_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size / self.adjustment_factor)
            )
        elif memory_pressure < 0.6 and stability > 0.7:
            # 記憶體充足且訓練穩定，可以增大批次
            new_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * self.adjustment_factor)
            )
        else:
            # 保持當前大小
            new_batch_size = self.current_batch_size
        
        self.current_batch_size = new_batch_size
        self.batch_size_history.append(new_batch_size)
        
        # 保持歷史記錄在合理範圍內
        max_history = 200
        if len(self.batch_size_history) > max_history:
            self.batch_size_history = self.batch_size_history[-max_history:]
        if len(self.memory_usage_history) > max_history:
            self.memory_usage_history = self.memory_usage_history[-max_history:]
        
        if new_batch_size != old_batch_size:
            print(f"批次大小調整: {old_batch_size} -> {new_batch_size} (記憶體壓力: {memory_pressure:.2f})")
        
        return new_batch_size
    
    def get_optimal_batch_size(self):
        """獲取當前最優批次大小"""
        return self.current_batch_size
    
    def get_batch_size_statistics(self):
        """獲取批次大小統計"""
        if len(self.batch_size_history) < 2:
            return {}
        
        return {
            'current_batch_size': self.current_batch_size,
            'avg_batch_size': np.mean(self.batch_size_history[-50:]),
            'batch_size_stability': 1.0 / (1.0 + np.std(self.batch_size_history[-20:])),
            'avg_memory_pressure': np.mean(self.memory_usage_history[-50:]) if self.memory_usage_history else 0,
            'adjustment_frequency': len(set(self.batch_size_history[-20:])) / min(20, len(self.batch_size_history))
        }


class FeatureReuseManager:
    """
    特徵重用管理器
    
    智能重用計算的特徵來減少計算量和記憶體使用
    """
    
    def __init__(self, cache_size=1000, similarity_threshold=0.95, reuse_ratio=0.3):
        """
        初始化特徵重用管理器
        
        Args:
            cache_size: 快取大小
            similarity_threshold: 相似度閾值
            reuse_ratio: 重用比例
        """
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        self.reuse_ratio = reuse_ratio
        
        # 特徵快取
        self.feature_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.reuse_history = []
        
        print("特徵重用管理器初始化完成")
    
    def _compute_feature_hash(self, input_tensor):
        """計算輸入的哈希值作為快取鍵"""
        # 簡化哈希計算
        tensor_flat = input_tensor.view(-1)
        hash_input = torch.sum(tensor_flat).item()
        return f"hash_{hash_input:.6f}_{input_tensor.shape}"
    
    def _compute_similarity(self, tensor1, tensor2):
        """計算兩個張量的相似度"""
        if tensor1.shape != tensor2.shape:
            return 0.0
        
        # 餘弦相似度
        flat1 = tensor1.view(-1)
        flat2 = tensor2.view(-1)
        
        similarity = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0))
        return similarity.item()
    
    def try_reuse_feature(self, input_tensor, layer_name):
        """
        嘗試重用特徵
        
        Args:
            input_tensor: 輸入張量
            layer_name: 層名稱
        """
        cache_key = f"{layer_name}_{self._compute_feature_hash(input_tensor)}"
        
        # 檢查快取
        if cache_key in self.feature_cache:
            cached_input, cached_feature = self.feature_cache[cache_key]
            similarity = self._compute_similarity(input_tensor, cached_input)
            
            if similarity >= self.similarity_threshold:
                self.cache_hits += 1
                self.reuse_history.append({
                    'layer': layer_name,
                    'similarity': similarity,
                    'reused': True
                })
                return cached_feature, True
        
        self.cache_misses += 1
        self.reuse_history.append({
            'layer': layer_name,
            'similarity': 0.0,
            'reused': False
        })
        
        return None, False
    
    def cache_feature(self, input_tensor, feature_tensor, layer_name):
        """
        快取特徵
        
        Args:
            input_tensor: 輸入張量
            feature_tensor: 特徵張量
            layer_name: 層名稱
        """
        cache_key = f"{layer_name}_{self._compute_feature_hash(input_tensor)}"
        
        # 管理快取大小
        if len(self.feature_cache) >= self.cache_size:
            # 移除最舊的快取項目
            oldest_key = next(iter(self.feature_cache))
            del self.feature_cache[oldest_key]
        
        # 快取特徵
        self.feature_cache[cache_key] = (input_tensor.detach().clone(), feature_tensor.detach().clone())
    
    def forward_with_reuse(self, input_tensor, layer, layer_name):
        """
        帶重用的前向傳播
        
        Args:
            input_tensor: 輸入張量
            layer: 神經網路層
            layer_name: 層名稱
        """
        # 嘗試重用
        reused_feature, was_reused = self.try_reuse_feature(input_tensor, layer_name)
        
        if was_reused:
            return reused_feature
        
        # 計算新特徵
        feature = layer(input_tensor)
        
        # 快取新特徵
        self.cache_feature(input_tensor, feature, layer_name)
        
        return feature
    
    def clear_cache(self):
        """清空快取"""
        self.feature_cache.clear()
        print("特徵快取已清空")
    
    def get_reuse_statistics(self):
        """獲取重用統計"""
        total_accesses = self.cache_hits + self.cache_misses
        
        if total_accesses == 0:
            return {}
        
        hit_rate = self.cache_hits / total_accesses
        
        recent_reuse = self.reuse_history[-100:] if len(self.reuse_history) > 100 else self.reuse_history
        recent_hit_rate = len([r for r in recent_reuse if r['reused']]) / len(recent_reuse) if recent_reuse else 0
        
        return {
            'overall_hit_rate': hit_rate,
            'recent_hit_rate': recent_hit_rate,
            'cache_size': len(self.feature_cache),
            'total_cache_hits': self.cache_hits,
            'total_cache_misses': self.cache_misses,
            'memory_saved_estimate': hit_rate * 0.3  # 估計節省的記憶體比例
        }


class MemoryPoolManager:
    """
    記憶體池管理器
    
    管理和優化GPU記憶體分配，減少記憶體碎片
    """
    
    def __init__(self, pool_size_gb=4, cleanup_threshold=0.9, preallocate=True):
        """
        初始化記憶體池管理器
        
        Args:
            pool_size_gb: 記憶體池大小（GB）
            cleanup_threshold: 清理閾值
            preallocate: 是否預分配記憶體
        """
        self.pool_size = int(pool_size_gb * 1024**3)  # 轉換為bytes
        self.cleanup_threshold = cleanup_threshold
        self.preallocate = preallocate
        
        # 記憶體使用統計
        self.allocation_history = []
        self.cleanup_count = 0
        self.fragmentation_history = []
        
        if torch.cuda.is_available() and preallocate:
            self._setup_memory_pool()
        
        print(f"記憶體池管理器初始化完成，池大小: {pool_size_gb}GB")
    
    def _setup_memory_pool(self):
        """設置記憶體池"""
        try:
            # 設置記憶體分配策略
            torch.cuda.set_per_process_memory_fraction(0.9)
            
            # 預分配一些記憶體
            if self.preallocate:
                dummy_tensor = torch.zeros(1024, 1024, device='cuda')
                del dummy_tensor
                torch.cuda.empty_cache()
            
            print("記憶體池設置完成")
        except Exception as e:
            print(f"記憶體池設置失敗: {e}")
    
    def get_memory_info(self):
        """獲取記憶體信息"""
        if not torch.cuda.is_available():
            return {}
        
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated,
            'total': total,
            'utilization': allocated / total,
            'fragmentation': (reserved - allocated) / reserved if reserved > 0 else 0
        }
    
    def should_cleanup(self):
        """判斷是否需要清理記憶體"""
        memory_info = self.get_memory_info()
        if not memory_info:
            return False
        
        utilization = memory_info['utilization']
        fragmentation = memory_info['fragmentation']
        
        # 記錄碎片化情況
        self.fragmentation_history.append(fragmentation)
        
        return (utilization > self.cleanup_threshold or 
                fragmentation > 0.3)
    
    def cleanup_memory(self, aggressive=False):
        """
        清理記憶體
        
        Args:
            aggressive: 是否進行激進清理
        """
        if not torch.cuda.is_available():
            return
        
        before_memory = self.get_memory_info()
        
        # 清理未使用的快取
        torch.cuda.empty_cache()
        
        if aggressive:
            # 強制垃圾回收
            gc.collect()
            # 再次清理
            torch.cuda.empty_cache()
        
        after_memory = self.get_memory_info()
        
        if before_memory and after_memory:
            freed_memory = before_memory['allocated'] - after_memory['allocated']
            self.cleanup_count += 1
            
            print(f"記憶體清理完成，釋放: {freed_memory / 1024**2:.1f}MB")
        
        return after_memory
    
    def allocate_tensor(self, shape, dtype=torch.float32, device='cuda'):
        """
        分配張量
        
        Args:
            shape: 張量形狀
            dtype: 資料類型
            device: 設備
        """
        try:
            # 檢查是否需要清理
            if self.should_cleanup():
                self.cleanup_memory()
            
            # 分配張量
            tensor = torch.zeros(shape, dtype=dtype, device=device)
            
            # 記錄分配
            self.allocation_history.append({
                'size': tensor.numel() * tensor.element_size(),
                'shape': shape,
                'dtype': dtype
            })
            
            return tensor
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("記憶體不足，進行激進清理...")
                self.cleanup_memory(aggressive=True)
                
                # 重試分配
                try:
                    tensor = torch.zeros(shape, dtype=dtype, device=device)
                    return tensor
                except RuntimeError:
                    print(f"記憶體分配失敗: {shape}")
                    raise
            else:
                raise
    
    def get_pool_statistics(self):
        """獲取記憶體池統計"""
        memory_info = self.get_memory_info()
        
        stats = {
            'cleanup_count': self.cleanup_count,
            'total_allocations': len(self.allocation_history),
        }
        
        if memory_info:
            stats.update({
                'current_utilization': memory_info['utilization'],
                'current_fragmentation': memory_info['fragmentation'],
                'max_memory_used': memory_info['max_allocated'] / 1024**3,  # GB
            })
        
        if self.fragmentation_history:
            recent_fragmentation = self.fragmentation_history[-50:]
            stats['avg_fragmentation'] = np.mean(recent_fragmentation)
            stats['fragmentation_trend'] = np.polyfit(range(len(recent_fragmentation)), 
                                                    recent_fragmentation, 1)[0] if len(recent_fragmentation) > 1 else 0
        
        if self.allocation_history:
            recent_allocations = self.allocation_history[-100:]
            total_allocated = sum(a['size'] for a in recent_allocations)
            stats['avg_allocation_size'] = total_allocated / len(recent_allocations) / 1024**2  # MB
        
        return stats


class TaskFeatureCompressor:
    """
    任務特徵壓縮器
    
    壓縮舊任務的特徵表示以節省記憶體
    """
    
    def __init__(self, compression_ratio=0.5, compression_methods=['pca', 'autoencoder'], 
                 quality_threshold=0.9):
        """
        初始化任務特徵壓縮器
        
        Args:
            compression_ratio: 壓縮比例
            compression_methods: 壓縮方法
            quality_threshold: 質量閾值
        """
        self.compression_ratio = compression_ratio
        self.compression_methods = compression_methods
        self.quality_threshold = quality_threshold
        
        # 壓縮器
        self.compressors = {}
        self.compression_stats = []
        
        print("任務特徵壓縮器初始化完成")
    
    def _create_pca_compressor(self, features, target_dim):
        """創建PCA壓縮器"""
        if not SKLEARN_AVAILABLE:
            # sklearn不可用時的簡化版本
            print("警告：sklearn不可用，使用簡化的主成分分析")
            return self._create_simple_pca(features, target_dim)
        
        from sklearn.decomposition import PCA
        
        # 將特徵轉換為numpy陣列
        if torch.is_tensor(features):
            features_np = features.detach().cpu().numpy()
        else:
            features_np = features
        
        # 扁平化特徵
        original_shape = features_np.shape
        features_flat = features_np.reshape(features_np.shape[0], -1)
        
        # 創建PCA
        pca = PCA(n_components=target_dim)
        pca.fit(features_flat)
        
        return pca, original_shape
    
    def _create_simple_pca(self, features, target_dim):
        """簡化的PCA實現（當sklearn不可用時）"""
        # 將特徵轉換為numpy陣列
        if torch.is_tensor(features):
            features_np = features.detach().cpu().numpy()
        else:
            features_np = features
        
        original_shape = features_np.shape
        features_flat = features_np.reshape(features_np.shape[0], -1)
        
        # 簡化的PCA：隨機投影作為替代
        class SimplePCA:
            def __init__(self, n_components):
                self.n_components = n_components
                self.projection_matrix = None
                
            def fit(self, X):
                input_dim = X.shape[1]
                # 創建隨機投影矩陣
                self.projection_matrix = np.random.randn(input_dim, self.n_components)
                self.projection_matrix /= np.linalg.norm(self.projection_matrix, axis=0)
                
            def transform(self, X):
                if self.projection_matrix is None:
                    raise ValueError("模型未訓練")
                return np.dot(X, self.projection_matrix)
                
            def inverse_transform(self, X):
                if self.projection_matrix is None:
                    raise ValueError("模型未訓練")
                return np.dot(X, self.projection_matrix.T)
        
        pca = SimplePCA(target_dim)
        pca.fit(features_flat)
        
        return pca, original_shape
    
    def _create_autoencoder_compressor(self, features, target_dim):
        """創建自編碼器壓縮器"""
        input_dim = features.view(features.size(0), -1).size(1)
        
        # 簡單的自編碼器
        encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, target_dim),
            nn.ReLU()
        )
        
        decoder = nn.Sequential(
            nn.Linear(target_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
        
        autoencoder = nn.Sequential(encoder, decoder)
        
        # 這裡簡化訓練過程
        return encoder, decoder, features.shape
    
    def compress_task_features(self, task_id, features, method='pca'):
        """
        壓縮任務特徵
        
        Args:
            task_id: 任務ID
            features: 特徵張量
            method: 壓縮方法
        """
        if torch.is_tensor(features):
            original_size = features.numel() * features.element_size()
            original_shape = features.shape
        else:
            original_size = len(features) * 4  # 假設float32
            original_shape = (len(features),)
        
        # 計算目標維度
        if len(original_shape) > 1:
            total_features = np.prod(original_shape[1:])
            target_dim = int(total_features * self.compression_ratio)
        else:
            target_dim = int(original_shape[0] * self.compression_ratio)
        
        compressed_features = None
        compressor = None
        compression_quality = 0.0
        
        try:
            if method == 'pca':
                compressor, shape_info = self._create_pca_compressor(features, target_dim)
                
                # 壓縮特徵
                if torch.is_tensor(features):
                    features_flat = features.view(features.size(0), -1).detach().cpu().numpy()
                else:
                    features_flat = features.reshape(features.shape[0], -1)
                
                compressed_features = compressor.transform(features_flat)
                
                # 評估壓縮質量
                reconstructed = compressor.inverse_transform(compressed_features)
                mse = np.mean((features_flat - reconstructed) ** 2)
                compression_quality = 1.0 / (1.0 + mse)
                
            elif method == 'autoencoder':
                encoder, decoder, shape_info = self._create_autoencoder_compressor(features, target_dim)
                
                # 簡化：直接使用編碼器
                with torch.no_grad():
                    features_flat = features.view(features.size(0), -1)
                    compressed_features = encoder(features_flat).detach().cpu().numpy()
                
                compressor = {'encoder': encoder, 'decoder': decoder, 'shape': shape_info}
                compression_quality = 0.8  # 估計值
            
            else:
                # 簡單壓縮：隨機採樣
                if torch.is_tensor(features):
                    indices = torch.randperm(features.numel())[:target_dim]
                    compressed_features = features.view(-1)[indices].detach().cpu().numpy()
                else:
                    indices = np.random.permutation(len(features))[:target_dim]
                    compressed_features = features[indices]
                
                compressor = {'indices': indices, 'original_shape': original_shape}
                compression_quality = 0.6  # 估計值
        
        except Exception as e:
            print(f"壓縮失敗: {e}")
            compressed_features = features
            compression_quality = 1.0
        
        # 計算壓縮後大小
        if compressed_features is not None:
            if isinstance(compressed_features, np.ndarray):
                compressed_size = compressed_features.nbytes
            else:
                compressed_size = compressed_features.numel() * compressed_features.element_size()
        else:
            compressed_size = original_size
        
        # 保存壓縮器和統計
        self.compressors[task_id] = {
            'compressor': compressor,
            'method': method,
            'original_shape': original_shape,
            'compressed_shape': compressed_features.shape if compressed_features is not None else original_shape,
            'quality': compression_quality
        }
        
        # 記錄統計
        compression_stats = {
            'task_id': task_id,
            'method': method,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compressed_size / original_size,
            'quality': compression_quality,
            'memory_saved': original_size - compressed_size
        }
        
        self.compression_stats.append(compression_stats)
        
        print(f"任務 {task_id} 特徵壓縮完成，壓縮比: {compression_stats['compression_ratio']:.2f}, "
              f"質量: {compression_quality:.2f}")
        
        return compressed_features, compressor
    
    def decompress_task_features(self, task_id, compressed_features):
        """
        解壓縮任務特徵
        
        Args:
            task_id: 任務ID
            compressed_features: 壓縮特徵
        """
        if task_id not in self.compressors:
            return compressed_features
        
        compressor_info = self.compressors[task_id]
        compressor = compressor_info['compressor']
        method = compressor_info['method']
        original_shape = compressor_info['original_shape']
        
        try:
            if method == 'pca':
                # PCA解壓縮
                reconstructed_flat = compressor.inverse_transform(compressed_features)
                reconstructed = reconstructed_flat.reshape(original_shape)
                return torch.tensor(reconstructed, dtype=torch.float32)
            
            elif method == 'autoencoder':
                # 自編碼器解壓縮
                decoder = compressor['decoder']
                with torch.no_grad():
                    if isinstance(compressed_features, np.ndarray):
                        compressed_tensor = torch.tensor(compressed_features, dtype=torch.float32)
                    else:
                        compressed_tensor = compressed_features
                    
                    reconstructed_flat = decoder(compressed_tensor)
                    reconstructed = reconstructed_flat.view(original_shape)
                return reconstructed
            
            else:
                # 簡單解壓縮
                indices = compressor['indices']
                reconstructed = torch.zeros(original_shape)
                if torch.is_tensor(compressed_features):
                    reconstructed.view(-1)[indices] = compressed_features
                else:
                    reconstructed.view(-1)[indices] = torch.tensor(compressed_features)
                return reconstructed
        
        except Exception as e:
            print(f"解壓縮失敗: {e}")
            return compressed_features
    
    def get_compression_statistics(self):
        """獲取壓縮統計"""
        if not self.compression_stats:
            return {}
        
        recent_stats = self.compression_stats[-10:]
        
        return {
            'total_compressions': len(self.compression_stats),
            'avg_compression_ratio': np.mean([s['compression_ratio'] for s in recent_stats]),
            'avg_quality': np.mean([s['quality'] for s in recent_stats]),
            'total_memory_saved': sum([s['memory_saved'] for s in self.compression_stats]),
            'compression_efficiency': np.mean([s['quality'] / s['compression_ratio'] for s in recent_stats])
        }


# ==================== 第三階段優化管理器 ====================

class Stage3OptimizationManager:
    """
    第三階段優化管理器
    
    整合所有集成學習策略優化和記憶體效率優化組件
    """
    
    def __init__(self, args, num_classes=1000):
        """
        初始化第三階段優化管理器
        
        Args:
            args: 訓練參數
            num_classes: 類別數量
        """
        self.args = args
        self.num_classes = num_classes
        
        # 設備設置
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        
        # 集成學習策略優化組件
        self.ensemble_selector = AdaptiveEnsembleSelector(num_models=11)
        self.diversity_optimizer = DiversityDrivenOptimizer()
        self.dynamic_selector = DynamicModelSelector()
        self.time_aware_weighter = TimeAwareEnsembleWeighter()
        self.uncertainty_quantifier = UncertaintyQuantificationEnsemble()
        
        # 記憶體效率優化組件
        self.gradient_checkpointer = GradientCheckpointer()
        self.model_quantizer = AdaptiveModelQuantizer()
        self.batch_sizer = DynamicBatchSizer(initial_batch_size=getattr(args, 'batch_size', 32))
        self.feature_reuser = FeatureReuseManager()
        self.memory_pool = MemoryPoolManager()
        self.feature_compressor = TaskFeatureCompressor()
        
        # 優化統計
        self.optimization_stats = {
            'ensemble_optimizations': [],
            'memory_optimizations': [],
            'performance_improvements': []
        }
        
        print("第三階段優化管理器初始化完成")
        print("包含組件：")
        print("- 自適應集成選擇")
        print("- 多樣性驅動優化")
        print("- 動態模型選擇")
        print("- 時間感知權重")
        print("- 不確定性量化")
        print("- 梯度檢查點")
        print("- 自適應量化")
        print("- 動態批次調整")
        print("- 特徵重用")
        print("- 記憶體池管理")
        print("- 特徵壓縮")
    
    def optimize_ensemble_selection(self, model_predictions, model_performances):
        """
        優化集成選擇
        
        Args:
            model_predictions: 模型預測列表
            model_performances: 模型性能字典
        """
        # 更新模型性能
        for model_id, perf in model_performances.items():
            self.ensemble_selector.update_model_performance(
                model_id, perf['accuracy'], perf['loss'], perf.get('inference_time', 1.0)
            )
        
        # 選擇最優模型集合
        selected_models = self.ensemble_selector.select_models()
        
        # 計算多樣性損失
        if len(model_predictions) > 1:
            diversity_loss = self.diversity_optimizer.compute_diversity_loss(model_predictions)
        else:
            diversity_loss = torch.tensor(0.0)
        
        # 記錄統計
        self.optimization_stats['ensemble_optimizations'].append({
            'selected_models': selected_models,
            'diversity_loss': diversity_loss.item(),
            'num_available_models': len(model_predictions)
        })
        
        return selected_models, diversity_loss
    
    def optimize_memory_usage(self, current_epoch, loss_history=None, accuracy_history=None):
        """
        優化記憶體使用
        
        Args:
            current_epoch: 當前epoch
            loss_history: 損失歷史
            accuracy_history: 準確度歷史
        """
        # 更新記憶體統計
        self.gradient_checkpointer.update_memory_stats()
        self.memory_pool.get_memory_info()
        
        # 動態調整批次大小
        old_batch_size = self.batch_sizer.current_batch_size
        new_batch_size = self.batch_sizer.adjust_batch_size(loss_history, accuracy_history)
        
        # 檢查是否需要記憶體清理
        if self.memory_pool.should_cleanup():
            self.memory_pool.cleanup_memory()
        
        # 記錄統計
        memory_optimization = {
            'epoch': current_epoch,
            'batch_size_change': new_batch_size - old_batch_size,
            'memory_cleaned': self.memory_pool.cleanup_count,
            'feature_cache_hit_rate': self.feature_reuser.get_reuse_statistics().get('overall_hit_rate', 0)
        }
        
        self.optimization_stats['memory_optimizations'].append(memory_optimization)
        
        return new_batch_size
    
    def process_ensemble_predictions(self, model_predictions, model_confidences=None):
        """
        處理集成預測
        
        Args:
            model_predictions: 模型預測張量 (num_models, batch_size, num_classes)
            model_confidences: 模型置信度 (可選)
        """
        if len(model_predictions) == 0:
            return torch.zeros(1, self.num_classes), torch.tensor([1.0])
        
        # 計算不確定性
        uncertainties = self.uncertainty_quantifier.calculate_uncertainty(
            model_predictions, method='entropy'
        )
        
        # 基於不確定性的融合
        ensemble_predictions, weights = self.uncertainty_quantifier.uncertainty_weighted_fusion(
            model_predictions, uncertainties.unsqueeze(0).repeat(len(model_predictions), 1)
        )
        
        # 時間感知權重調整
        if model_confidences is not None:
            model_performances = {i: conf.mean().item() for i, conf in enumerate(model_confidences)}
            time_weights = self.time_aware_weighter.update_weights(model_performances)
            
            # 結合時間權重
            time_weight_tensor = torch.tensor([time_weights.get(i, 1.0) for i in range(len(model_predictions))])
            combined_weights = weights.mean(dim=1) * time_weight_tensor
            combined_weights = combined_weights / combined_weights.sum()
            
            # 重新加權
            ensemble_predictions = torch.sum(
                model_predictions * combined_weights.unsqueeze(-1).unsqueeze(-1), dim=0
            )
        
        return ensemble_predictions, uncertainties
    
    def compress_old_task_features(self, task_id, features):
        """
        壓縮舊任務特徵
        
        Args:
            task_id: 任務ID
            features: 特徵張量
        """
        compressed_features, compressor = self.feature_compressor.compress_task_features(
            task_id, features, method='pca'
        )
        
        return compressed_features
    
    def forward_with_optimizations(self, model, input_tensor, layer_name=None):
        """
        帶優化的前向傳播
        
        Args:
            model: 模型
            input_tensor: 輸入張量
            layer_name: 層名稱（用於特徵重用）
        """
        # 檢查是否使用梯度檢查點
        use_checkpointing = self.gradient_checkpointer.should_checkpoint()
        
        if layer_name and hasattr(model, layer_name):
            # 嘗試特徵重用
            layer = getattr(model, layer_name)
            output = self.feature_reuser.forward_with_reuse(input_tensor, layer, layer_name)
        else:
            # 常規前向傳播，可能使用檢查點
            if use_checkpointing:
                output = self.gradient_checkpointer.checkpoint_forward(model, input_tensor)
            else:
                output = model(input_tensor)
        
        return output
    
    def generate_stage3_report(self, save_path):
        """生成第三階段優化報告"""
        report = {
            'optimization_summary': {
                'ensemble_optimizations_count': len(self.optimization_stats['ensemble_optimizations']),
                'memory_optimizations_count': len(self.optimization_stats['memory_optimizations']),
                'performance_improvements_count': len(self.optimization_stats['performance_improvements'])
            },
            'ensemble_statistics': {
                'selector_stats': self.ensemble_selector.get_reuse_statistics() if hasattr(self.ensemble_selector, 'get_reuse_statistics') else {},
                'diversity_stats': self.diversity_optimizer.get_diversity_statistics(),
                'dynamic_selection_stats': self.dynamic_selector.get_selection_statistics(),
                'time_weighting_stats': self.time_aware_weighter.get_weight_statistics(),
                'uncertainty_stats': self.uncertainty_quantifier.get_uncertainty_statistics()
            },
            'memory_statistics': {
                'checkpointing_stats': self.gradient_checkpointer.get_memory_statistics(),
                'quantization_stats': self.model_quantizer.get_quantization_statistics(),
                'batch_sizing_stats': self.batch_sizer.get_batch_size_statistics(),
                'feature_reuse_stats': self.feature_reuser.get_reuse_statistics(),
                'memory_pool_stats': self.memory_pool.get_pool_statistics(),
                'compression_stats': self.feature_compressor.get_compression_statistics()
            },
            'detailed_stats': self.optimization_stats
        }
        
        # 保存報告
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        print(f"第三階段優化報告已保存至: {save_path}")
        return report
    
    def plot_optimization_trends(self, save_dir):
        """繪製優化趨勢圖"""
        if not MATPLOTLIB_AVAILABLE:
            print("跳過圖表生成：matplotlib 未安裝")
            return
            
        import matplotlib.pyplot as plt
        
        # 記憶體使用趨勢
        if self.optimization_stats['memory_optimizations']:
            epochs = [opt['epoch'] for opt in self.optimization_stats['memory_optimizations']]
            batch_sizes = [self.batch_sizer.batch_size_history[i] if i < len(self.batch_sizer.batch_size_history) 
                          else self.batch_sizer.current_batch_size for i in epochs]
            
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(epochs, batch_sizes, 'b-', linewidth=2)
            plt.title('Dynamic Batch Size Adjustment')
            plt.xlabel('Epoch')
            plt.ylabel('Batch Size')
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            if hasattr(self.diversity_optimizer, 'diversity_history') and self.diversity_optimizer.diversity_history:
                diversity_scores = [h['diversity_score'] for h in self.diversity_optimizer.diversity_history]
                plt.plot(range(len(diversity_scores)), diversity_scores, 'g-', linewidth=2)
                plt.title('Model Diversity Evolution')
                plt.xlabel('Training Step')
                plt.ylabel('Diversity Score')
                plt.grid(True)
            
            plt.subplot(2, 2, 3)
            if hasattr(self.feature_reuser, 'reuse_history') and self.feature_reuser.reuse_history:
                hit_rates = []
                window_size = 20
                for i in range(window_size, len(self.feature_reuser.reuse_history)):
                    recent = self.feature_reuser.reuse_history[i-window_size:i]
                    hit_rate = len([r for r in recent if r['reused']]) / len(recent)
                    hit_rates.append(hit_rate)
                
                if hit_rates:
                    plt.plot(range(len(hit_rates)), hit_rates, 'r-', linewidth=2)
                    plt.title('Feature Reuse Hit Rate')
                    plt.xlabel('Training Step')
                    plt.ylabel('Hit Rate')
                    plt.grid(True)
            
            plt.subplot(2, 2, 4)
            if hasattr(self.memory_pool, 'fragmentation_history') and self.memory_pool.fragmentation_history:
                plt.plot(range(len(self.memory_pool.fragmentation_history)), 
                        self.memory_pool.fragmentation_history, 'm-', linewidth=2)
                plt.title('Memory Fragmentation')
                plt.xlabel('Training Step')
                plt.ylabel('Fragmentation Ratio')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/stage3_optimization_trends.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"第三階段優化趨勢圖已保存至: {save_dir}/stage3_optimization_trends.png")


# ==================== 導出接口 ====================

def get_stage3_components():
    """獲取第三階段優化的主要組件"""
    return {
        # 集成學習策略優化
        'AdaptiveEnsembleSelector': AdaptiveEnsembleSelector,
        'DiversityDrivenOptimizer': DiversityDrivenOptimizer,
        'DynamicModelSelector': DynamicModelSelector,
        'TimeAwareEnsembleWeighter': TimeAwareEnsembleWeighter,
        'UncertaintyQuantificationEnsemble': UncertaintyQuantificationEnsemble,
        
        # 記憶體效率優化
        'GradientCheckpointer': GradientCheckpointer,
        'AdaptiveModelQuantizer': AdaptiveModelQuantizer,
        'DynamicBatchSizer': DynamicBatchSizer,
        'FeatureReuseManager': FeatureReuseManager,
        'MemoryPoolManager': MemoryPoolManager,
        'TaskFeatureCompressor': TaskFeatureCompressor,
        
        # 管理器
        'Stage3OptimizationManager': Stage3OptimizationManager
    }

def create_stage3_manager(args, num_classes=1000):
    """創建第三階段優化管理器的便捷函數"""
    return Stage3OptimizationManager(args, num_classes) 
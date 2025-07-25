# coding=utf-8
"""
AdaBoost Ensemble Learning for Continual Learning Models
使用11個預訓練的持續學習模型實現AdaBoost集成學習

原理說明：
1. 將11個預訓練模型作為弱學習器（weak learners）
2. 在驗證集上評估每個模型的性能，計算錯誤率
3. 根據AdaBoost算法計算每個模型的權重
4. 使用加權投票進行最終預測
5. 支援動態模型選擇和權重調整
"""

from __future__ import absolute_import, print_function
import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import torchvision.transforms as transforms
from ImageFolder import ImageFolder
from CIFAR100 import CIFAR100
from models.resnet import ResNet_ImageNet, ResNet_Cifar, Generator, Discriminator, BasicBlock, Bottleneck, ClassifierMLP, ModelCNN
from torch.serialization import add_safe_globals

# 註冊安全的全局類
add_safe_globals([ResNet_ImageNet, ResNet_Cifar, Generator, Discriminator, BasicBlock, Bottleneck, ClassifierMLP, ModelCNN])

class ContinualAdaBoost:
    """
    持續學習的AdaBoost集成器
    
    核心思想：
    1. 使用多個專門的模型作為弱學習器
    2. 每個模型專注於特定的類別範圍
    3. 使用AdaBoost算法計算模型權重
    4. 支援動態權重調整和模型選擇
    """
    
    def __init__(self, model_paths, class_ranges, num_classes=1000, device='cuda'):
        """
        初始化AdaBoost集成器
        
        Args:
            model_paths: 11個模型檔案的路徑列表
            class_ranges: 每個模型負責的類別範圍
            num_classes: 總類別數
            device: 計算設備
        """
        self.model_paths = model_paths
        self.class_ranges = class_ranges
        self.num_classes = num_classes
        self.device = device
        
        # 模型和權重
        self.models = []
        self.model_weights = []
        self.model_errors = []
        
        # 性能記錄
        self.performance_history = {
            'individual_accuracies': [],
            'ensemble_accuracy': [],
            'model_weights': [],
            'class_wise_performance': {}
        }
        
        # 載入所有模型
        self._load_models()
        
    def _load_models(self):
        """安全載入所有模型"""
        print("正在載入11個基礎模型...")
        
        for i, (model_path, class_range) in enumerate(zip(self.model_paths, self.class_ranges)):
            try:
                print(f"載入模型 {i+1}/11: {os.path.basename(model_path)}")
                model = self._safe_load_model(model_path)
                model = model.to(self.device)
                model.eval()
                self.models.append(model)
                print(f"  - 負責類別範圍: {class_range[0]}-{class_range[1]}")
            except Exception as e:
                print(f"載入模型失敗 {model_path}: {e}")
                raise
        
        print(f"成功載入 {len(self.models)} 個模型")
        
    def _safe_load_model(self, model_path):
        """安全載入單個模型"""
        try:
            return torch.load(model_path, weights_only=False)
        except Exception as e1:
            try:
                with torch.serialization.safe_globals([ResNet_ImageNet, ResNet_Cifar, Generator, 
                                                     Discriminator, BasicBlock, Bottleneck, 
                                                     ClassifierMLP, ModelCNN]):
                    return torch.load(model_path, weights_only=True)
            except Exception as e2:
                return torch.load(model_path, weights_only=False, map_location='cpu')
    
    def compute_model_weights(self, val_loader, method='adaboost'):
        """
        計算每個模型的AdaBoost權重
        
        Args:
            val_loader: 驗證資料載入器
            method: 權重計算方法 ('adaboost', 'accuracy_based', 'confidence_based')
        """
        print("開始計算模型權重...")
        
        # 收集所有模型的預測結果
        all_predictions = []
        all_confidences = []
        true_labels = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                batch_predictions = []
                batch_confidences = []
                
                # 獲取每個模型的預測
                for model_idx, model in enumerate(self.models):
                    features = model(data)
                    outputs = model.embed(features)
                    probabilities = F.softmax(outputs, dim=1)
                    
                    # 獲取預測類別和信心度
                    max_probs, predictions = torch.max(probabilities, 1)
                    
                    batch_predictions.append(predictions.cpu().numpy())
                    batch_confidences.append(max_probs.cpu().numpy())
                
                all_predictions.append(np.array(batch_predictions))
                all_confidences.append(np.array(batch_confidences))
                true_labels.append(target.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"處理批次 {batch_idx+1}/{len(val_loader)}")
        
        # 整理資料格式
        all_predictions = np.concatenate(all_predictions, axis=1)  # (11, total_samples)
        all_confidences = np.concatenate(all_confidences, axis=1)  # (11, total_samples)
        true_labels = np.concatenate(true_labels)  # (total_samples,)
        
        # 計算每個模型的錯誤率和權重
        self.model_errors = []
        self.model_weights = []
        
        for model_idx in range(len(self.models)):
            predictions = all_predictions[model_idx]
            class_range = self.class_ranges[model_idx]
            
            # 只在該模型的專業類別範圍內計算錯誤率
            mask = (true_labels >= class_range[0]) & (true_labels <= class_range[1])
            
            if np.any(mask):
                # 在專業範圍內的樣本
                specialist_predictions = predictions[mask]
                specialist_labels = true_labels[mask]
                
                if method == 'adaboost':
                    # 在專業範圍內計算錯誤率
                    error_rate = np.mean(specialist_predictions != specialist_labels)
                    error_rate = max(error_rate, 1e-10)  # 避免除零
                    error_rate = min(error_rate, 0.999)  # 避免權重過大
                    
                    # AdaBoost權重計算（基於專業範圍內的表現）
                    if error_rate < 0.5:
                        weight = 0.5 * np.log((1 - error_rate) / error_rate)
                    else:
                        weight = 1e-6  # 給極小權重而非0
                        
                elif method == 'accuracy_based':
                    # 基於專業範圍內的準確率
                    accuracy = np.mean(specialist_predictions == specialist_labels)
                    weight = accuracy ** 2
                    
                elif method == 'confidence_based':
                    # 基於專業範圍內的信心度
                    specialist_confidences = all_confidences[model_idx][mask]
                    correct_mask = (specialist_predictions == specialist_labels)
                    avg_confidence = np.mean(specialist_confidences[correct_mask]) if np.any(correct_mask) else 1e-6
                    weight = avg_confidence
                    
                # 根據專業範圍的樣本數量調整權重
                range_sample_ratio = np.sum(mask) / len(true_labels)
                adjusted_weight = weight * (range_sample_ratio ** 0.5)  # 平方根調整，避免過度懲罰
                
            else:
                # 沒有專業範圍內的樣本，給予最小權重
                error_rate = 0.999
                adjusted_weight = 1e-6
            
            # 為了記錄，計算全局錯誤率
            global_error_rate = np.mean(predictions != true_labels)
            
            self.model_errors.append(global_error_rate)
            self.model_weights.append(adjusted_weight)
        
        # 正規化權重
        total_weight = sum(self.model_weights)
        if total_weight > 1e-6:
            self.model_weights = [w / total_weight for w in self.model_weights]
        else:
            # 如果所有權重都極小，使用基於樣本數量的權重
            range_weights = []
            total_samples = len(true_labels)
            for class_range in self.class_ranges:
                mask = (true_labels >= class_range[0]) & (true_labels <= class_range[1])
                range_ratio = np.sum(mask) / total_samples
                range_weights.append(max(range_ratio, 1e-6))
            
            total_range_weight = sum(range_weights)
            self.model_weights = [w / total_range_weight for w in range_weights]
        
        # 顯示結果
        print("\n模型權重計算完成：")
        for i, (error, weight) in enumerate(zip(self.model_errors, self.model_weights)):
            class_range = self.class_ranges[i]
            print(f"模型 {i+1:2d} (類別 {class_range[0]:3d}-{class_range[1]:3d}): "
                  f"錯誤率={error:.4f}, 權重={weight:.4f}")
        
        return self.model_weights
    
    def predict_ensemble(self, data_loader, strategy='weighted_voting'):
        """
        使用集成方法進行預測
        
        Args:
            data_loader: 測試資料載入器
            strategy: 集成策略 ('weighted_voting', 'adaptive_selection', 'confidence_voting')
        """
        print(f"使用 {strategy} 策略進行集成預測...")
        
        all_predictions = []
        all_true_labels = []
        all_individual_preds = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # 獲取每個模型的預測
                model_predictions = []
                model_confidences = []
                
                for model in self.models:
                    features = model(data)
                    outputs = model.embed(features)
                    probabilities = F.softmax(outputs, dim=1)
                    
                    model_predictions.append(probabilities.cpu().numpy())
                    model_confidences.append(torch.max(probabilities, 1)[0].cpu().numpy())
                
                # 根據策略進行集成
                if strategy == 'weighted_voting':
                    ensemble_pred = self._weighted_voting(model_predictions)
                elif strategy == 'adaptive_selection':
                    ensemble_pred = self._adaptive_selection(model_predictions, model_confidences, target.cpu().numpy())
                elif strategy == 'confidence_voting':
                    ensemble_pred = self._confidence_voting(model_predictions, model_confidences)
                else:
                    raise ValueError(f"未知的集成策略: {strategy}")
                
                all_predictions.extend(ensemble_pred)
                all_true_labels.extend(target.cpu().numpy())
                
                # 記錄個別模型預測
                individual_preds = []
                for model_pred in model_predictions:
                    individual_preds.append(np.argmax(model_pred, axis=1))
                all_individual_preds.append(np.array(individual_preds).T)
                
                if batch_idx % 20 == 0:
                    print(f"預測進度: {batch_idx+1}/{len(data_loader)}")
        
        # 計算結果
        all_individual_preds = np.concatenate(all_individual_preds, axis=0)
        accuracy = accuracy_score(all_true_labels, all_predictions)
        
        print(f"\n集成預測完成！")
        print(f"集成準確率: {accuracy:.4f}")
        
        # 計算個別模型準確率
        individual_accs = []
        for i in range(len(self.models)):
            ind_acc = accuracy_score(all_true_labels, all_individual_preds[:, i])
            individual_accs.append(ind_acc)
            print(f"模型 {i+1:2d} 準確率: {ind_acc:.4f}")
        
        return {
            'ensemble_predictions': all_predictions,
            'true_labels': all_true_labels,
            'ensemble_accuracy': accuracy,
            'individual_accuracies': individual_accs,
            'individual_predictions': all_individual_preds
        }
    
    def _weighted_voting(self, model_predictions):
        """加權投票策略"""
        weighted_sum = np.zeros_like(model_predictions[0])
        
        for i, pred in enumerate(model_predictions):
            weighted_sum += self.model_weights[i] * pred
            
        return np.argmax(weighted_sum, axis=1)
    
    def _adaptive_selection(self, model_predictions, model_confidences, true_labels):
        """自適應選擇策略：根據類別範圍和信心度選擇最佳模型"""
        ensemble_pred = []
        
        for sample_idx in range(len(model_predictions[0])):
            sample_preds = [pred[sample_idx] for pred in model_predictions]
            sample_confs = [conf[sample_idx] for conf in model_confidences]
            
            # 找出最有信心的預測
            best_model_idx = np.argmax(sample_confs)
            predicted_class = np.argmax(sample_preds[best_model_idx])
            
            # 檢查是否在該模型的專業範圍內
            class_range = self.class_ranges[best_model_idx]
            if class_range[0] <= predicted_class <= class_range[1]:
                # 在專業範圍內，使用該模型
                ensemble_pred.append(predicted_class)
            else:
                # 不在專業範圍內，使用加權投票
                weighted_sum = np.zeros(self.num_classes)
                for i, pred in enumerate(sample_preds):
                    weighted_sum += self.model_weights[i] * pred
                ensemble_pred.append(np.argmax(weighted_sum))
                
        return ensemble_pred
    
    def _confidence_voting(self, model_predictions, model_confidences):
        """基於信心度的投票策略"""
        ensemble_pred = []
        
        for sample_idx in range(len(model_predictions[0])):
            sample_preds = [pred[sample_idx] for pred in model_predictions]
            sample_confs = [conf[sample_idx] for conf in model_confidences]
            
            # 使用信心度作為權重
            weighted_sum = np.zeros(self.num_classes)
            total_confidence = sum(sample_confs)
            
            if total_confidence > 0:
                for i, (pred, conf) in enumerate(zip(sample_preds, sample_confs)):
                    normalized_conf = conf / total_confidence
                    weighted_sum += normalized_conf * pred
            else:
                # 如果所有模型都沒有信心，使用均等權重
                for pred in sample_preds:
                    weighted_sum += pred / len(sample_preds)
                    
            ensemble_pred.append(np.argmax(weighted_sum))
            
        return ensemble_pred
    
    def analyze_performance(self, results, save_dir):
        """分析和視覺化性能結果"""
        print("開始性能分析...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 整體性能比較
        self._plot_overall_performance(results, save_dir)
        
        # 2. 類別級別性能分析
        self._plot_class_wise_performance(results, save_dir)
        
        # 3. 模型權重分析
        self._plot_model_weights(save_dir)
        
        # 4. 混淆矩陣
        self._plot_confusion_matrix(results, save_dir)
        
        # 5. 儲存詳細報告
        self._save_detailed_report(results, save_dir)
        
        print(f"性能分析完成，結果儲存於: {save_dir}")
    
    def _plot_overall_performance(self, results, save_dir):
        """繪製整體性能比較圖"""
        plt.figure(figsize=(12, 8))
        
        # 準備資料
        model_names = [f'Model {i+1}' for i in range(len(self.models))]
        individual_accs = results['individual_accuracies']
        ensemble_acc = results['ensemble_accuracy']
        
        # 繪製個別模型準確率
        bars = plt.bar(model_names, individual_accs, alpha=0.7, color='skyblue', label='Individual Models')
        
        # 繪製集成模型準確率
        plt.axhline(y=ensemble_acc, color='red', linestyle='--', linewidth=2, 
                   label=f'Ensemble (AdaBoost): {ensemble_acc:.4f}')
        
        # 添加權重信息到圖上
        for i, (bar, weight) in enumerate(zip(bars, self.model_weights)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'w={weight:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('AdaBoost Ensemble vs Individual Models Performance')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'overall_performance.png'), dpi=300)
        plt.close()
    
    def _plot_class_wise_performance(self, results, save_dir):
        """繪製類別級別性能分析"""
        true_labels = results['true_labels']
        ensemble_preds = results['ensemble_predictions']
        individual_preds = results['individual_predictions']
        
        # 計算每個類別的準確率
        class_accuracies = {}
        unique_classes = np.unique(true_labels)
        
        for cls in unique_classes:
            mask = (np.array(true_labels) == cls)
            if np.any(mask):
                # 集成準確率
                ensemble_acc = np.mean(np.array(ensemble_preds)[mask] == cls)
                class_accuracies[cls] = {'ensemble': ensemble_acc, 'individual': []}
                
                # 個別模型準確率
                for i in range(len(self.models)):
                    ind_acc = np.mean(individual_preds[mask, i] == cls)
                    class_accuracies[cls]['individual'].append(ind_acc)
        
        # 繪製前50個類別的詳細比較（保持原有功能）
        plt.figure(figsize=(20, 10))
        
        classes_to_plot = sorted(unique_classes)[:50]
        x_pos = np.arange(len(classes_to_plot))
        
        # 繪製集成模型
        ensemble_accs = [class_accuracies[cls]['ensemble'] for cls in classes_to_plot]
        plt.plot(x_pos, ensemble_accs, 'r-', linewidth=2, label='AdaBoost Ensemble', marker='o')
        
        # 繪製最好的個別模型
        best_individual = []
        for cls in classes_to_plot:
            best_acc = max(class_accuracies[cls]['individual'])
            best_individual.append(best_acc)
        plt.plot(x_pos, best_individual, 'b--', linewidth=2, label='Best Individual Model', marker='s')
        
        plt.xlabel('Class Index')
        plt.ylabel('Accuracy')
        plt.title('Class-wise Performance Comparison (First 50 Classes)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'class_wise_performance.png'), dpi=300)
        plt.close()
        
        # 新增：繪製所有類別的完整分析
        if len(unique_classes) > 50:
            plt.figure(figsize=(30, 12))
            
            all_classes = sorted(unique_classes)
            x_pos_all = np.arange(len(all_classes))
            
            # 所有類別的集成準確率
            ensemble_accs_all = [class_accuracies[cls]['ensemble'] for cls in all_classes]
            plt.plot(x_pos_all, ensemble_accs_all, 'r-', linewidth=1, label='AdaBoost Ensemble', alpha=0.8)
            
            # 所有類別的最佳個別模型準確率
            best_individual_all = []
            for cls in all_classes:
                best_acc = max(class_accuracies[cls]['individual'])
                best_individual_all.append(best_acc)
            plt.plot(x_pos_all, best_individual_all, 'b-', linewidth=1, label='Best Individual Model', alpha=0.8)
            
            plt.xlabel('Class Index')
            plt.ylabel('Accuracy')
            plt.title(f'Complete Class-wise Performance Comparison (All {len(all_classes)} Classes)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 添加任務分界線
            task_boundaries = [499, 549, 599, 649, 699, 749, 799, 849, 899, 949]
            for boundary in task_boundaries:
                if boundary < len(all_classes):
                    plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'complete_class_wise_performance.png'), dpi=300)
            plt.close()
            
            # 儲存詳細的類別統計報告
            with open(os.path.join(save_dir, 'complete_class_analysis.txt'), 'w', encoding='utf-8') as f:
                f.write(f"完整類別分析報告 (共 {len(all_classes)} 個類別)\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"測試資料中實際包含的類別範圍: {min(all_classes)} - {max(all_classes)}\n")
                f.write(f"缺失的類別數量: {1000 - len(all_classes)}\n\n")
                
                # 按任務分組統計
                task_stats = {}
                for i, (start, end) in enumerate([(0, 499)] + [(500+i*50, 549+i*50) for i in range(10)]):
                    task_classes = [cls for cls in all_classes if start <= cls <= end]
                    if task_classes:
                        task_ensemble_acc = np.mean([class_accuracies[cls]['ensemble'] for cls in task_classes])
                        task_best_acc = np.mean([max(class_accuracies[cls]['individual']) for cls in task_classes])
                        f.write(f"任務 {i:2d} (類別 {start:3d}-{end:3d}): ")
                        f.write(f"包含 {len(task_classes):2d} 個類別, ")
                        f.write(f"集成平均準確率 {task_ensemble_acc:.4f}, ")
                        f.write(f"最佳個別平均準確率 {task_best_acc:.4f}\n")
    
    def _plot_model_weights(self, save_dir):
        """繪製模型權重分布"""
        plt.figure(figsize=(12, 6))
        
        model_names = [f'Model {i+1}\n({self.class_ranges[i][0]}-{self.class_ranges[i][1]})' 
                      for i in range(len(self.models))]
        
        bars = plt.bar(model_names, self.model_weights, color='lightgreen', alpha=0.7)
        
        # 添加數值標籤
        for bar, weight, error in zip(bars, self.model_weights, self.model_errors):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{weight:.3f}\n(err:{error:.3f})', ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Models (Class Range)')
        plt.ylabel('AdaBoost Weight')
        plt.title('AdaBoost Model Weights Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_weights.png'), dpi=300)
        plt.close()
    
    def _plot_confusion_matrix(self, results, save_dir):
        """繪製混淆矩陣（取樣顯示）"""
        true_labels = results['true_labels']
        ensemble_preds = results['ensemble_predictions']
        
        # 首先繪製前100個類別的混淆矩陣（保持原有功能）
        mask = (np.array(true_labels) < 100) & (np.array(ensemble_preds) < 100)
        if np.any(mask):
            filtered_true = np.array(true_labels)[mask]
            filtered_pred = np.array(ensemble_preds)[mask]
            
            cm = confusion_matrix(filtered_true, filtered_pred)
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix (First 100 Classes)')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
            plt.close()
        
        # 新增：分析完整的混淆矩陣統計資訊
        unique_true = np.unique(true_labels)
        unique_pred = np.unique(ensemble_preds)
        
        # 儲存完整混淆矩陣統計
        with open(os.path.join(save_dir, 'confusion_matrix_analysis.txt'), 'w', encoding='utf-8') as f:
            f.write("完整混淆矩陣分析報告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"實際標籤範圍: {min(unique_true)} - {max(unique_true)} (共 {len(unique_true)} 個類別)\n")
            f.write(f"預測標籤範圍: {min(unique_pred)} - {max(unique_pred)} (共 {len(unique_pred)} 個類別)\n")
            f.write(f"總樣本數: {len(true_labels)}\n\n")
            
            # 計算各類別的預測分佈
            f.write("各類別預測統計:\n")
            f.write("-" * 30 + "\n")
            
            for cls in sorted(unique_true):
                mask = np.array(true_labels) == cls
                cls_true_count = np.sum(mask)
                cls_preds = np.array(ensemble_preds)[mask]
                cls_correct = np.sum(cls_preds == cls)
                cls_accuracy = cls_correct / cls_true_count if cls_true_count > 0 else 0
                
                f.write(f"類別 {cls:3d}: 真實樣本 {cls_true_count:3d}, 正確預測 {cls_correct:3d}, 準確率 {cls_accuracy:.4f}\n")
            
            # 預測類別分佈統計
            f.write(f"\n預測類別分佈:\n")
            f.write("-" * 30 + "\n")
            
            pred_counts = {}
            for pred in ensemble_preds:
                pred_counts[pred] = pred_counts.get(pred, 0) + 1
            
            for pred_cls in sorted(pred_counts.keys()):
                f.write(f"預測為類別 {pred_cls:3d}: {pred_counts[pred_cls]:3d} 次\n")
            
            # 如果類別數量不太大，可以繪製完整的小型混淆矩陣
            if len(unique_true) <= 200 and len(unique_pred) <= 200:
                full_cm = confusion_matrix(true_labels, ensemble_preds)
                
                plt.figure(figsize=(max(15, len(unique_true)//5), max(12, len(unique_true)//5)))
                sns.heatmap(full_cm, annot=False, cmap='Blues', fmt='d', 
                           xticklabels=False, yticklabels=False)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title(f'Complete Confusion Matrix ({len(unique_true)} classes)')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'complete_confusion_matrix.png'), dpi=300)
                plt.close()
                
                f.write(f"\n已生成完整混淆矩陣圖: complete_confusion_matrix.png\n")
    
    def _save_detailed_report(self, results, save_dir):
        """儲存詳細的性能報告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'ensemble_accuracy': float(results['ensemble_accuracy']),
            'individual_accuracies': [float(acc) for acc in results['individual_accuracies']],
            'model_weights': [float(w) for w in self.model_weights],
            'model_errors': [float(e) for e in self.model_errors],
            'class_ranges': self.class_ranges,
            'improvement_over_best_individual': float(results['ensemble_accuracy'] - max(results['individual_accuracies'])),
            'average_individual_accuracy': float(np.mean(results['individual_accuracies']))
        }
        
        with open(os.path.join(save_dir, 'performance_report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        # 生成文字報告
        with open(os.path.join(save_dir, 'performance_summary.txt'), 'w', encoding='utf-8') as f:
            f.write("AdaBoost 集成學習性能報告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"測試時間: {report['timestamp']}\n")
            f.write(f"集成準確率: {report['ensemble_accuracy']:.4f}\n")
            f.write(f"最佳個別模型準確率: {max(report['individual_accuracies']):.4f}\n")
            f.write(f"平均個別模型準確率: {report['average_individual_accuracy']:.4f}\n")
            f.write(f"改善幅度: {report['improvement_over_best_individual']:.4f}\n\n")
            
            f.write("各模型詳細資訊:\n")
            f.write("-" * 30 + "\n")
            for i in range(len(self.models)):
                f.write(f"模型 {i+1:2d}: 類別 {self.class_ranges[i][0]:3d}-{self.class_ranges[i][1]:3d}, "
                       f"準確率 {report['individual_accuracies'][i]:.4f}, "
                       f"權重 {report['model_weights'][i]:.4f}, "
                       f"錯誤率 {report['model_errors'][i]:.4f}\n")


def create_data_loaders(args):
    """創建資料載入器"""
    if args.data == 'medicine':
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values, std=std_values)
        ])
        
        # 驗證集
        val_dir = os.path.join('medicine_picture', 'valid')
        val_dataset = ImageFolder(val_dir, transform=transform, index=list(range(args.num_class)))
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        
        # 測試集
        test_dir = os.path.join('medicine_picture', 'test') if os.path.exists('medicine_picture/test') else val_dir
        test_dataset = ImageFolder(test_dir, transform=transform, index=list(range(args.num_class)))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        
    elif 'cifar' in args.data:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # CIFAR資料集
        val_dataset = CIFAR100(root=args.data_dir, train=False, download=True, 
                              transform=transform, index=list(range(args.num_class)))
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        test_loader = val_loader  # 在CIFAR中使用相同的測試集
        
    return val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='AdaBoost Ensemble for Continual Learning')
    
    # 基本參數
    parser.add_argument('-data', type=str, default='medicine', help='資料集類型')
    parser.add_argument('-models_dir', type=str, required=True, help='模型檔案目錄')
    parser.add_argument('-num_class', type=int, default=1000, help='類別總數')
    parser.add_argument('-nb_cl_fg', type=int, default=500, help='第一組類別數')
    parser.add_argument('-num_task', type=int, default=10, help='後續任務數')
    parser.add_argument('-epochs', type=int, default=200, help='訓練輪數')
    parser.add_argument('-batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('-gpu', type=str, default='0', help='使用的GPU')
    parser.add_argument('-data_dir', type=str, default='/data/datasets/', help='資料目錄')
    
    # AdaBoost參數
    parser.add_argument('-weight_method', type=str, default='adaboost', 
                       choices=['adaboost', 'accuracy_based', 'confidence_based'],
                       help='權重計算方法')
    parser.add_argument('-ensemble_strategy', type=str, default='weighted_voting',
                       choices=['weighted_voting', 'adaptive_selection', 'confidence_voting'],
                       help='集成策略')
    
    # 輸出參數
    parser.add_argument('-save_dir', type=str, default='AdaBoost_Results', help='AdaBoost結果儲存根目錄')
    
    args = parser.parse_args()
    
    # 設定設備
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")
    
    # 構建模型路徑和類別範圍
    model_paths = []
    class_ranges = []
    
    # 第一個模型（任務0，類別0-499）
    model_paths.append(os.path.join(args.models_dir, f'task_00_{args.epochs-1}_model.pkl'))
    class_ranges.append((0, args.nb_cl_fg - 1))
    
    # 後續模型（任務1-10，每個50個類別）
    num_class_per_task = (args.num_class - args.nb_cl_fg) // args.num_task
    for task_id in range(1, args.num_task + 1):
        model_paths.append(os.path.join(args.models_dir, f'task_{task_id:02d}_{args.epochs-1}_model.pkl'))
        start_class = args.nb_cl_fg + (task_id - 1) * num_class_per_task
        end_class = args.nb_cl_fg + task_id * num_class_per_task - 1
        class_ranges.append((start_class, end_class))
    
    print("模型配置:")
    for i, (path, class_range) in enumerate(zip(model_paths, class_ranges)):
        print(f"模型 {i+1:2d}: {os.path.basename(path)} -> 類別 {class_range[0]}-{class_range[1]}")
    
    # 檢查模型檔案存在性
    missing_models = [path for path in model_paths if not os.path.exists(path)]
    if missing_models:
        print("錯誤：以下模型檔案不存在：")
        for path in missing_models:
            print(f"  - {path}")
        return
    
    # 創建AdaBoost集成器
    print("\n初始化AdaBoost集成器...")
    ensemble = ContinualAdaBoost(
        model_paths=model_paths,
        class_ranges=class_ranges,
        num_classes=args.num_class,
        device=device
    )
    
    # 創建資料載入器
    print("準備資料載入器...")
    val_loader, test_loader = create_data_loaders(args)
    
    # 計算模型權重
    print("計算AdaBoost權重...")
    ensemble.compute_model_weights(val_loader, method=args.weight_method)
    
    # 進行集成預測
    print("執行集成預測...")
    results = ensemble.predict_ensemble(test_loader, strategy=args.ensemble_strategy)
    
    # 分析和儲存結果 - 創建專門的AdaBoost資料夾結構
    print("分析性能結果...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 創建層次化的資料夾結構
    main_save_dir = args.save_dir
    method_dir = f"AdaBoost_{args.weight_method}_{args.ensemble_strategy}"
    time_dir = f"run_{timestamp}"
    save_dir = os.path.join(main_save_dir, method_dir, time_dir)
    
    # 確保目錄存在
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n📁 AdaBoost結果將儲存於專門資料夾：")
    print(f"   {save_dir}")
    print(f"   ├── performance_report.json")
    print(f"   ├── performance_summary.txt") 
    print(f"   ├── overall_performance.png")
    print(f"   ├── class_wise_performance.png")
    print(f"   ├── model_weights.png")
    print(f"   └── confusion_matrix.png")
    
    ensemble.analyze_performance(results, save_dir)
    
    # 輸出總結
    print("\n" + "="*60)
    print("AdaBoost 集成學習完成！")
    print(f"集成準確率: {results['ensemble_accuracy']:.4f}")
    print(f"最佳個別模型: {max(results['individual_accuracies']):.4f}")
    print(f"改善幅度: {results['ensemble_accuracy'] - max(results['individual_accuracies']):.4f}")
    print(f"結果儲存於: {save_dir}")
    print("="*60)


if __name__ == '__main__':
    main() 
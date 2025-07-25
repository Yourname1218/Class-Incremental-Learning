#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AdaBoost-Stacking混合集成學習
採用Stacking的前期處理方式，但保持AdaBoost的最終加權投票決策
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import json
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns

# 導入專案的自訂ImageFolder和CIFAR100
from ImageFolder import ImageFolder
from CIFAR100 import CIFAR100


class AdaBoostStackingHybrid:
    """
    AdaBoost-Stacking混合集成器
    採用Stacking的模型評估方式，但使用AdaBoost的最終決策機制
    """
    
    def __init__(self, model_paths, class_ranges, num_classes=1000, device='cuda'):
        self.model_paths = model_paths
        self.class_ranges = class_ranges
        self.num_classes = num_classes
        self.device = device
        self.models = []
        self.model_weights = []
        self.model_errors = []
        
        print(f"初始化AdaBoost-Stacking混合集成器...")
        print(f"模型數量: {len(model_paths)}")
        print(f"類別總數: {num_classes}")
        
        # 載入模型
        self._load_models()
    
    def _load_models(self):
        """載入所有基礎模型（採用Stacking方式）"""
        print("載入基礎模型...")
        for i, path in enumerate(self.model_paths):
            try:
                model = torch.load(path, weights_only=False, map_location=self.device)
                model.to(self.device)
                model.eval()
                self.models.append(model)
                class_range = self.class_ranges[i]
                print(f"✅ 模型 {i+1:2d}: 載入成功 -> 專業範圍 {class_range[0]}-{class_range[1]}")
            except Exception as e:
                print(f"❌ 模型 {i+1:2d}: 載入失敗 - {e}")
        
        print(f"成功載入 {len(self.models)} 個模型")
    
    def collect_all_predictions(self, dataloader, desc="收集預測"):
        """
        收集所有模型的預測結果（採用Stacking方式）
        返回: (predictions_list, true_labels)
        """
        print(f"{desc}中...")
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc=desc)):
                data, target = data.to(self.device), target.to(self.device)
                batch_predictions = []
                
                # 獲取每個模型的預測概率分布
                for model in self.models:
                    features = model(data)
                    outputs = model.embed(features)
                    probabilities = F.softmax(outputs, dim=1)
                    batch_predictions.append(probabilities)
                
                all_predictions.append(batch_predictions)
                all_labels.append(target)
        
        # 整理數據格式 - 每個模型的所有預測
        model_predictions = []
        for i in range(len(self.models)):
            model_preds = []
            for batch_idx in range(len(all_predictions)):
                model_preds.append(all_predictions[batch_idx][i])
            model_predictions.append(torch.cat(model_preds, dim=0))
        
        # 將預測移到CPU以節省GPU內存
        model_predictions = [x.cpu() for x in model_predictions]
        true_labels = torch.cat(all_labels, dim=0).cpu()
        
        return model_predictions, true_labels
    
    def compute_stacking_weights(self, val_loader):
        """
        使用Stacking方式計算模型權重
        基於整體驗證集表現，而非專業範圍限制
        """
        print("使用Stacking方式計算模型權重...")
        
        # 收集驗證集上的預測
        model_predictions, true_labels = self.collect_all_predictions(val_loader, "驗證集評估")
        
        # 計算每個模型在整體驗證集上的準確率
        self.model_errors = []
        individual_accuracies = []
        
        for i, predictions in enumerate(model_predictions):
            # 獲取預測類別
            predicted_classes = torch.argmax(predictions, dim=1)
            
            # 計算準確率
            accuracy = accuracy_score(true_labels.numpy(), predicted_classes.numpy())
            individual_accuracies.append(accuracy)
            
            # 計算錯誤率
            error_rate = 1.0 - accuracy
            self.model_errors.append(error_rate)
            
            print(f"模型 {i+1:2d}: 準確率={accuracy:.4f}, 錯誤率={error_rate:.4f}")
        
        # 使用AdaBoost權重公式，但基於整體表現
        self.model_weights = []
        for i, error_rate in enumerate(self.model_errors):
            # 避免除零和權重過大
            error_rate = max(error_rate, 1e-10)
            error_rate = min(error_rate, 0.999)
            
            if error_rate < 0.5:
                weight = 0.5 * np.log((1 - error_rate) / error_rate)
            else:
                weight = 1e-6  # 表現差的模型給極小權重
            
            self.model_weights.append(weight)
        
        # 權重正規化（確保總和為1）
        total_weight = sum(self.model_weights)
        if total_weight > 1e-6:
            self.model_weights = [w / total_weight for w in self.model_weights]
        else:
            # 如果所有權重都極小，使用均等權重
            self.model_weights = [1.0 / len(self.models) for _ in self.models]
        
        print("\n🎯 Stacking方式權重計算完成：")
        for i, (error, weight, acc) in enumerate(zip(self.model_errors, self.model_weights, individual_accuracies)):
            class_range = self.class_ranges[i]
            print(f"模型 {i+1:2d} (類別 {class_range[0]:3d}-{class_range[1]:3d}): "
                  f"準確率={acc:.4f}, 錯誤率={error:.4f}, AdaBoost權重={weight:.4f}")
        
        return self.model_weights
    
    def adaboost_weighted_voting(self, model_predictions):
        """
        AdaBoost最終決策：加權投票
        與Stacking不同，這裡是所有模型參與投票，而非選擇單一模型
        """
        # 確保預測在相同設備上
        device = model_predictions[0].device if isinstance(model_predictions[0], torch.Tensor) else 'cpu'
        
        # 轉換為numpy進行計算
        if isinstance(model_predictions[0], torch.Tensor):
            numpy_predictions = [pred.cpu().numpy() for pred in model_predictions]
        else:
            numpy_predictions = model_predictions
        
        # AdaBoost加權求和
        weighted_sum = np.zeros_like(numpy_predictions[0])
        
        for i, pred in enumerate(numpy_predictions):
            weighted_sum += self.model_weights[i] * pred
        
        # argmax選擇最終類別
        final_predictions = np.argmax(weighted_sum, axis=1)
        
        return final_predictions, weighted_sum
    
    def predict_ensemble(self, test_loader):
        """
        集成預測：使用AdaBoost的加權投票決策
        """
        print("執行AdaBoost加權投票預測...")
        
        # 收集測試集預測
        model_predictions, true_labels = self.collect_all_predictions(test_loader, "測試集預測")
        
        # 使用AdaBoost加權投票
        ensemble_predictions, ensemble_probabilities = self.adaboost_weighted_voting(model_predictions)
        
        # 計算準確率
        ensemble_accuracy = accuracy_score(true_labels.numpy(), ensemble_predictions)
        
        # 計算個別模型準確率
        individual_accuracies = []
        individual_predictions = []
        
        for i, predictions in enumerate(model_predictions):
            predicted_classes = torch.argmax(predictions, dim=1).numpy()
            individual_predictions.append(predicted_classes)
            acc = accuracy_score(true_labels.numpy(), predicted_classes)
            individual_accuracies.append(acc)
        
        print(f"\n🎯 AdaBoost集成預測完成！")
        print(f"集成準確率: {ensemble_accuracy:.4f}")
        print("各模型準確率:")
        for i, acc in enumerate(individual_accuracies):
            print(f"  模型 {i+1:2d}: {acc:.4f}")
        
        return {
            'ensemble_predictions': ensemble_predictions,
            'ensemble_probabilities': ensemble_probabilities,
            'true_labels': true_labels.numpy(),
            'ensemble_accuracy': ensemble_accuracy,
            'individual_accuracies': individual_accuracies,
            'individual_predictions': np.array(individual_predictions).T,
            'model_weights': self.model_weights.copy()
        }
    
    def analyze_performance(self, results, save_dir):
        """分析和視覺化性能結果"""
        print("開始性能分析...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 整體性能比較
        self._plot_overall_performance(results, save_dir)
        
        # 2. 權重分析
        self._plot_model_weights(save_dir)
        
        # 3. 混淆矩陣（前100類）
        self._plot_confusion_matrix(results, save_dir)
        
        # 4. 儲存詳細報告
        self._save_detailed_report(results, save_dir)
        
        print(f"性能分析完成，結果儲存於: {save_dir}")
    
    def _plot_overall_performance(self, results, save_dir):
        """繪製整體性能比較圖"""
        plt.figure(figsize=(12, 8))
        
        model_names = [f'Model {i+1}' for i in range(len(self.models))]
        individual_accs = results['individual_accuracies']
        ensemble_acc = results['ensemble_accuracy']
        
        # 繪製個別模型準確率
        bars = plt.bar(model_names, individual_accs, alpha=0.7, color='skyblue', label='Individual Models')
        
        # 繪製集成模型準確率
        plt.axhline(y=ensemble_acc, color='red', linestyle='--', linewidth=2, 
                   label=f'AdaBoost Ensemble: {ensemble_acc:.4f}')
        
        # 添加權重信息
        for i, (bar, weight) in enumerate(zip(bars, self.model_weights)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'w={weight:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('AdaBoost-Stacking Hybrid: Performance Comparison')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'overall_performance.png'), dpi=300)
        plt.close()
    
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
        plt.title('AdaBoost-Stacking Hybrid: Model Weights Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_weights.png'), dpi=300)
        plt.close()
    
    def _plot_confusion_matrix(self, results, save_dir):
        """繪製混淆矩陣（前100類）"""
        true_labels = results['true_labels']
        ensemble_preds = results['ensemble_predictions']
        
        # 只顯示前100個類別
        mask = (true_labels < 100) & (ensemble_preds < 100)
        if np.any(mask):
            filtered_true = true_labels[mask]
            filtered_pred = ensemble_preds[mask]
            
            cm = confusion_matrix(filtered_true, filtered_pred)
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix (First 100 Classes) - AdaBoost-Stacking Hybrid')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
            plt.close()
    
    def _save_detailed_report(self, results, save_dir):
        """儲存詳細的性能報告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'method': 'AdaBoost-Stacking Hybrid',
            'ensemble_accuracy': float(results['ensemble_accuracy']),
            'individual_accuracies': [float(acc) for acc in results['individual_accuracies']],
            'model_weights': [float(w) for w in self.model_weights],
            'model_errors': [float(e) for e in self.model_errors],
            'class_ranges': self.class_ranges,
            'improvement_over_best_individual': float(results['ensemble_accuracy'] - max(results['individual_accuracies'])),
            'average_individual_accuracy': float(np.mean(results['individual_accuracies']))
        }
        
        # JSON報告
        with open(os.path.join(save_dir, 'performance_report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        # 文字報告
        with open(os.path.join(save_dir, 'performance_summary.txt'), 'w', encoding='utf-8') as f:
            f.write("AdaBoost-Stacking 混合集成學習性能報告\n")
            f.write("=" * 60 + "\n\n")
            f.write("方法特徵:\n")
            f.write("- 前期處理: 採用Stacking方式 (整體驗證集評估)\n")
            f.write("- 最終決策: 採用AdaBoost方式 (加權投票)\n")
            f.write("- 權重計算: AdaBoost公式基於整體表現\n\n")
            
            f.write(f"測試時間: {report['timestamp']}\n")
            f.write(f"集成準確率: {report['ensemble_accuracy']:.4f}\n")
            f.write(f"最佳個別模型準確率: {max(report['individual_accuracies']):.4f}\n")
            f.write(f"平均個別模型準確率: {report['average_individual_accuracy']:.4f}\n")
            f.write(f"改善幅度: {report['improvement_over_best_individual']:.4f}\n\n")
            
            f.write("各模型詳細資訊:\n")
            f.write("-" * 40 + "\n")
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
        
        # 創建目標轉換，CIFAR100需要這個參數
        target_transform = np.arange(args.num_class)
        
        val_dataset = CIFAR100(root=args.data_dir, train=False, download=True, 
                              transform=transform, target_transform=target_transform,
                              index=list(range(args.num_class)))
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        test_loader = val_loader
        
    return val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='AdaBoost-Stacking Hybrid Ensemble Learning')
    
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
    parser.add_argument('-save_dir', type=str, default='AdaBoost_Stacking_Hybrid_Results', help='結果儲存目錄')
    
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
    
    print("🎯 AdaBoost-Stacking混合集成配置:")
    print("方法特徵:")
    print("  ✅ 前期處理: Stacking方式 (整體驗證集評估)")
    print("  ✅ 最終決策: AdaBoost方式 (加權投票)")
    print("  ✅ 權重計算: AdaBoost公式基於整體表現")
    print("\n模型配置:")
    for i, (path, class_range) in enumerate(zip(model_paths, class_ranges)):
        print(f"模型 {i+1:2d}: {os.path.basename(path)} -> 類別 {class_range[0]}-{class_range[1]}")
    
    # 檢查模型檔案存在性
    missing_models = [path for path in model_paths if not os.path.exists(path)]
    if missing_models:
        print("❌ 錯誤：以下模型檔案不存在：")
        for path in missing_models:
            print(f"  - {path}")
        return
    
    # 創建混合集成器
    print("\n初始化AdaBoost-Stacking混合集成器...")
    ensemble = AdaBoostStackingHybrid(
        model_paths=model_paths,
        class_ranges=class_ranges,
        num_classes=args.num_class,
        device=device
    )
    
    # 創建資料載入器
    print("準備資料載入器...")
    val_loader, test_loader = create_data_loaders(args)
    
    # 使用Stacking方式計算權重
    print("使用Stacking方式計算AdaBoost權重...")
    ensemble.compute_stacking_weights(val_loader)
    
    # 使用AdaBoost方式進行集成預測
    print("使用AdaBoost方式進行集成預測...")
    results = ensemble.predict_ensemble(test_loader)
    
    # 分析和儲存結果
    print("分析性能結果...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n📁 混合集成結果將儲存於：")
    print(f"   {save_dir}")
    print(f"   ├── performance_report.json")
    print(f"   ├── performance_summary.txt") 
    print(f"   ├── overall_performance.png")
    print(f"   ├── model_weights.png")
    print(f"   └── confusion_matrix.png")
    
    ensemble.analyze_performance(results, save_dir)
    
    # 輸出總結
    print("\n" + "="*70)
    print("🎯 AdaBoost-Stacking 混合集成學習完成！")
    print("="*70)
    print(f"方法特徵: Stacking前期處理 + AdaBoost最終決策")
    print(f"集成準確率: {results['ensemble_accuracy']:.4f}")
    print(f"最佳個別模型: {max(results['individual_accuracies']):.4f}")
    print(f"改善幅度: {results['ensemble_accuracy'] - max(results['individual_accuracies']):.4f}")
    print(f"結果儲存於: {save_dir}")
    print("="*70)


if __name__ == '__main__':
    main() 
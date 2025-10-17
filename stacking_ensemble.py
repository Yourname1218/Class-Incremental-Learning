# stacking_ensemble.py
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from ImageFolder import ImageFolder
from torch.serialization import add_safe_globals
from models.resnet import ResNet_ImageNet, ResNet_Cifar, Generator, Discriminator, BasicBlock, Bottleneck, ClassifierMLP, ModelCNN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
import json

# 添加調試輸出
print("腳本開始執行...")

# 命令行參數
parser = argparse.ArgumentParser(description='Stacking Ensemble for Medicine Classification')
parser.add_argument('-models_dir', type=str, default='checkpoints/medicine_10tasks', help='Directory containing base models')
parser.add_argument('-data_dir', type=str, default='medicine_picture/valid', help='Directory containing validation data')
parser.add_argument('-output_dir', type=str, default='ensemble_results', help='Directory to save results')
parser.add_argument('-batch_size', type=int, default=32, help='Batch size for meta-model training')
parser.add_argument('-epochs', type=int, default=30, help='Number of epochs for meta-model training')
parser.add_argument('-lr', type=float, default=0.001, help='Learning rate for meta-model training')
parser.add_argument('-val_split', type=float, default=0.7, help='Portion of data to use for meta-model training')
parser.add_argument('-gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--eval_only', action='store_true', help='只評估不重新訓練（需已有訓練好的模型）')
parser.add_argument('--use_confidence', action='store_true', help='使用置信度特徵增強Meta Model（推薦）')
args = parser.parse_args()

# 打印參數
print(f"參數設置: models_dir={args.models_dir}, data_dir={args.data_dir}, output_dir={args.output_dir}")

# 設置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print(f"使用GPU: {args.gpu}")

# 創建輸出目錄
os.makedirs(args.output_dir, exist_ok=True)
print(f"創建輸出目錄: {args.output_dir}")

# 註冊安全類，以便加載模型
add_safe_globals([ResNet_ImageNet, ResNet_Cifar, Generator, Discriminator, 
                  BasicBlock, Bottleneck, ClassifierMLP, ModelCNN])
print("已註冊安全類")

# 定義數據轉換
def get_transforms():
    mean_values = [0.485, 0.456, 0.406]
    std_values = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values, std=std_values)
    ])
    return transform

# 加載基礎模型
def load_base_models(models_dir):
    base_models = []
    model_paths = []
    
    print(f"開始在 {models_dir} 中搜尋模型...")
    
    # 尋找所有模型文件
    for i in range(11):  # 任務 0 到 10
        model_path = os.path.join(models_dir, f"task_{str(i).zfill(2)}_200_model.pkl")
        if os.path.exists(model_path):
            model_paths.append(model_path)
            print(f"找到模型文件: {model_path}")
        else:
            print(f"未找到模型文件: {model_path}")
    
    # 如果沒有找到任何模型，嘗試列出目錄內容
    if len(model_paths) == 0:
        print(f"搜尋規則未找到任何模型，列出目錄內容:")
        try:
            all_files = os.listdir(models_dir)
            for file in all_files:
                if file.endswith(".pkl"):
                    print(f"  發現 .pkl 文件: {os.path.join(models_dir, file)}")
        except Exception as e:
            print(f"無法列出目錄內容: {e}")
    
    print(f"找到 {len(model_paths)} 個基礎模型")
    
    # 檢查是否有模型
    if len(model_paths) == 0:
        print("沒有找到任何模型文件，程序將退出")
        return []
    
    # 載入模型
    for path in tqdm(model_paths, desc="加載基礎模型"):
        try:
            print(f"嘗試載入模型: {path}")
            # 使用weights_only=False以確保加載完整模型
            model = torch.load(path, weights_only=False)
            model.cuda()
            model.eval()  # 設置為評估模式
            base_models.append(model)
            print(f"成功加載模型: {path}")
        except Exception as e:
            print(f"加載模型失敗 {path}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"成功載入 {len(base_models)} 個模型")
    return base_models

# 定義元模型 (Meta-Model)
class StackingMetaModel(nn.Module):
    def __init__(self, num_models, num_classes=1000):
        super(StackingMetaModel, self).__init__()
        self.num_models = num_models
        self.num_classes = num_classes
        
        # 每個模型對每個類別的權重
        self.weights = nn.Parameter(torch.ones(num_models, num_classes))
        
        # 初始化權重，讓第0個模型在前500類有較高權重，其他模型在各自範圍有較高權重
        with torch.no_grad():
            # 第0個模型在前500類有適中權重
            self.weights[0, :500] = 1.5
            
            # 其他模型各自負責後500類的特定區域，但給予更高的初始權重
            classes_per_task = 50
            for i in range(1, num_models):
                start_idx = 500 + (i-1) * classes_per_task
                end_idx = min(500 + i * classes_per_task, num_classes)
                self.weights[i, start_idx:end_idx] = 3.0  # 更高的初始權重，提升後500類表現
    
    def forward(self, predictions):
        """
        輸入: predictions - List of tensors, each with shape [batch_size, num_classes]
        輸出: Combined predictions with shape [batch_size, num_classes]
        """
        # 堆疊所有預測 [batch_size, num_models, num_classes]
        stacked = torch.stack(predictions, dim=1)
        
        # 直接使用權重進行線性加權，不使用softmax
        weighted_preds = (stacked * self.weights.unsqueeze(0)).sum(dim=1)
        
        return weighted_preds

# 支持置信度特徵的改進Meta Model
class StackingMetaModelWithConfidence(nn.Module):
    def __init__(self, num_models, num_classes=1000, actual_model_output_dim=None):
        super(StackingMetaModelWithConfidence, self).__init__()
        self.num_models = num_models
        self.num_classes = num_classes
        
        # 動態檢測實際模型輸出維度
        if actual_model_output_dim is None:
            # 根據檢查結果，所有Base Model的embed層都是1000維
            print("✅ 使用標準配置：Base Model輸出維度 = 1000")
            self.actual_model_dim = num_classes  # 默認1000
        else:
            self.actual_model_dim = actual_model_output_dim
            print(f"✅ 使用檢測到的Base Model輸出維度: {actual_model_output_dim}")
            
        # 每個模型貢獻: 預測(actual_dim) + 置信度(1) + 熵(1) + 一致性(1)
        input_dim_per_model = self.actual_model_dim + 3
        total_input_dim = num_models * input_dim_per_model
        
        print(f"✅ Enhanced Meta Model with confidence features")
        print(f"   📊 Detected model output dim: {self.actual_model_dim}")
        print(f"   📊 Input per model: {input_dim_per_model} (predictions + 3 confidence features)")
        print(f"   📊 Total input dim: {total_input_dim}")
        
        # 根據實際輸入維度調整網路架構
        if total_input_dim > 8000:
            # 大輸入維度架構
            hidden_dims = [2048, 1024, 512]
        elif total_input_dim > 4000:
            # 中等輸入維度架構  
            hidden_dims = [1024, 512, 256]
        else:
            # 小輸入維度架構
            hidden_dims = [512, 256, 128]
        
        self.meta_network = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dims[2], num_classes)
        )
        
        print(f"   🏗️ Architecture: {total_input_dim} -> {hidden_dims[0]} -> {hidden_dims[1]} -> {hidden_dims[2]} -> {num_classes}")
    
    def forward(self, enhanced_predictions):
        """
        輸入: enhanced_predictions - List of dicts, each containing:
               {'predictions': tensor, 'confidence': tensor, 'entropy': tensor, 'agreement': tensor}
        """
        batch_size = enhanced_predictions[0]['predictions'].size(0)
        combined_features = []
        
        for model_output in enhanced_predictions:
            predictions = model_output['predictions']  # [batch_size, num_classes]
            confidence = model_output['confidence'].unsqueeze(1)  # [batch_size, 1]
            entropy = model_output['entropy'].unsqueeze(1)  # [batch_size, 1]
            agreement = model_output['agreement'].unsqueeze(1)  # [batch_size, 1]
            
            # 連接預測和置信度特徵 [batch_size, num_classes + 3]
            model_features = torch.cat([predictions, confidence, entropy, agreement], dim=1)
            combined_features.append(model_features)
        
        # 合併所有模型的特徵 [batch_size, num_models * (num_classes + 3)]
        input_features = torch.cat(combined_features, dim=1)
        
        # 通過神經網路得到最終預測
        output = self.meta_network(input_features)
        return output

# 收集基礎模型的預測
def collect_predictions(models, dataloader):
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="收集預測")):
            images = images.cuda()
            batch_predictions = []
            
            # 獲取每個模型的預測
            for model in models:
                features = model(images)
                logits = model.embed(features)
                batch_predictions.append(logits)
            
            all_predictions.append(batch_predictions)
            all_labels.append(labels)
    
    # 整理數據格式
    X = []
    for i in range(len(models)):
        model_preds = []
        for batch_idx in range(len(all_predictions)):
            model_preds.append(all_predictions[batch_idx][i])
        X.append(torch.cat(model_preds, dim=0))
    
    X = [x.cpu() for x in X]  # 將預測移到CPU以節省GPU內存
    y = torch.cat(all_labels, dim=0).long()  # 確保標籤是長整型
    
    return X, y

# 收集基礎模型的預測和置信度特徵
def collect_predictions_with_confidence(models, dataloader):
    """
    收集Base Model預測並計算三種置信度特徵
    返回格式: List of dicts, each containing predictions and confidence features
    """
    all_enhanced_predictions = [[] for _ in range(len(models))]
    all_labels = []
    
    print("🔍 Collecting predictions with confidence features...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="收集增強預測")):
            images, labels = images.cuda(), labels.cuda()
            
            # 收集所有模型對當前batch的預測
            batch_predictions = []
            batch_raw_predictions = []
            
            for model in models:
                # 正確的兩階段調用：特徵提取 + 分類
                features = model(images)  # 提取特徵 (512維)
                outputs = model.embed(features)  # 分類輸出 (1000維)
                probabilities = F.softmax(outputs, dim=1)
                batch_predictions.append(torch.argmax(probabilities, dim=1))  # 預測類別
                batch_raw_predictions.append(probabilities)  # 概率分佈
            
            # 為每個模型計算置信度特徵
            for model_idx in range(len(models)):
                model_probs = batch_raw_predictions[model_idx]  # [batch_size, num_classes]
                
                # 1. 最大概率置信度
                confidence_scores = torch.max(model_probs, dim=1)[0]  # [batch_size]
                
                # 2. 預測熵 (不確定性)
                epsilon = 1e-8
                entropy_scores = -torch.sum(model_probs * torch.log(model_probs + epsilon), dim=1)  # [batch_size]
                
                # 3. 模型間一致性 (對當前樣本，所有模型預測的一致程度)
                batch_size = model_probs.size(0)
                agreement_scores = torch.zeros(batch_size).cuda()
                
                for sample_idx in range(batch_size):
                    # 當前樣本所有模型的預測類別
                    sample_predictions = [batch_predictions[m][sample_idx].item() for m in range(len(models))]
                    # 計算最常見預測的占比
                    from collections import Counter
                    most_common_count = Counter(sample_predictions).most_common(1)[0][1]
                    agreement_scores[sample_idx] = most_common_count / len(models)
                
                # 組織該模型的輸出
                model_output = {
                    'predictions': model_probs.cpu(),  # [batch_size, num_classes]
                    'confidence': confidence_scores.cpu(),  # [batch_size]
                    'entropy': entropy_scores.cpu(),  # [batch_size]
                    'agreement': agreement_scores.cpu()  # [batch_size]
                }
                
                all_enhanced_predictions[model_idx].append(model_output)
            
            all_labels.append(labels.cpu())
    
    # 合併所有批次的結果
    final_enhanced_predictions = []
    
    for model_idx in range(len(models)):
        # 合併該模型所有批次的預測和特徵
        model_predictions = torch.cat([batch['predictions'] for batch in all_enhanced_predictions[model_idx]], dim=0)
        model_confidences = torch.cat([batch['confidence'] for batch in all_enhanced_predictions[model_idx]], dim=0)
        model_entropies = torch.cat([batch['entropy'] for batch in all_enhanced_predictions[model_idx]], dim=0)
        model_agreements = torch.cat([batch['agreement'] for batch in all_enhanced_predictions[model_idx]], dim=0)
        
        final_enhanced_predictions.append({
            'predictions': model_predictions,
            'confidence': model_confidences,
            'entropy': model_entropies,
            'agreement': model_agreements
        })
    
    # 合併所有標籤
    y = torch.cat(all_labels, dim=0).long()
    
    print(f"✅ Enhanced predictions collected:")
    print(f"   📊 Models: {len(final_enhanced_predictions)}")
    print(f"   📊 Samples: {len(y)}")
    print(f"   📊 Features per model: predictions + confidence + entropy + agreement")
    
    # 輸出置信度特徵的統計信息
    print(f"\n📈 Confidence Features Statistics:")
    for i, model_data in enumerate(final_enhanced_predictions):
        avg_conf = model_data['confidence'].mean().item()
        avg_entropy = model_data['entropy'].mean().item()
        avg_agreement = model_data['agreement'].mean().item()
        print(f"   Model {i+1:2d}: Conf={avg_conf:.3f}, Entropy={avg_entropy:.3f}, Agreement={avg_agreement:.3f}")
    
    return final_enhanced_predictions, y

# 動態檢測Base Model的實際輸出維度
def detect_model_output_dim(models, dataloader):
    """
    檢測Base Model的實際分類輸出維度
    """
    print("🔍 動態檢測Base Model分類輸出維度...")
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.cuda()
            # 正確的檢測方式：特徵提取 + 分類層
            features = models[0](images)  # 提取特徵 (通常是512維)
            logits = models[0].embed(features)  # 分類輸出 (1000維)
            actual_dim = logits.size(1)  # [batch_size, num_classes]
            
            print(f"   📊 特徵維度: {features.size(1)}")
            print(f"   📊 檢測到Base Model分類輸出維度: {actual_dim}")
            return actual_dim
    
    # 如果無法檢測，返回默認值
    print("   ⚠️ 無法檢測輸出維度，使用默認值 1000")
    return 1000

# 繪製訓練過程曲線
def plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, 
                        learning_rates, output_dir, best_acc):
    """
    繪製Meta Model訓練過程的詳細視覺化圖表
    """
    epochs = range(1, len(train_losses) + 1)
    
    # 創建包含4個子圖的大圖
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 設置整體標題
    fig.suptitle(f'Meta Model Training Progress (Best Val Acc: {best_acc:.2f}%)', 
                fontsize=16, fontweight='bold')
    
    # 1. 損失函數曲線
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    ax1.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 標記最低驗證loss
    min_val_loss_idx = val_losses.index(min(val_losses))
    ax1.annotate(f'Min Val Loss\n{min(val_losses):.4f}', 
                xy=(min_val_loss_idx + 1, min(val_losses)),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 2. 準確率曲線
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2, alpha=0.8)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2, alpha=0.8)
    ax2.set_title('Accuracy Curves', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 標記最高驗證accuracy
    max_val_acc_idx = val_accuracies.index(max(val_accuracies))
    ax2.annotate(f'Best Accuracy\n{max(val_accuracies):.2f}%', 
                xy=(max_val_acc_idx + 1, max(val_accuracies)),
                xytext=(10, -15), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 3. 學習率變化
    ax3.plot(epochs, learning_rates, 'g-', linewidth=2, alpha=0.8)
    ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')  # 使用對數刻度更好顯示學習率變化
    ax3.grid(True, alpha=0.3)
    
    # 4. 訓練收斂性分析
    # 計算移動平均來觀察收斂趨勢
    window = min(10, len(val_accuracies) // 10)  # 窗口大小
    if window > 1:
        val_acc_smooth = []
        for i in range(len(val_accuracies)):
            start_idx = max(0, i - window + 1)
            val_acc_smooth.append(sum(val_accuracies[start_idx:i+1]) / (i - start_idx + 1))
        
        ax4.plot(epochs, val_accuracies, 'lightcoral', alpha=0.5, label='Raw Validation Accuracy')
        ax4.plot(epochs, val_acc_smooth, 'red', linewidth=2, label=f'Moving Average (window={window})')
    else:
        ax4.plot(epochs, val_accuracies, 'red', linewidth=2, label='Validation Accuracy')
    
    ax4.set_title('Convergence Analysis', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 添加收斂判斷
    if len(val_accuracies) > 20:
        last_20_std = np.std(val_accuracies[-20:])
        convergence_text = "Converged" if last_20_std < 0.5 else "May need more epochs"
        ax4.text(0.02, 0.98, f'Status: {convergence_text}\nLast 20 epochs std: {last_20_std:.3f}',
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 調整布局
    plt.tight_layout()
    
    # 保存圖表
    plot_path = os.path.join(output_dir, 'meta_model_training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"📊 Training visualization saved: {plot_path}")
    
    # 保存訓練數據到CSV
    import csv
    csv_path = os.path.join(output_dir, 'meta_model_training_log.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc(%)', 'Val_Loss', 'Val_Acc(%)', 'Learning_Rate'])
        for i in range(len(train_losses)):
            writer.writerow([i+1, f'{train_losses[i]:.6f}', f'{train_accuracies[i]:.2f}', 
                           f'{val_losses[i]:.6f}', f'{val_accuracies[i]:.2f}', f'{learning_rates[i]:.8f}'])
    
    print(f"📋 Training log saved: {csv_path}")

# 訓練元模型
def train_meta_model(meta_model, train_predictions, train_labels, val_predictions, val_labels, epochs, lr, output_dir=None):
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_acc = 0.0
    best_model_state = None
    
    # 記錄訓練過程的指標
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    
    # 將數據移到GPU並確保是長整型
    train_labels = train_labels.cuda().long()
    val_labels = val_labels.cuda().long()
    train_predictions = [p.cuda() for p in train_predictions]
    val_predictions = [p.cuda() for p in val_predictions]
    
    batch_size = 32
    num_samples = train_labels.size(0)
    indices = torch.randperm(num_samples)
    
    print(f"\n🚀 Starting Meta Model Training - {epochs} epochs")
    print("="*60)
    
    for epoch in range(epochs):
        meta_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 創建小批次
        for i in range(0, num_samples, batch_size):
            # 獲取批次索引
            batch_indices = indices[i:min(i+batch_size, num_samples)]
            
            # 提取批次數據
            batch_train_predictions = [p[batch_indices] for p in train_predictions]
            batch_train_labels = train_labels[batch_indices].long()  # 確保標籤是長整型
            
            # 前向傳播
            outputs = meta_model(batch_train_predictions)
            loss = criterion(outputs, batch_train_labels)
            
            # 反向傳播和優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 統計
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_train_labels.size(0)
            correct += predicted.eq(batch_train_labels).sum().item()
        
        # 計算訓練準確率
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        
        # 評估驗證集
        meta_model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # 創建小批次
            val_indices = torch.arange(len(val_labels))
            for i in range(0, len(val_labels), batch_size):
                batch_end = min(i+batch_size, len(val_labels))
                batch_indices = val_indices[i:batch_end]
                
                # 提取批次數據
                batch_val_predictions = [p[batch_indices] for p in val_predictions]
                batch_val_labels = val_labels[batch_indices].long()
                
                outputs = meta_model(batch_val_predictions)
                loss = criterion(outputs, batch_val_labels)
                
                val_loss += loss.item() * batch_val_labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == batch_val_labels).sum().item()
                val_total += batch_val_labels.size(0)
            
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total * 100
        
        # 記錄指標
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # 詳細的epoch輸出
        if (epoch + 1) % 10 == 0 or epoch < 5:  # 前5個epoch和每10個epoch輸出詳細信息
            print(f'Epoch {epoch+1:3d}/{epochs} | '
                  f'Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}% | '
                  f'Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}% | '
                  f'LR={current_lr:.6f}')
        
        # 調整學習率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = meta_model.state_dict().copy()
            print(f'⭐ New best validation accuracy: {best_acc:.2f}% (Epoch {epoch+1})')
    
    # 恢復最佳模型
    meta_model.load_state_dict(best_model_state)
    
    print("="*60)
    print(f"🏆 Meta Model Training Completed! Best Validation Accuracy: {best_acc:.2f}%")
    
    # 生成訓練過程視覺化圖表
    if output_dir:
        plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, 
                           learning_rates, output_dir, best_acc)
    
    return meta_model, best_acc

# 訓練支持置信度特徵的Meta Model
def train_enhanced_meta_model(meta_model, train_enhanced_predictions, train_labels, 
                              val_enhanced_predictions, val_labels, epochs, lr, output_dir=None):
    """
    訓練支持置信度特徵的Enhanced Meta Model
    """
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=lr, weight_decay=1e-5)  # 添加L2正則化
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_acc = 0.0
    best_model_state = None
    
    # 記錄訓練過程的指標
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    
    # 將標籤移到GPU
    train_labels = train_labels.cuda().long()
    val_labels = val_labels.cuda().long()
    
    # 將置信度特徵移到GPU
    for model_data in train_enhanced_predictions:
        model_data['predictions'] = model_data['predictions'].cuda()
        model_data['confidence'] = model_data['confidence'].cuda()
        model_data['entropy'] = model_data['entropy'].cuda()
        model_data['agreement'] = model_data['agreement'].cuda()
    
    for model_data in val_enhanced_predictions:
        model_data['predictions'] = model_data['predictions'].cuda()
        model_data['confidence'] = model_data['confidence'].cuda()
        model_data['entropy'] = model_data['entropy'].cuda()
        model_data['agreement'] = model_data['agreement'].cuda()
    
    batch_size = 32
    num_samples = train_labels.size(0)
    
    print(f"\n🚀 Starting Enhanced Meta Model Training - {epochs} epochs")
    print("="*60)
    
    for epoch in range(epochs):
        meta_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 創建隨機批次索引
        indices = torch.randperm(num_samples)
        
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_indices = indices[i:batch_end]
            
            # 提取批次數據 - 置信度特徵版本
            batch_enhanced_predictions = []
            for model_data in train_enhanced_predictions:
                batch_model_data = {
                    'predictions': model_data['predictions'][batch_indices],
                    'confidence': model_data['confidence'][batch_indices],
                    'entropy': model_data['entropy'][batch_indices],
                    'agreement': model_data['agreement'][batch_indices]
                }
                batch_enhanced_predictions.append(batch_model_data)
            
            batch_train_labels = train_labels[batch_indices]
            
            # 前向傳播
            outputs = meta_model(batch_enhanced_predictions)
            loss = criterion(outputs, batch_train_labels)
            
            # 反向傳播和優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 統計
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_train_labels.size(0)
            correct += predicted.eq(batch_train_labels).sum().item()
        
        # 計算訓練準確率
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        
        # 評估驗證集
        meta_model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            val_indices = torch.arange(len(val_labels))
            for i in range(0, len(val_labels), batch_size):
                batch_end = min(i + batch_size, len(val_labels))
                batch_indices = val_indices[i:batch_end]
                
                # 提取驗證批次數據
                batch_val_enhanced_predictions = []
                for model_data in val_enhanced_predictions:
                    batch_model_data = {
                        'predictions': model_data['predictions'][batch_indices],
                        'confidence': model_data['confidence'][batch_indices],
                        'entropy': model_data['entropy'][batch_indices],
                        'agreement': model_data['agreement'][batch_indices]
                    }
                    batch_val_enhanced_predictions.append(batch_model_data)
                
                batch_val_labels = val_labels[batch_indices]
                
                outputs = meta_model(batch_val_enhanced_predictions)
                loss = criterion(outputs, batch_val_labels)
                
                val_loss += loss.item() * batch_val_labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == batch_val_labels).sum().item()
                val_total += batch_val_labels.size(0)
            
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total * 100
        
        # 記錄指標
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # 詳細的epoch輸出
        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f'Epoch {epoch+1:3d}/{epochs} | '
                  f'Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}% | '
                  f'Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}% | '
                  f'LR={current_lr:.6f}')
        
        # 調整學習率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = meta_model.state_dict().copy()
            print(f'⭐ New best validation accuracy: {best_acc:.2f}% (Epoch {epoch+1})')
    
    # 恢復最佳模型
    meta_model.load_state_dict(best_model_state)
    
    print("="*60)
    print(f"🏆 Enhanced Meta Model Training Completed! Best Validation Accuracy: {best_acc:.2f}%")
    
    # 生成訓練過程視覺化圖表
    if output_dir:
        plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, 
                           learning_rates, output_dir, best_acc)
    
    return meta_model, best_acc

# 評估置信度增強的Meta Model
def evaluate_enhanced_meta_model(meta_model, enhanced_predictions, test_labels, output_dir):
    """
    評估支持置信度特徵的Enhanced Meta Model
    """
    meta_model.eval()
    test_labels = test_labels.cuda().long()
    
    # 將置信度特徵移到GPU
    for model_data in enhanced_predictions:
        model_data['predictions'] = model_data['predictions'].cuda()
        model_data['confidence'] = model_data['confidence'].cuda()
        model_data['entropy'] = model_data['entropy'].cuda()
        model_data['agreement'] = model_data['agreement'].cuda()
    
    # 批次評估
    batch_size = 32
    num_samples = len(test_labels)
    all_predictions = []
    
    print(f"📊 評估 {num_samples} 個樣本（批次大小: {batch_size}）...")
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_indices = torch.arange(i, batch_end)
            
            # 提取批次增強預測數據
            batch_enhanced_predictions = []
            for model_data in enhanced_predictions:
                batch_model_data = {
                    'predictions': model_data['predictions'][batch_indices],
                    'confidence': model_data['confidence'][batch_indices],
                    'entropy': model_data['entropy'][batch_indices],
                    'agreement': model_data['agreement'][batch_indices]
                }
                batch_enhanced_predictions.append(batch_model_data)
            
            # Meta Model預測
            outputs = meta_model(batch_enhanced_predictions)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.append(predicted)
    
    # 合併所有預測
    all_predictions = torch.cat(all_predictions, dim=0)
    
    # 計算總體準確率
    correct = (all_predictions == test_labels).sum().item()
    total = test_labels.size(0)
    overall_accuracy = 100.0 * correct / total
    
    print(f"🎯 置信度增強Meta Model測試準確率: {overall_accuracy:.2f}%")
    
    # 計算各類別準確率
    class_correct = {}
    class_total = {}
    
    for i in range(len(test_labels)):
        label = test_labels[i].item()
        pred = all_predictions[i].item()
        
        if label not in class_total:
            class_total[label] = 0
            class_correct[label] = 0
        
        class_total[label] += 1
        if label == pred:
            class_correct[label] += 1
    
    # 分析前500類和後500類的準確率
    front_classes = [cls for cls in class_total.keys() if cls < 500]
    back_classes = [cls for cls in class_total.keys() if cls >= 500]
    
    front_acc = np.mean([100.0 * class_correct[cls] / class_total[cls] for cls in front_classes]) if front_classes else 0
    back_acc = np.mean([100.0 * class_correct[cls] / class_total[cls] for cls in back_classes]) if back_classes else 0
    
    print(f"📊 前500類平均準確率: {front_acc:.2f}%")
    print(f"📊 後500類平均準確率: {back_acc:.2f}%")
    
    # 繪製結果可視化（重用現有的繪圖邏輯）
    accuracies = []
    class_indices = []
    
    for cls in sorted(class_total.keys()):
        if class_total[cls] > 0:
            acc = 100.0 * class_correct[cls] / class_total[cls]
            accuracies.append(acc)
            class_indices.append(cls)
    
    # 繪製柱狀圖
    plt.figure(figsize=(12, 6))
    plt.bar(class_indices, accuracies, alpha=0.7)
    plt.axvline(x=500, color='r', linestyle='--', label='Class Boundary (500)')
    plt.xlabel('Class Index')
    plt.ylabel('Accuracy (%)')
    plt.title('Enhanced Meta Model: Per-Class Accuracy Distribution')
    plt.legend()
    plt.tight_layout()
    
    # 保存圖表
    plt.savefig(os.path.join(output_dir, 'enhanced_meta_model_class_accuracy.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 類別準確率圖表已保存: {output_dir}/enhanced_meta_model_class_accuracy.png")
    
    # 保存準確率報告
    report = {
        'overall_accuracy': overall_accuracy,
        'front_500_accuracy': front_acc,
        'back_500_accuracy': back_acc,
        'total_samples': total,
        'correct_predictions': correct,
        'num_classes_tested': len(class_total)
    }
    
    with open(os.path.join(output_dir, 'enhanced_meta_model_results.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📋 詳細報告已保存: {output_dir}/enhanced_meta_model_results.json")
    
    return overall_accuracy

# 評估元模型並繪製結果
def evaluate_and_visualize(meta_model, test_predictions, test_labels, output_dir):
    meta_model.eval()
    test_labels = test_labels.cuda()
    test_predictions = [p.cuda() for p in test_predictions]
    
    batch_size = 32
    with torch.no_grad():
        total_correct = 0
        total = test_labels.size(0)
        predictions = []
        
        # 創建小批次
        test_indices = torch.arange(total)
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch_indices = test_indices[i:batch_end]
            
            # 提取批次數據
            batch_test_predictions = [p[batch_indices] for p in test_predictions]
            batch_test_labels = test_labels[batch_indices]
            
            outputs = meta_model(batch_test_predictions)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == batch_test_labels).sum().item()
            predictions.append(predicted)
        
        predicted = torch.cat(predictions)
    
    test_acc = 100.0 * total_correct / total
    print(f'測試準確率: {test_acc:.2f}%')
    
    # 繪製混淆矩陣熱圖
    plt.figure(figsize=(12, 10))
    
    # 類別級別準確率分析
    class_correct = {}
    class_total = {}
    
    for i in range(total):
        label = test_labels[i].item()
        pred = predicted[i].item()
        
        if label not in class_total:
            class_total[label] = 0
            class_correct[label] = 0
        
        class_total[label] += 1
        if label == pred:
            class_correct[label] += 1
    
    # 繪製類別準確率
    accuracies = []
    class_indices = []
    
    for cls in sorted(class_total.keys()):
        if class_total[cls] > 0:
            acc = 100.0 * class_correct[cls] / class_total[cls]
            accuracies.append(acc)
            class_indices.append(cls)
    
    # 分析前500類和後500類的準確率
    front_classes = [cls for cls in class_indices if cls < 500]
    back_classes = [cls for cls in class_indices if cls >= 500]
    
    front_acc = np.mean([100.0 * class_correct[cls] / class_total[cls] for cls in front_classes]) if front_classes else 0
    back_acc = np.mean([100.0 * class_correct[cls] / class_total[cls] for cls in back_classes]) if back_classes else 0
    
    print(f'前500類平均準確率: {front_acc:.2f}%')
    print(f'後500類平均準確率: {back_acc:.2f}%')
    
    # 繪製柱狀圖
    plt.figure(figsize=(12, 6))
    plt.bar(class_indices, accuracies, alpha=0.7)
    plt.axvline(x=500, color='r', linestyle='--', label='類別分界線 (500)')
    plt.xlabel('類別索引')
    plt.ylabel('準確率 (%)')
    plt.title('Stacking集成模型各類別準確率')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'class_accuracies.png'))
    
    # 保存結果文本
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write(f'整體測試準確率: {test_acc:.2f}%\n')
        f.write(f'前500類平均準確率: {front_acc:.2f}%\n')
        f.write(f'後500類平均準確率: {back_acc:.2f}%\n')
    
    return test_acc

# 主程序
def main():
    # 加載基礎模型
    base_models = load_base_models(args.models_dir)
    if len(base_models) == 0:
        print("未能加載任何基礎模型，退出程序")
        return

    # 只評估模式
    if args.eval_only:
        print("只評估模式：載入已訓練好的集成模型...")
        meta_model_path = os.path.join(args.output_dir, 'stacking_meta_model.pkl')
        
        if not os.path.exists(meta_model_path):
            print(f"錯誤：找不到已訓練好的模型 {meta_model_path}")
            print("請先執行訓練，或確認輸出目錄路徑是否正確")
            return
            
        try:
            meta_model = torch.load(meta_model_path)
            meta_model.cuda()
            print(f"成功載入集成模型：{meta_model_path}")
        except Exception as e:
            print(f"載入模型失敗：{e}")
            return
        
        # 準備測試集
        transform = get_transforms()
        all_classes_index = list(range(1000))
        print(f"創建包含 {len(all_classes_index)} 個類別的索引")
        
        dataset = ImageFolder(args.data_dir, transform=transform, index=all_classes_index)
        print(f"數據集總大小: {len(dataset)}")
        
        # 創建數據加載器
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # 收集基礎模型在測試集上的預測
        print("收集基礎模型在測試集上的預測...")
        test_predictions, test_labels = collect_predictions(base_models, test_loader)
        
        # 評估元模型並繪製結果
        print("評估中...")
        test_acc = evaluate_and_visualize(meta_model, test_predictions, test_labels, args.output_dir)
        
        print(f"評估完成！測試準確率: {test_acc:.2f}%")
        return
    
    # 以下是原來的訓練模式
    # 準備數據集
    transform = get_transforms()
    
    # 創建一個包含所有類別的索引（0-999）
    all_classes_index = list(range(1000))  # 假設有1000個類別
    print(f"創建包含 {len(all_classes_index)} 個類別的索引")
    
    # 加載數據集，提供有效的索引參數
    dataset = ImageFolder(args.data_dir, transform=transform, index=all_classes_index)
    
    # 分割數據集為元模型訓練集和測試集
    train_size = int(args.val_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"數據集總大小: {len(dataset)}")
    print(f"元模型訓練集大小: {len(train_dataset)}")
    print(f"測試集大小: {len(test_dataset)}")
    
    # 創建數據加載器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 收集基礎模型在訓練集和測試集上的預測
    print("收集基礎模型在訓練集上的預測...")
    train_predictions, train_labels = collect_predictions(base_models, train_loader)
    
    print("收集基礎模型在測試集上的預測...")
    test_predictions, test_labels = collect_predictions(base_models, test_loader)
    
    # 再次分割訓練集以獲得驗證集
    val_size = int(0.2 * len(train_labels))
    train_idx, val_idx = train_test_split(
        range(len(train_labels)), test_size=val_size/len(train_labels), random_state=42
    )
    
    # 準備驗證集
    val_predictions = [p[val_idx] for p in train_predictions]
    val_labels = train_labels[val_idx]
    
    # 調整訓練集
    train_predictions = [p[train_idx] for p in train_predictions]
    train_labels = train_labels[train_idx]
    
    print(f"最終元模型訓練集大小: {len(train_labels)}")
    print(f"驗證集大小: {len(val_labels)}")
    
    # 檢查是否使用置信度特徵
    if args.use_confidence:
        print("\n🚀 使用置信度特徵增強的Meta Model")
        print("="*70)
        
        # 動態檢測Base Model的實際輸出維度
        actual_dim = detect_model_output_dim(base_models, train_loader)
        
        # 重新收集帶置信度特徵的預測
        print("重新收集帶置信度特徵的訓練集預測...")
        train_enhanced_predictions, _ = collect_predictions_with_confidence(base_models, train_loader)
        
        print("重新收集帶置信度特徵的測試集預測...")
        test_enhanced_predictions, _ = collect_predictions_with_confidence(base_models, test_loader)
        
        # 分割增強預測數據
        val_enhanced_predictions = []
        for model_data in train_enhanced_predictions:
            val_enhanced_predictions.append({
                'predictions': model_data['predictions'][val_idx],
                'confidence': model_data['confidence'][val_idx],
                'entropy': model_data['entropy'][val_idx],
                'agreement': model_data['agreement'][val_idx]
            })
        
        # 調整訓練集增強預測
        for i, model_data in enumerate(train_enhanced_predictions):
            train_enhanced_predictions[i] = {
                'predictions': model_data['predictions'][train_idx],
                'confidence': model_data['confidence'][train_idx],
                'entropy': model_data['entropy'][train_idx],
                'agreement': model_data['agreement'][train_idx]
            }
        
        # 創建和訓練置信度增強Meta Model (使用檢測到的實際維度)
        meta_model = StackingMetaModelWithConfidence(
            num_models=len(base_models), 
            num_classes=1000,
            actual_model_output_dim=actual_dim
        ).cuda()
        
        meta_model, best_val_acc = train_enhanced_meta_model(
            meta_model, train_enhanced_predictions, train_labels,
            val_enhanced_predictions, val_labels, args.epochs, args.lr, args.output_dir
        )
        
        # 評估置信度增強模型（需要使用增強預測數據）
        print("🔍 在測試集上評估置信度增強模型...")
        test_acc = evaluate_enhanced_meta_model(meta_model, test_enhanced_predictions, test_labels, args.output_dir)
        
        # 保存置信度增強模型
        model_save_path = os.path.join(args.output_dir, 'stacking_meta_model_with_confidence.pkl')
        torch.save(meta_model, model_save_path)
        print(f"置信度增強元模型已保存到: {model_save_path}")
        
    else:
        print("\n📊 使用傳統Stacking方法")
        print("="*50)
        
        # 創建和訓練傳統元模型
        meta_model = StackingMetaModel(num_models=len(base_models)).cuda()
        meta_model, best_val_acc = train_meta_model(
            meta_model, train_predictions, train_labels, 
            val_predictions, val_labels, args.epochs, args.lr, args.output_dir
        )
        
        # 評估元模型並繪製結果
        test_acc = evaluate_and_visualize(meta_model, test_predictions, test_labels, args.output_dir)
        
        # 保存傳統模型
        model_save_path = os.path.join(args.output_dir, 'stacking_meta_model.pkl')
        torch.save(meta_model, model_save_path)
        print(f"元模型已保存到: {model_save_path}")
    
    print(f"🏆 最佳驗證準確率: {best_val_acc:.2f}%")
    print(f"📊 測試準確率: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
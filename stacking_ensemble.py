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
        
        # 溫度參數控制權重分佈的軟硬程度
        self.temperature = nn.Parameter(torch.tensor(2.0))
        
        # 前500類與後500類的整體權重平衡參數
        self.front_weight = nn.Parameter(torch.tensor(1.0))  # 前500類權重
        self.back_weight = nn.Parameter(torch.tensor(1.0))   # 後500類權重
        
        # 初始化權重，讓第0個模型在前500類有較高權重，其他模型在各自範圍有較高權重
        with torch.no_grad():
            # 第0個模型在前500類有較高權重
            self.weights[0, :500] = 2.0
            
            # 其他模型各自負責約50個類別
            classes_per_task = 50
            for i in range(1, num_models):
                start_idx = 500 + (i-1) * classes_per_task
                end_idx = min(500 + i * classes_per_task, num_classes)
                self.weights[i, start_idx:end_idx] = 2.0
    
    def forward(self, predictions):
        """
        輸入: predictions - List of tensors, each with shape [batch_size, num_classes]
        輸出: Combined predictions with shape [batch_size, num_classes]
        """
        # 堆疊所有預測 [batch_size, num_models, num_classes]
        stacked = torch.stack(predictions, dim=1)
        batch_size = stacked.size(0)
        
        # 計算權重 - 混合線性加權和softmax
        normalized_weights = F.softmax(self.weights / self.temperature, dim=0)
        
        # 套用前500類與後500類的不同權重
        balance_weights = torch.ones_like(normalized_weights)
        balance_weights[:, :500] *= self.front_weight  # 前500類套用front_weight
        balance_weights[:, 500:] *= self.back_weight   # 後500類套用back_weight
        
        # 最終權重結合softmax正規化和線性加權
        final_weights = normalized_weights * balance_weights
        
        # 重新正規化確保權重有合理比例
        final_weights = final_weights / final_weights.sum(dim=0, keepdim=True)
        final_weights = final_weights.unsqueeze(0)  # 添加batch維度 [1, num_models, num_classes]
        
        # 加權平均所有預測 [batch_size, num_classes]
        weighted_preds = (stacked * final_weights).sum(dim=1)
        
        return weighted_preds

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

# 訓練元模型
def train_meta_model(meta_model, train_predictions, train_labels, val_predictions, val_labels, epochs, lr):
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_acc = 0.0
    best_model_state = None
    
    # 將數據移到GPU並確保是長整型
    train_labels = train_labels.cuda().long()
    val_labels = val_labels.cuda().long()
    train_predictions = [p.cuda() for p in train_predictions]
    val_predictions = [p.cuda() for p in val_predictions]
    
    batch_size = 32
    num_samples = train_labels.size(0)
    indices = torch.randperm(num_samples)
    
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
            print(f'Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 調整學習率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = meta_model.state_dict().copy()
            print(f'新的最佳驗證準確率: {best_acc:.2f}%')
    
    # 恢復最佳模型
    meta_model.load_state_dict(best_model_state)
    return meta_model, best_acc

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
    
    # 創建和訓練元模型
    meta_model = StackingMetaModel(num_models=len(base_models)).cuda()
    meta_model, best_val_acc = train_meta_model(
        meta_model, train_predictions, train_labels, 
        val_predictions, val_labels, args.epochs, args.lr
    )
    
    # 評估元模型並繪製結果
    test_acc = evaluate_and_visualize(meta_model, test_predictions, test_labels, args.output_dir)
    
    # 保存元模型
    torch.save(meta_model, os.path.join(args.output_dir, 'stacking_meta_model.pkl'))
    print(f"元模型已保存到: {os.path.join(args.output_dir, 'stacking_meta_model.pkl')}")
    print(f"最佳驗證準確率: {best_val_acc:.2f}%")
    print(f"測試準確率: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
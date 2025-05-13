# stacking_predict.py
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from ImageFolder import ImageFolder
from torch.serialization import add_safe_globals
from models.resnet import ResNet_ImageNet, ResNet_Cifar, Generator, Discriminator, BasicBlock, Bottleneck, ClassifierMLP, ModelCNN
from tqdm import tqdm

# 命令行參數
parser = argparse.ArgumentParser(description='Stacking Ensemble Prediction')
parser.add_argument('-base_models_dir', type=str, default='checkpoints/medicine_10tasks', help='Directory containing base models')
parser.add_argument('-meta_model_path', type=str, default='ensemble_results/stacking_meta_model.pkl', help='Path to the trained meta-model')
parser.add_argument('-data_dir', type=str, default='medicine_picture/valid', help='Directory containing test data')
parser.add_argument('-batch_size', type=int, default=32, help='Batch size for prediction')
parser.add_argument('-gpu', type=str, default='0', help='GPU to use')
args = parser.parse_args()

# 設置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# 註冊安全類，以便加載模型
add_safe_globals([ResNet_ImageNet, ResNet_Cifar, Generator, Discriminator, 
                  BasicBlock, Bottleneck, ClassifierMLP, ModelCNN])

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
    
    # 尋找所有模型文件
    for i in range(11):  # 任務 0 到 10
        model_path = os.path.join(models_dir, f"task_{str(i).zfill(2)}_200_model.pkl")
        if os.path.exists(model_path):
            model_paths.append(model_path)
    
    print(f"找到 {len(model_paths)} 個基礎模型")
    
    # 載入模型
    for path in tqdm(model_paths, desc="加載基礎模型"):
        try:
            model = torch.load(path, weights_only=False)
            model.cuda()
            model.eval()
            base_models.append(model)
            print(f"成功加載模型: {path}")
        except Exception as e:
            print(f"加載模型失敗 {path}: {e}")
    
    return base_models

# 預測函數
def predict(base_models, meta_model, test_loader):
    all_predictions = []
    all_labels = []
    
    meta_model.eval()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="預測中"):
            images = images.cuda()
            
            # 獲取基礎模型預測
            base_predictions = []
            for model in base_models:
                features = model(images)
                logits = model.embed(features)
                base_predictions.append(logits)
            
            # 元模型預測
            outputs = meta_model(base_predictions)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 計算整體準確率
    correct = sum(p == l for p, l in zip(all_predictions, all_labels))
    total = len(all_labels)
    accuracy = 100.0 * correct / total
    
    # 計算各類別準確率
    class_correct = {}
    class_total = {}
    
    for pred, label in zip(all_predictions, all_labels):
        if label not in class_total:
            class_total[label] = 0
            class_correct[label] = 0
        
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    # 分析前500類和後500類的準確率
    front_classes = [cls for cls in class_total.keys() if cls < 500]
    back_classes = [cls for cls in class_total.keys() if cls >= 500]
    
    front_acc = sum(class_correct[cls] for cls in front_classes) / sum(class_total[cls] for cls in front_classes) * 100 if front_classes else 0
    back_acc = sum(class_correct[cls] for cls in back_classes) / sum(class_total[cls] for cls in back_classes) * 100 if back_classes else 0
    
    print(f"測試集總數: {total}")
    print(f"整體準確率: {accuracy:.2f}%")
    print(f"前500類準確率: {front_acc:.2f}%")
    print(f"後500類準確率: {back_acc:.2f}%")
    
    return accuracy

def main():
    # 加載基礎模型
    base_models = load_base_models(args.base_models_dir)
    if len(base_models) == 0:
        print("未能加載任何基礎模型，退出程序")
        return
    
    # 加載元模型
    try:
        meta_model = torch.load(args.meta_model_path)
        meta_model.cuda()
        meta_model.eval()
        print(f"成功加載元模型: {args.meta_model_path}")
    except Exception as e:
        print(f"加載元模型失敗 {args.meta_model_path}: {e}")
        return
    
    # 準備測試數據
    transform = get_transforms()
    test_dataset = ImageFolder(args.data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 進行預測
    accuracy = predict(base_models, meta_model, test_loader)
    print(f"預測完成！整體準確率: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
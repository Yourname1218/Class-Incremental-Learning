# coding=utf-8
from __future__ import absolute_import, print_function
import argparse

import torch
from torch.backends import cudnn
from evaluations import extract_features, pairwise_distance, extract_features_classification
from evaluations import Recall_at_ks, NMI, Recall_at_ks_products
import os
import numpy as np
from utils import to_numpy
from torch.nn import functional as F
import torchvision.transforms as transforms
from ImageFolder import *
from utils import *
from sklearn.metrics.pairwise import euclidean_distances
from scipy.special import softmax
from CIFAR100 import CIFAR100
from models.resnet import ResNet_Cifar, ResNet_ImageNet, BasicBlock, Bottleneck
from torch.serialization import add_safe_globals

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
plt.switch_backend('agg')  # 使用非交互式後端

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='PyTorch Testing')

parser.add_argument('-data', type=str, default='cifar100')
parser.add_argument('-r', type=str, default='model.pkl', metavar='PATH')
parser.add_argument('-name', type=str, default='tmp', metavar='PATH')

parser.add_argument("-gpu", type=str, default='0', help='which gpu to choose')
parser.add_argument('-seed', default=1993, type=int, metavar='N',
                    help='seeds for training process')
parser.add_argument('-epochs', default=600, type=int, metavar='N', help='epochs for training process')
parser.add_argument('-num_task', type=int, default=2, help="learning rate of new parameters")
parser.add_argument('-nb_cl_fg', type=int, default=50, help="learning rate of new parameters")

parser.add_argument('-num_class', type=int, default=100, help="learning rate of new parameters")
parser.add_argument('-dir', default='/data/datasets/featureGeneration/',
                        help='data dir')
parser.add_argument('-top5', action = 'store_true', help='output top5 accuracy')

args = parser.parse_args()
cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
models = []
for i in os.listdir(args.r):
    if i.endswith("%d_model.pkl" % (args.epochs - 1)):  # 500_model.pkl
        models.append(os.path.join(args.r, i))

models.sort()

if 'cifar' in args.data:
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    testdir = args.dir + '/cifar'

if args.data == 'imagenet_sub' or args.data == 'imagenet_full':
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        transform_test = transforms.Compose([
            #transforms.Resize(256),
            transforms.CenterCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values)
        ])
        testdir = os.path.join(args.dir, 'ILSVRC12_256', 'val')

if args.data == 'medicine':
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        transform_test = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values)
        ])
        testdir = os.path.join('medicine_picture', 'valid')

num_classes = args.num_class
num_task = args.num_task
num_class_per_task = (num_classes -  args.nb_cl_fg) // num_task

np.random.seed(args.seed)
#random_perm = np.random.permutation(num_classes)
random_perm = list(range(num_classes))

print('Test starting -->\t')
acc_all = np.zeros((num_task+3, num_task+1), dtype = 'float') # Save for csv

# 註冊安全的全局類
add_safe_globals([ResNet_Cifar, ResNet_ImageNet, BasicBlock, Bottleneck])

def load_model_safely(model_path):
    try:
        # 直接使用 weights_only=False 加載模型
        model = torch.load(model_path, weights_only=False)
    except Exception as e:
        print(f"Error loading model with weights_only=False: {e}")
        try:
            # 如果失敗，嘗試其它方法
            model = torch.load(model_path, map_location='cpu', weights_only=False)
        except Exception as e2:
            print(f"All loading methods failed: {e2}")
            raise
    return model

def test_task(args, test_loader, current_task, model):
    # 初始化測試數據收集器
    test_data = {
        'tasks': [],
        'accuracy': [],
        'class_accuracies': {}
    }
    
    model.eval()
    correct = 0
    total = 0
    
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            
            # 使用 forward 而不是 predict
            features = model(images)  # 獲取特徵
            outputs = model.embed(features)  # 使用 embed 層進行分類
            
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 收集每個類別的準確率
            for label, pred in zip(labels, predicted):
                label_item = label.item()
                if label_item not in class_correct:
                    class_correct[label_item] = 0
                    class_total[label_item] = 0
                class_total[label_item] += 1
                if label_item == pred.item():
                    class_correct[label_item] += 1
    
    accuracy = 100 * correct / total
    
    # 更新測試數據
    test_data['tasks'].append(current_task)
    test_data['accuracy'].append(accuracy)
    
    # 計算每個類別的準確率
    for class_id in class_total.keys():
        if class_id not in test_data['class_accuracies']:
            test_data['class_accuracies'][class_id] = []
        class_acc = 100 * class_correct[class_id] / class_total[class_id]
        test_data['class_accuracies'][class_id].append(class_acc)
    
    # 繪製測試結果圖表
    plot_test_results(test_data, args.log_dir, current_task)
    
    return accuracy, test_data

def plot_test_results(test_data, save_path, current_task):
    """繪製測試結果的圖表"""
    plt.figure(figsize=(15, 10))
    
    # 創建子圖
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    
    # 1. 整體準確率趨勢圖
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(test_data['tasks'], test_data['accuracy'], 
             marker='o', linewidth=2, markersize=8)
    ax1.set_title('Overall Accuracy Trend')
    ax1.set_xlabel('Task')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True)
    
    # 2. 每個類別的準確率熱圖
    ax2 = plt.subplot(gs[0, 1])
    class_acc_data = []
    for class_id in sorted(test_data['class_accuracies'].keys()):
        class_acc_data.append(test_data['class_accuracies'][class_id])
    
    sns.heatmap(class_acc_data, 
                ax=ax2, 
                cmap='YlOrRd', 
                cbar_kws={'label': 'Accuracy (%)'})
    ax2.set_title('Class-wise Accuracy Heatmap')
    ax2.set_xlabel('Task')
    ax2.set_ylabel('Class ID')
    
    # 3. 當前任務的性能摘要
    ax3 = plt.subplot(gs[1, :])
    current_acc = test_data['accuracy'][-1]
    ax3.text(0.5, 0.5, 
             f'Current Task: {current_task}\n'
             f'Current Accuracy: {current_acc:.2f}%\n'
             f'Average Accuracy: {np.mean(test_data["accuracy"]):.2f}%',
             ha='center', va='center', fontsize=12)
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'test_results_task_{current_task}.png'))
    plt.close()

if __name__ == '__main__':
    # 添加多進程支持
    import multiprocessing
    multiprocessing.freeze_support()
    
    # 移除重複的參數解析器定義，只保留一個
    parser = argparse.ArgumentParser(description='PyTorch Testing')
    
    # task setting
    parser.add_argument('-data', type=str, default='medicine')
    parser.add_argument('-num_task', type=int, default=2, help="Number of tasks")
    parser.add_argument('-nb_cl_fg', type=int, default=500, help="Number of classes in first group")
    parser.add_argument('-num_class', type=int, default=1000, help="Total number of classes")
    
    # basic parameters
    parser.add_argument('-r', type=str, default='model.pkl', metavar='PATH', help='Path to load models')
    parser.add_argument('-name', type=str, default='tmp', metavar='PATH', help='Name for saving results')
    parser.add_argument('-dir', default='/data/datasets/featureGeneration/', help='Data directory')
    parser.add_argument("-gpu", type=str, default='0', help='Which GPU to use')
    
    # other parameters
    parser.add_argument('-seed', default=1993, type=int, help='Random seed')
    parser.add_argument('-epochs', default=201, type=int, help='Number of epochs')
    parser.add_argument('-top5', action='store_true', help='Report top-5 accuracy')
    # 添加特徵維度參數，默認為 ResNet50 的 2048
    parser.add_argument('-feat_dim', type=int, default=2048, help="Feature dimension")
    parser.add_argument('-hidden_dim', type=int, default=512, help="Hidden dimension")
    
    args = parser.parse_args()
    
    # 創建結果保存目錄
    results_dir = os.path.join('results', 'test_plots', args.name)
    os.makedirs(results_dir, exist_ok=True)
    args.log_dir = results_dir
    
    # 設置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Test starting
    print('Test starting -->\t')
    acc_all = np.zeros((num_task+3, num_task+1), dtype = 'float') # Save for csv
    
    for task_id in range(num_task+1):
        if task_id == 0:
            index = random_perm[:args.nb_cl_fg]
        else:
            index = random_perm[:args.nb_cl_fg + (task_id) * num_class_per_task]
            
        # 設置數據加載器
        if args.data == 'cifar100':
            np.random.seed(args.seed)
            target_transform = np.random.permutation(num_classes)
            testset = CIFAR100(root=testdir, train=False, download=True, 
                             transform=transform_test, 
                             target_transform=target_transform, 
                             index=index)
            # 修改 num_workers=0 來避免多進程問題
            test_loader = torch.utils.data.DataLoader(
                testset, 
                batch_size=128, 
                shuffle=False, 
                num_workers=0,  # 改為 0
                drop_last=False
            )
        elif args.data == 'medicine':
            testfolder = ImageFolder(testdir, transform=transform_test, index=index)
            test_loader = torch.utils.data.DataLoader(
                testfolder, 
                batch_size=128, 
                shuffle=False, 
                num_workers=0,
                drop_last=False
            )
        
        print('Test %d\t' % task_id)
        model_id = task_id
        model = load_model_safely(models[model_id])
        model = model.cuda()
        model.eval()
        
        # 進行測試
        accuracy, test_data = test_task(args, test_loader, task_id, model)
        print(f'Task {task_id} Accuracy: {accuracy:.2f}%')
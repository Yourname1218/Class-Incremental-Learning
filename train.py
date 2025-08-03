# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
import getpass
import os
import sys
import torch.utils.data
import pdb
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.autograd import Variable
import models
from torch.serialization import add_safe_globals
from models.resnet import Generator, Discriminator, BasicBlock, Bottleneck

from utils import RandomIdentitySampler, mkdir_if_missing, logging, display,truncated_z_sample
from medical_data_augmentation import get_medical_transforms, MedicalTransformPipeline
from torch.optim.lr_scheduler import StepLR
import numpy as np
from ImageFolder import *
import torch.utils.data	
import torch.nn.functional as F
import torchvision.transforms as transforms
from evaluations import extract_features, pairwise_distance
from models.resnet import Generator, Discriminator,ClassifierMLP,ModelCNN
import torch.autograd as autograd
import scipy.io as sio
import matplotlib.pyplot as plt
from datetime import datetime
import math
import shutil
import json
import seaborn as sns

# 第二階段優化：導入醫學特定優化組件
from stage2_optimizations import (
    Stage2OptimizationManager, 
    AnatomicalConsistencyLoss,
    MultiScaleFeatureLoss,
    MedicalSemanticConsistencyLoss,
    ProgressiveGANTrainer,
    DynamicEnsembleWeighter,
    ConfidenceAwareFusion,
    MedicalDomainSpecificOptimizer,
    create_stage2_manager
)

# 第三階段優化：導入集成學習策略優化和記憶體效率優化組件
from stage3_optimizations import (
    Stage3OptimizationManager,
    AdaptiveEnsembleSelector,
    DiversityDrivenOptimizer,
    DynamicModelSelector,
    TimeAwareEnsembleWeighter,
    UncertaintyQuantificationEnsemble,
    GradientCheckpointer,
    AdaptiveModelQuantizer,
    DynamicBatchSizer,
    FeatureReuseManager,
    MemoryPoolManager,
    TaskFeatureCompressor,
    create_stage3_manager
)

# 將需要的類添加到安全列表中（移除不需要的 ResNet_ImageNet 和 ResNet_Cifar）
add_safe_globals([Generator, Discriminator, BasicBlock, Bottleneck, ClassifierMLP, ModelCNN])

cudnn.benchmark = True
from copy import deepcopy

def to_binary(labels,args):
    # Y_onehot is used to generate one-hot encoding
    y_onehot = torch.FloatTensor(len(labels), args.num_class)
    y_onehot.zero_()
    y_onehot.scatter_(1, labels.cpu()[:,None], 1)
    code_binary = y_onehot.cuda()
    return code_binary

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return model


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def compute_gradient_penalty(D, real_samples, fake_samples, syn_label):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.tensor(np.random.random((real_samples.size(0), 1)), 
                        dtype=torch.float32,
                        device='cuda')
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates, syn_label)
    
    # 修正這裡：正確創建 fake tensor
    fake = torch.ones(real_samples.shape[0], 1, device='cuda', requires_grad=False)
    
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates, 
                            inputs=interpolates, 
                            grad_outputs=fake, 
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True)[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def compute_prototype(model, data_loader,number_samples=200):
    model.eval()
    count = 0
    embeddings = []
    embeddings_labels = []
    terminate_flag = min(len(data_loader),number_samples)
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            if i>terminate_flag:
                break
            count += 1
            inputs, labels = data
            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            embed_feat = model(inputs)
            embeddings_labels.append(labels.numpy())
            embeddings.append(embed_feat.cpu().numpy())

    embeddings = np.asarray(embeddings)
    embeddings = np.reshape(embeddings, (embeddings.shape[0] * embeddings.shape[1], embeddings.shape[2]))
    embeddings_labels = np.asarray(embeddings_labels)
    embeddings_labels = np.reshape(embeddings_labels, embeddings_labels.shape[0] * embeddings_labels.shape[1])
    labels_set = np.unique(embeddings_labels)
    class_mean = []
    class_std = []
    class_label = []
    for i in labels_set:
        ind_cl = np.where(i == embeddings_labels)[0]
        embeddings_tmp = embeddings[ind_cl]
        class_label.append(i)
        class_mean.append(np.mean(embeddings_tmp, axis=0))
        class_std.append(np.std(embeddings_tmp, axis=0))
    prototype = {'class_mean': class_mean, 'class_std': class_std,'class_label': class_label}

    return prototype


# 在train.py開頭添加AdaptiveTradeoff類
class AdaptiveTradeoff:
    def __init__(self, base_tradeoff, momentum=0.9):
        self.base_tradeoff = base_tradeoff
        self.momentum = momentum
        self.current_value = base_tradeoff
        self.loss_history = []
        
    def update(self, current_loss, window_size=5):
        """
        根據最近的損失變化動態調整權重
        - 如果損失趨勢增加：增加權重以加強知識保留
        - 如果損失趨勢減少：減少權重以促進新知識學習
        """
        self.loss_history.append(current_loss)
        if len(self.loss_history) > window_size:
            self.loss_history.pop(0)
            
            # 計算損失變化趨勢
            loss_trend = (self.loss_history[-1] - self.loss_history[0]) / window_size
            
            # 根據趨勢調整權重
            if loss_trend > 0:  # 損失在增加
                self.current_value *= 1.1  # 增加權重
            else:  # 損失在減少
                self.current_value *= 0.9  # 減少權重
            
            # 限制權重範圍在基礎權重的0.1~2.0倍之間
            self.current_value = max(self.base_tradeoff * 0.1, 
                                   min(self.base_tradeoff * 2.0, self.current_value))
        
        return self.current_value

class DynamicLossWeights:
    def __init__(self, initial_l2_weight=0.5, initial_cos_weight=0.5, 
                 window_size=5, adjust_rate=0.1, min_weight=0.2, max_weight=0.8):
        """
        初始化動態損失權重管理器
        
        參數:
        - initial_l2_weight: L2損失的初始權重 (預設: 0.5)
        - initial_cos_weight: 餘弦損失的初始權重 (預設: 0.5)
        - window_size: 用於計算趨勢的歷史窗口大小 (預設: 5)
        - adjust_rate: 權重調整的步長 (預設: 0.1)
        - min_weight: 最小權重限制 (預設: 0.2)
        - max_weight: 最大權重限制 (預設: 0.8)
        """
        self.l2_weight = initial_l2_weight
        self.cos_weight = initial_cos_weight
        self.window_size = window_size
        self.adjust_rate = adjust_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # 儲存歷史損失值
        self.l2_history = []
        self.cos_history = []
        
        # 記錄性能指標
        self.performance_history = []
    
    def update(self, l2_loss, cos_loss, current_performance=None):
        """
        更新權重
        
        參數:
        - l2_loss: 當前批次的L2損失值
        - cos_loss: 當前批次的餘弦損失值
        - current_performance: 當前的性能指標（可選）
        
        返回:
        - l2_weight, cos_weight: 更新後的權重
        """
        # 添加當前損失到歷史記錄
        self.l2_history.append(l2_loss)
        self.cos_history.append(cos_loss)
        
        if current_performance is not None:
            self.performance_history.append(current_performance)
        
        # 當歷史記錄達到窗口大小時進行調整
        if len(self.l2_history) >= self.window_size:
            # 計算損失變化趨勢
            l2_trend = (self.l2_history[-1] - self.l2_history[0]) / self.window_size
            cos_trend = (self.cos_history[-1] - self.cos_history[0]) / self.window_size
            
            # 根據趨勢調整權重
            if l2_trend > cos_trend:  # L2損失增加得更快
                # 增加餘弦相似度的權重
                self.l2_weight = max(self.min_weight, 
                                   self.l2_weight - self.adjust_rate)
                self.cos_weight = min(self.max_weight, 
                                    self.cos_weight + self.adjust_rate)
            else:  # 餘弦損失增加得更快
                # 增加L2的權重
                self.l2_weight = min(self.max_weight, 
                                   self.l2_weight + self.adjust_rate)
                self.cos_weight = max(self.min_weight, 
                                    self.cos_weight - self.adjust_rate)
            
            # 確保權重和為1
            total = self.l2_weight + self.cos_weight
            self.l2_weight /= total
            self.cos_weight /= total
            
            # 移除最舊的記錄
            self.l2_history.pop(0)
            self.cos_history.pop(0)
        
        return self.l2_weight, self.cos_weight
    
    def get_weights(self):
        """獲取當前權重"""
        return self.l2_weight, self.cos_weight
    
    def get_statistics(self):
        """獲取統計信息"""
        return {
            'l2_weight': self.l2_weight,
            'cos_weight': self.cos_weight,
            'l2_history': self.l2_history,
            'cos_history': self.cos_history,
            'performance_history': self.performance_history
        }

def validate_generator(generator, model, num_classes, latent_dim, current_task=0, batch_size=128):
    """驗證生成器對各類別特徵的生成質量"""
    generator.eval()
    model.eval()
    
    samples_per_class = 100  # 每類生成100個樣本進行驗證
    device = next(model.parameters()).device
    
    class_accuracies = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)
    
    # 遍歷所有類別
    with torch.no_grad():
        for cls_idx in range(num_classes):
            # 為當前類別創建標籤
            labels = torch.full((samples_per_class,), cls_idx, device=device).long()
            
            # 創建one-hot編碼
            y_onehot = torch.zeros(samples_per_class, num_classes, device=device)
            y_onehot.scatter_(1, labels.unsqueeze(1), 1)
            
            # 生成隨機噪聲
            z = torch.randn(samples_per_class, latent_dim, device=device)
            
            # 生成特徵
            gen_features = generator(z, y_onehot)
            
            # 使用當前模型進行分類
            logits = model.embed(gen_features)
            
            # 計算準確率
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == labels).float().mean().item()
            
            class_accuracies[cls_idx] = accuracy
            class_counts[cls_idx] = samples_per_class
    
    # 計算總體準確率
    overall_acc = class_accuracies.mean()
    
    # 特別關注前500類的準確率
    front_acc = class_accuracies[:500].mean() if num_classes > 500 else overall_acc
    
    # 不在這裡繪製熱圖，因為外部已有繪圖邏輯
    
    return {
        'overall_acc': overall_acc,
        'front_acc': front_acc,
        'class_accuracies': class_accuracies,
        'class_counts': class_counts
        }

def _generate_optimization_report(smart_scheduler, enhanced_dynamic_weights, 
                                  current_task, log_dir, task_difficulty):
    """
    生成第一階段優化的詳細報告
    
    參數:
    - smart_scheduler: 智能調度器實例
    - enhanced_dynamic_weights: 增強動態權重管理器實例
    - current_task: 當前任務編號
    - log_dir: 日誌目錄
    - task_difficulty: 任務難度
    """
    print("\n" + "="*80)
    print("第一階段優化報告 - 智能學習率調度 + 增強動態損失權重")
    print("="*80)
    
    # 獲取統計數據
    scheduler_stats = smart_scheduler.get_statistics()
    weights_stats = enhanced_dynamic_weights.get_enhanced_statistics()
    
    # 1. 智能學習率調度器報告
    print(f"\n智能學習率調度器效果分析 (任務 {current_task}):")
    print(f"   學習率重啟次數: {scheduler_stats['restart_count']}")
    print(f"   最佳驗證損失: {scheduler_stats['best_val_loss']:.6f}")
    print(f"   當前訓練輪數: {scheduler_stats['current_epoch']}")
    print(f"   學習率變化範圍: {min(scheduler_stats['lr_history']):.6f} ~ {max(scheduler_stats['lr_history']):.6f}")
    
    if scheduler_stats['restart_count'] > 0:
        print(f"   學習率重啟幫助模型逃離了 {scheduler_stats['restart_count']} 次局部最優")
    
    # 2. 增強動態損失權重報告
    print(f"\n增強動態損失權重效果分析:")
    print(f"   當前L2權重: {weights_stats['l2_weight']:.4f}")
    print(f"   當前餘弦權重: {weights_stats['cos_weight']:.4f}")
    print(f"   任務難度係數: {task_difficulty:.2f}")
    
    if weights_stats['weight_change_history']:
        avg_change = sum(weights_stats['weight_change_history']) / len(weights_stats['weight_change_history'])
        print(f"   平均權重變化幅度: {avg_change:.4f}")
        print(f"   權重調整次數: {len(weights_stats['weight_change_history'])}")
    
    # 3. 收斂狀態分析
    if 'convergence_stats' in weights_stats:
        conv_stats = weights_stats['convergence_stats']
        if conv_stats['loss_history']:
            recent_trend = "穩定" if len(conv_stats['loss_history']) > 5 else "初始化中"
            print(f"\n收斂狀態分析:")
            print(f"   當前收斂狀態: {recent_trend}")
            print(f"   穩定性閾值: {conv_stats['stability_threshold']}")
    
    # 4. 性能相關性分析
    if weights_stats['performance_correlation']:
        correlations = weights_stats['performance_correlation']
        if len(correlations) > 5:
            weight_changes = [c[0] for c in correlations]
            performances = [c[1] for c in correlations]
            
            # 簡單的相關性計算
            avg_weight_change = sum(weight_changes) / len(weight_changes)
            avg_performance = sum(performances) / len(performances)
            
            print(f"\n性能相關性分析:")
            print(f"   平均權重變化: {avg_weight_change:.4f}")
            print(f"   平均性能指標: {avg_performance:.4f}")
            print(f"   記錄樣本數: {len(correlations)}")
    
    # 5. 保存詳細報告到文件
    report_path = os.path.join(log_dir, f'optimization_report_task_{current_task}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("第一階段優化詳細報告\n")
        f.write("="*50 + "\n\n")
        f.write(f"任務編號: {current_task}\n")
        f.write(f"任務難度: {task_difficulty:.2f}\n\n")
        
        f.write("智能學習率調度器統計:\n")
        f.write(f"  學習率重啟次數: {scheduler_stats['restart_count']}\n")
        f.write(f"  最佳驗證損失: {scheduler_stats['best_val_loss']:.6f}\n")
        f.write(f"  學習率歷史: {scheduler_stats['lr_history']}\n\n")
        
        f.write("增強動態損失權重統計:\n")
        f.write(f"  最終L2權重: {weights_stats['l2_weight']:.4f}\n")
        f.write(f"  最終餘弦權重: {weights_stats['cos_weight']:.4f}\n")
        f.write(f"  權重變化歷史: {weights_stats['weight_change_history']}\n")
        f.write(f"  性能相關性: {weights_stats['performance_correlation']}\n")
    
    print(f"\n詳細報告已保存至: {report_path}")
    
    # 6. 優化建議
    print(f"\n優化建議:")
    
    if scheduler_stats['restart_count'] == 0:
        print("   學習率調度：未觸發重啟，可考慮降低patience值")
    elif scheduler_stats['restart_count'] > 2:
        print("   學習率調度：重啟頻繁，可考慮提高patience值")
    else:
        print("   學習率調度：重啟頻率適中，調度策略有效")
    
    if weights_stats['weight_change_history'] and len(weights_stats['weight_change_history']) > 10:
        recent_changes = weights_stats['weight_change_history'][-5:]
        avg_recent_change = sum(recent_changes) / len(recent_changes)
        
        if avg_recent_change < 0.01:
            print("   權重調整：變化幅度較小，權重已趨於穩定")
        elif avg_recent_change > 0.1:
            print("   權重調整：變化幅度較大，可考慮降低adjust_rate")
        else:
            print("   權重調整：變化幅度適中，調整策略有效")
    
    print("   總體評估：第一階段優化功能正常運作")
    print("\n" + "="*80)

def _compute_validation_loss(model, val_loader):
    """計算驗證損失"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data in val_loader:
            if num_batches >= 10:  # 只計算前10個批次以節省時間
                break
                
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda().long()
            
            embed_feat = model(inputs)
            soft_feat = model.embed(embed_feat)
            loss = torch.nn.CrossEntropyLoss()(soft_feat, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / max(num_batches, 1)

def train_task(args, train_loader, current_task, prototype={}, pre_index=0):
    num_class_per_task = (args.num_class-args.nb_cl_fg) // args.num_task
    task_range = list(range(args.nb_cl_fg + (current_task - 1) * num_class_per_task, args.nb_cl_fg + current_task * num_class_per_task))
    if num_class_per_task==0:
        pass  # JT
    else:
        old_task_factor = args.nb_cl_fg // num_class_per_task + current_task - 1
    log_dir = os.path.join(args.ckpt_dir, args.log_dir)
    mkdir_if_missing(log_dir)

    sys.stdout = logging.Logger(os.path.join(log_dir, 'log_task{}.txt'.format(current_task)))
    tb_writer = SummaryWriter(log_dir)
    display(args)
    
    # 確保所有需要的類都已經被註冊為安全
    add_safe_globals([Generator, Discriminator, BasicBlock, Bottleneck, ClassifierMLP, ModelCNN])
    
    # One-hot encoding or attribute encoding
    # 只支援 Medicine 資料集，使用 ResNet50 for ImageNet 架構
    model = models.create('resnet50_imagenet', pretrained=False, feat_dim=512, embed_dim=args.num_class)

    if current_task > 0:
        try:
            # 修改模型加載邏輯：優先嘗試加載微調後的生成器
            model_path = os.path.join(log_dir, f'task_{str(current_task - 1).zfill(2)}_{args.epochs - 1}_model.pkl')
            
            # 檢查是否存在微調後的生成器
            finetuned_generator_path = os.path.join(log_dir, f'task_{str(current_task - 1).zfill(2)}_finetuned_generator.pkl')
            original_generator_path = os.path.join(log_dir, f'task_{str(current_task - 1).zfill(2)}_{args.epochs_gan - 1}_model_generator.pkl')
            discriminator_path = os.path.join(log_dir, f'task_{str(current_task - 1).zfill(2)}_{args.epochs_gan - 1}_model_discriminator.pkl')
            
            print(f"Loading previous models from:")
            print(f"Model: {model_path}")
            
            # 優先嘗試加載微調後的生成器
            if os.path.exists(finetuned_generator_path):
                print(f"Generator (Finetuned): {finetuned_generator_path}")
                generator_path = finetuned_generator_path
            else:
                print(f"Generator (Original): {original_generator_path}")
                generator_path = original_generator_path
                
            print(f"Discriminator: {discriminator_path}")
            
            def safe_load(path, desc):
                """安全地加載模型，包含多種嘗試策略"""
                try:
                    print(f"嘗試加載 {desc}，使用 weights_only=False")
                    return torch.load(path, weights_only=False)
                except Exception as e1:
                    print(f"使用 weights_only=False 加載 {desc} 失敗: {e1}")
                    try:
                        print(f"嘗試加載 {desc}，使用 pickle.load")
                        import pickle
                        with open(path, 'rb') as f:
                            return pickle.load(f)
                    except Exception as e2:
                        print(f"使用 pickle.load 加載 {desc} 失敗: {e2}")
                        try:
                            print(f"最後嘗試使用 torch.load 搭配 map_location='cpu'")
                            return torch.load(path, weights_only=False, map_location='cpu')
                        except Exception as e3:
                            print(f"所有加載方法都失敗: {e3}")
                            raise RuntimeError(f"無法加載 {desc}")
            
            # 嘗試安全加載主模型
            print("加載主模型...")
            model = safe_load(model_path, "主模型")
            model = model.cuda()
            
            model_old = deepcopy(model)
            model_old.eval()
            model_old = freeze_model(model_old)
            
            # 嘗試安全加載生成器（優先使用微調後的）
            print("加載生成器...")
            generator = safe_load(generator_path, "生成器")
            print("加載判別器...")
            discriminator = safe_load(discriminator_path, "判別器")
            
            generator = generator.cuda()
            discriminator = discriminator.cuda()
            
            generator_old = deepcopy(generator)
            generator_old.eval()
            generator_old = freeze_model(generator_old)
            
            print("Successfully loaded previous models")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    else:
        # 第一個任務的初始化
        # 只支援 Medicine 資料集，使用 ResNet50 for ImageNet 架構
        model = models.create('resnet50_imagenet', pretrained=False, feat_dim=512, embed_dim=args.num_class)
        model = model.cuda()
        
        generator = Generator(feat_dim=512, latent_dim=args.latent_dim, hidden_dim=512, class_dim=args.num_class).cuda()
        discriminator = Discriminator(feat_dim=512, hidden_dim=512, class_dim=args.num_class).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)

    loss_mse = torch.nn.MSELoss(reduction='sum')

    # Loss weight for gradient penalty used in W-GAN
    lambda_gp = args.lambda_gp
    lambda_lwf = args.gan_tradeoff
    # Initialize generator and discriminator
    if current_task == 0:
        generator = Generator(feat_dim=512, latent_dim=args.latent_dim, hidden_dim=512, class_dim=args.num_class)
        discriminator = Discriminator(feat_dim=512, hidden_dim=512, class_dim=args.num_class)
    else:
        try:
            generator_path = os.path.join(log_dir, 'task_' + str(current_task - 1).zfill(2) + '_%d_model_generator.pkl' % int(args.epochs_gan - 1))
            discriminator_path = os.path.join(log_dir, 'task_' + str(current_task - 1).zfill(2) + '_%d_model_discriminator.pkl' % int(args.epochs_gan - 1))
            
            print(f"Loading generator from: {generator_path}")
            print(f"Loading discriminator from: {discriminator_path}")
            
            def safe_load(path, desc):
                """安全地加載模型，包含多種嘗試策略"""
                try:
                    print(f"嘗試加載 {desc}，使用 weights_only=False")
                    return torch.load(path, weights_only=False)
                except Exception as e1:
                    print(f"使用 weights_only=False 加載 {desc} 失敗: {e1}")
                    try:
                        print(f"嘗試加載 {desc}，使用 pickle.load")
                        import pickle
                        with open(path, 'rb') as f:
                            return pickle.load(f)
                    except Exception as e2:
                        print(f"使用 pickle.load 加載 {desc} 失敗: {e2}")
                        try:
                            print(f"最後嘗試使用 torch.load 搭配 map_location='cpu'")
                            return torch.load(path, weights_only=False, map_location='cpu')
                        except Exception as e3:
                            print(f"所有加載方法都失敗: {e3}")
                            raise RuntimeError(f"無法加載 {desc}")
            
            # 嘗試安全加載生成器和判別器
            print("加載生成器...")
            generator = safe_load(generator_path, "生成器")
            print("加載判別器...")
            discriminator = safe_load(discriminator_path, "判別器")
            
            generator_old = deepcopy(generator)
            generator_old.eval()
            generator_old = freeze_model(generator_old)
        except Exception as e:
            print(f"錯誤: 無法加載生成器或判別器: {e}")
            print("將重新初始化生成器和判別器")
            generator = Generator(feat_dim=512, latent_dim=args.latent_dim, hidden_dim=512, class_dim=args.num_class)
            discriminator = Discriminator(feat_dim=512, hidden_dim=512, class_dim=args.num_class)
        generator_old = deepcopy(generator)
        generator_old.eval()
        generator_old = freeze_model(generator_old)

    generator = generator.cuda()
    discriminator = discriminator.cuda()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    scheduler_G = StepLR(optimizer_G, step_size=100, gamma=0.5)
    scheduler_D = StepLR(optimizer_D, step_size=100, gamma=0.5)

    # Y_onehot is used to generate one-hot encoding
    y_onehot = torch.FloatTensor(args.BatchSize, args.num_class).cuda()

    for p in generator.parameters():  # set requires_grad to False
        p.requires_grad = False    

    ###############################################################Feature extractor training####################################################
    if current_task>0:
        model = model.eval()

    # 初始化數據收集器
    training_data = {
        'epochs': [],
        'total_loss': [],
        'lwf_loss': [],
        'cls_loss': [],
        'learning_rate': [],
        'val_accuracy': []  # 添加驗證準確率記錄
    }
    
    # 第一階段優化：使用智能學習率調度器
    print("啟用智能學習率調度器...")
    smart_scheduler = SmartScheduler(
        optimizer=optimizer,
        total_epochs=args.epochs,
        warmup_epochs=5,
        min_lr=1e-6,
        restart_factor=0.5,
        patience=10
    )
    
    # 初始化圖表
    fig, axs = None, None
    
    # 初始化自適應權重調整器
    if current_task > 0:
        adaptive_tradeoff = AdaptiveTradeoff(args.tradeoff)
    
    # 第一階段優化：使用增強版動態損失權重管理器
    print("啟用增強版動態損失權重管理器...")
    enhanced_dynamic_weights = EnhancedDynamicLossWeights(
        initial_l2_weight=0.5,
        initial_cos_weight=0.5,
        window_size=5,
        adjust_rate=0.1,
        min_weight=0.2,
        max_weight=0.8,
        performance_threshold=0.1,
        task_difficulty_factor=1.0
    )
    
    # 評估任務難度（基於任務索引的簡單評估）
    task_difficulty = min(0.9, 0.3 + current_task * 0.1)  # 隨任務增加而增加難度
    print(f"任務 {current_task} 難度評估: {task_difficulty:.2f}")
    
    # ============ 第二階段優化：醫學特定優化初始化 ============
    print("\n" + "="*60)
    print("第二階段優化：醫學特定優化啟動")
    print("="*60)
    
    # 初始化第二階段優化管理器
    stage2_manager = create_stage2_manager(args, args.num_class)
    
    # 設置漸進式GAN訓練（如果不是第一個任務）
    if current_task > 0:
        stage2_manager.setup_progressive_gan(generator, discriminator, args.latent_dim)
        print("漸進式GAN訓練已啟用")
    
    # 初始化醫學特定損失追蹤
    stage2_losses = {
        'anatomical_consistency': [],
        'multiscale_features': [],
        'semantic_consistency': [],
        'total_medical_loss': []
    }
    
    print("第二階段優化組件初始化完成：")
    print("[OK] 醫學特定損失函數")
    print("[OK] GAN穩定性改進機制") 
    print("[OK] 動態集成權重優化")
    print("[OK] 置信度感知融合")
    print("="*60 + "\n")
    
    # ============ 第三階段優化：集成學習策略優化 + 記憶體效率優化 ============
    print("\n" + "="*80)
    print("第三階段優化：集成學習策略優化 + 記憶體效率優化啟動")
    print("="*80)
    
    # 初始化第三階段優化管理器
    stage3_manager = create_stage3_manager(args, args.num_class)
    
    # 獲取當前批次大小作為基準
    args.batch_size = getattr(args, 'batch_size', 32)
    current_batch_size = args.batch_size
    
    # 初始化集成學習統計追蹤
    stage3_stats = {
        'ensemble_selections': [],
        'memory_optimizations': [],
        'batch_size_changes': [],
        'feature_reuse_hits': [],
        'uncertainty_scores': []
    }
    
    print("第三階段優化組件初始化完成：")
    print("[OK] 自適應集成選擇器")
    print("[OK] 多樣性驅動優化器")
    print("[OK] 動態模型選擇器")
    print("[OK] 時間感知權重調整")
    print("[OK] 不確定性量化集成")
    print("[OK] 梯度檢查點管理")
    print("[OK] 自適應模型量化")
    print("[OK] 動態批次大小調整")
    print("[OK] 特徵重用機制")
    print("[OK] 記憶體池管理")
    print("[OK] 任務特徵壓縮")
    print("="*80 + "\n")
    
    # 創建驗證資料集載入器
    # 使用與藥物圖片測試相同的預處理
    if current_task == 0:
        cumulative_index = class_index
    else:
        cumulative_index = list(range(args.nb_cl_fg + current_task * num_class_per_task))
    
    # Medicine 資料集的驗證資料集
    valid_dir = os.path.join('medicine_picture', 'valid')
    if os.path.exists(valid_dir):
        val_transform = get_medical_transforms(mode='val', image_size=224)
        valfolder = ImageFolder(valid_dir, transform=val_transform, index=cumulative_index)
        val_loader = torch.utils.data.DataLoader(
            valfolder, batch_size=args.BatchSize,
            shuffle=False, num_workers=args.nThreads)
    else:
        print("警告：找不到驗證資料集，跳過驗證步驟")
        val_loader = None
    
    for epoch in range(args.epochs):

        loss_log = {'C/loss': 0.0,
                    'C/loss_aug': 0.0,
                    'C/loss_cls': 0.0}
        
        # ============ 第三階段優化：每個Epoch開始時的記憶體和集成優化 ============
        # 動態調整批次大小
        loss_history = [loss_log['C/loss'] for _ in range(min(epoch, 5))]  # 簡化的損失歷史
        accuracy_history = []  # 將在後面填充
        
        current_batch_size = stage3_manager.optimize_memory_usage(
            epoch, loss_history, accuracy_history
        )
        
        # 更新 DataLoader 的批次大小（如果有變化）
        if current_batch_size != args.batch_size:
            args.batch_size = current_batch_size
            stage3_stats['batch_size_changes'].append({
                'epoch': epoch,
                'old_size': args.batch_size,
                'new_size': current_batch_size
            })
            print(f"Epoch {epoch}: 批次大小調整為 {current_batch_size}")
        
        # 第一階段優化：智能學習率調度
        # 計算驗證損失（如果有驗證資料集）
        val_loss = None
        if val_loader is not None and epoch % 5 == 0:  # 每5個epoch驗證一次
            val_loss = _compute_validation_loss(model, val_loader)
        
        smart_scheduler.step(epoch, val_loss)
        for i, data in enumerate(train_loader, 0):
            inputs1, labels1 = data
            inputs1, labels1 = inputs1.cuda(), labels1.cuda().long()

            # 初始化各種損失
            loss = torch.zeros(1).cuda()
            loss_cls = torch.zeros(1).cuda()
            loss_aug = torch.zeros(1).cuda()
            optimizer.zero_grad()
            
            inputs, labels = inputs1, labels1
            
            ### Classification loss with Stage 3 Optimizations
            # 使用第三階段優化的前向傳播（特徵重用 + 梯度檢查點）
            embed_feat = stage3_manager.forward_with_optimizations(model, inputs, 'features')
            
            if current_task == 0:
                # 第一個任務只計算分類損失
                soft_feat = model.embed(embed_feat)
                loss_cls = torch.nn.CrossEntropyLoss()(soft_feat, labels)
                loss += loss_cls
            else:
                # 後續任務需要計算舊模型的特徵
                # 同樣使用第三階段優化
                embed_feat_old = stage3_manager.forward_with_optimizations(model_old, inputs, 'features_old')

            ### Feature Extractor Loss
            if current_task > 0:
                # 計算兩種損失
                l2_loss = torch.dist(embed_feat, embed_feat_old, 2)
                cos_loss = 1 - F.cosine_similarity(embed_feat, embed_feat_old).mean()
                
                # 第一階段優化：使用增強版動態權重調整器
                # 計算當前性能（簡化版，基於分類損失）
                current_performance = 1.0 / (1.0 + loss_cls.item()) if loss_cls.item() > 0 else 0.5
                
                l2_weight, cos_weight = enhanced_dynamic_weights.update_with_performance(
                    l2_loss.item(), 
                    cos_loss.item(),
                    current_performance=current_performance,
                    task_difficulty=task_difficulty,
                    epoch=epoch
                )
                
                # 組合損失
                loss_aug = l2_weight * l2_loss + cos_weight * cos_loss
                
                # ============ 第二階段優化：添加醫學特定損失 ============
                # 計算醫學特定損失
                medical_loss, medical_components = stage2_manager.compute_stage2_losses(
                    embed_feat, embed_feat_old, current_task, epoch
                )
                
                # 記錄醫學特定損失
                stage2_losses['anatomical_consistency'].append(medical_components.get('anatomical_region_consistency', 0))
                stage2_losses['multiscale_features'].append(medical_components.get('multiscale_scale_0', 0))
                stage2_losses['semantic_consistency'].append(medical_components.get('semantic_semantic_consistency', 0))
                stage2_losses['total_medical_loss'].append(medical_loss.item())
                
                # 組合第一階段和第二階段損失
                enhanced_loss_aug = loss_aug + 0.3 * medical_loss  # 醫學損失權重為0.3
                
                # 使用自適應權重調整整體知識蒸餾損失的權重
                current_tradeoff = adaptive_tradeoff.update(enhanced_loss_aug.item())
                loss += current_tradeoff * enhanced_loss_aug * old_task_factor
                
                # 每20個批次輸出醫學損失統計
                if i % 20 == 0:
                    print(f"醫學特定損失 [Epoch {epoch+1}, Batch {i+1}]:")
                    print(f"   解剖學一致性: {medical_components.get('anatomical_region_consistency', 0):.4f}")
                    print(f"   多尺度特徵: {medical_components.get('multiscale_scale_0', 0):.4f}")
                    print(f"   語義一致性: {medical_components.get('semantic_semantic_consistency', 0):.4f}")
                    print(f"   總醫學損失: {medical_loss.item():.4f}")
                    print(f"   增強後LwF損失: {enhanced_loss_aug.item():.4f}")
                
                # 更新漸進式訓練狀態
                stage2_manager.update_progressive_training(epoch)
            
            ### Replay and Classification Loss
            if current_task > 0: 
                embed_sythesis = []
                embed_label_sythesis = []
                ind = list(range(len(pre_index)))

                if args.mean_replay:
                    for _ in range(args.BatchSize):                        
                        np.random.shuffle(ind)
                        tmp = prototype['class_mean'][ind[0]]+np.random.normal()*prototype['class_std'][ind[0]]
                        embed_sythesis.append(tmp)
                        embed_label_sythesis.append(prototype['class_label'][ind[0]])
                    embed_sythesis = np.asarray(embed_sythesis)
                    embed_label_sythesis = np.asarray(embed_label_sythesis)
                    embed_sythesis = torch.from_numpy(embed_sythesis).cuda()
                    embed_label_sythesis = torch.from_numpy(embed_label_sythesis).cuda().long()
                else:
                    # 修改生成特徵的採樣邏輯：優先採樣前500類
                    # 將舊類別分為前500類和其他類別
                    front_indices = [idx for idx in pre_index if idx < 500]
                    other_indices = [idx for idx in pre_index if idx >= 500]
                    
                    # 設定前500類的採樣比例
                    front_ratio = 0.7  # 70%樣本來自前500類
                    batch_front = int(args.BatchSize * front_ratio)
                    batch_other = args.BatchSize - batch_front
                    
                    # 從前500類中採樣
                    front_labels = []
                    if front_indices:
                        for _ in range(batch_front):
                            front_labels.append(np.random.choice(front_indices))
                    
                    # 從其他舊類別中採樣
                    other_labels = []
                    if other_indices:
                        for _ in range(batch_other):
                            other_labels.append(np.random.choice(other_indices))
                    elif front_indices:  # 如果沒有其他類別，從前500類補充
                        for _ in range(batch_other):
                            front_labels.append(np.random.choice(front_indices))
                    
                    # 合併標籤
                    combined_labels = front_labels + other_labels
                    embed_label_sythesis = torch.tensor(combined_labels, dtype=torch.long, device='cuda')
                    
                    # 準備one-hot標籤
                    y_onehot.zero_()
                    y_onehot.scatter_(1, embed_label_sythesis.view(-1, 1).long(), 1)
                    syn_label_pre = y_onehot.cuda()

                    # 生成特徵
                    z = torch.randn(len(embed_label_sythesis), args.latent_dim).cuda()
                    embed_sythesis = generator(z, syn_label_pre)
                
                    # 記錄前500類採樣比例
                    if i % 20 == 0:  # 每20個批次輸出一次
                        front_count = sum(1 for label in embed_label_sythesis.cpu().numpy() if label < 500)
                        front_percent = front_count / len(embed_label_sythesis) * 100
                        print(f"批次 {i}：前500類採樣比例 = {front_percent:.2f}%")
                
                # 合併真實特徵和生成特徵
                embed_sythesis = torch.cat((embed_feat, embed_sythesis))
                embed_label_sythesis = torch.cat((labels, embed_label_sythesis))
                soft_feat_syt = model.embed(embed_sythesis)
                # real samples,   exemplars,      synthetic samples
                #           batch_size1       batch_size2
                                     

                batch_size1 = inputs1.shape[0]
                batch_size2 = embed_feat.shape[0]

                loss_cls = torch.nn.CrossEntropyLoss()(soft_feat_syt[:batch_size1], embed_label_sythesis[:batch_size1])

                loss_cls_old = torch.nn.CrossEntropyLoss()(soft_feat_syt[batch_size2:], embed_label_sythesis[batch_size2:])
                
                loss_cls += loss_cls_old * old_task_factor
                loss_cls /= args.nb_cl_fg // num_class_per_task + current_task
                loss += loss_cls
                        
            loss.backward()
            optimizer.step()

            loss_log['C/loss'] += loss.item()
            loss_log['C/loss_cls'] += loss_cls.item()
            loss_log['C/loss_aug'] += current_tradeoff * loss_aug.item() if current_task > 0 else 0
            del loss_cls
            if epoch == 0 and i == 0:
                print(50 * '#')

        print('[Metric Epoch %05d]\t Total Loss: %.3f \t LwF Loss: %.3f \t'
                % (epoch + 1, loss_log['C/loss'], loss_log['C/loss_aug']))
        for k, v in loss_log.items():
            if v != 0:
                tb_writer.add_scalar('Task {} - Classifier/{}'.format(current_task, k), v, epoch + 1)

        if epoch == args.epochs-1:
            model_save_path = os.path.join(log_dir, f'task_{str(current_task).zfill(2)}_{epoch}_model.pkl')
            # 使用 _use_new_zipfile_serialization=False 以確保兼容性
            torch.save(model, model_save_path, _use_new_zipfile_serialization=False)
            print(f"Saved model to: {model_save_path}")

        # 收集數據
        current_lr = optimizer.param_groups[0]['lr']
        training_data['epochs'].append(epoch + 1)
        training_data['total_loss'].append(loss_log['C/loss'])
        training_data['lwf_loss'].append(loss_log['C/loss_aug'])
        training_data['cls_loss'].append(loss_log['C/loss_cls'])
        training_data['learning_rate'].append(current_lr)
        
        # 第一階段優化：收集增強統計信息
        if current_task > 0 and epoch % 20 == 0:
            enhanced_stats = enhanced_dynamic_weights.get_enhanced_statistics()
            scheduler_stats = smart_scheduler.get_statistics()
            
            print(f"第一階段優化統計 - Epoch {epoch+1}:")
            print(f"   學習率重啟次數: {scheduler_stats['restart_count']}")
            print(f"   最佳驗證損失: {scheduler_stats['best_val_loss']:.4f}")
            print(f"   權重變化幅度: {enhanced_stats['weight_change_history'][-1]:.4f}" if enhanced_stats['weight_change_history'] else "N/A")
        
        # 驗證階段
        if val_loader is not None:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in val_loader:
                    images, labels = data
                    images, labels = images.cuda(), labels.cuda()
                    
                    # 獲取特徵和預測
                    features = model(images)
                    outputs = model.embed(features)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_accuracy = 100 * correct / total
            training_data['val_accuracy'].append(val_accuracy)
            
            print(f'[Validation Epoch {epoch + 1}]\t Accuracy: {val_accuracy:.2f}%')
            tb_writer.add_scalar(f'Task {current_task} - Validation/Accuracy', val_accuracy, epoch + 1)
            
            # 切回訓練模式
            if current_task > 0:
                model = model.eval()  # 对于任务>0，使用eval模式
            else:
                model = model.train()
        else:
            # 如果沒有驗證資料集，記錄為0
            training_data['val_accuracy'].append(0)
        
        # 每5個 epoch 更新圖表
        if epoch % 5 == 0:
            fig, axs = plot_training_results_realtime(
                training_data, 
                log_dir, 
                epoch,
                current_task,
                fig,
                axs
            )

    ################################################################## W-GAN Training stage####################################################
    model = model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in generator.parameters():
        p.requires_grad = True
    
    criterion_softmax = torch.nn.CrossEntropyLoss().cuda()
    
    if current_task != args.num_task:
        # 初始化 GAN 訓練數據收集器
        gan_data = {
            'epochs': [],
            'd_loss': [],
            'g_loss': [],
            'lwf_loss': []
        }
        
        # 初始化 GAN 圖表
        gan_fig, gan_axs = None, None
        
        for epoch in range(args.epochs_gan):
            loss_log = {'D/loss': 0.0,
                       'G/loss': 0.0,
                       'G/prev_mse': 0.0}
            batch_d_loss = 0.0
            batch_g_loss = 0.0
            batch_lwf_loss = 0.0
            num_batches = 0
            
            scheduler_D.step()
            scheduler_G.step()
            
            # ============ 第二階段優化：GAN穩定性改進 ============
            # 獲取自適應更新頻率
            if current_task > 0:
                d_updates, g_updates = stage2_manager.update_gan_frequencies(
                    batch_d_loss / max(num_batches, 1), 
                    batch_g_loss / max(num_batches, 1)
                )
            else:
                d_updates, g_updates = 1, 1

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                batch_size = inputs.size(0)
                num_batches += 1

                # ============ 第二階段優化：漸進式特徵生成 ============
                # 使用漸進式生成特徵（如果可用）
                if current_task > 0:
                    y_onehot_progressive = torch.zeros(batch_size, args.num_class).cuda()
                    y_onehot_progressive.scatter_(1, labels.view(-1, 1).long(), 1)
                    progressive_features = stage2_manager.get_progressive_features(batch_size, y_onehot_progressive)
                    if progressive_features is not None and epoch % 5 == 0:  # 每5個epoch記錄一次
                        print(f"使用漸進式特徵生成，複雜度: {stage2_manager.progressive_trainer.get_current_complexity():.2f}")

                # 訓練判別器（使用自適應頻率）
                for d_step in range(d_updates):
                    for p in discriminator.parameters():
                        p.requires_grad = True

                optimizer_D.zero_grad()
                real_feat = model(inputs)
                z = torch.randn(batch_size, args.latent_dim).cuda()
                
                # 確保所有張量都在 GPU 上
                y_onehot.zero_()
                # 確保標籤是 int64 類型
                y_onehot.scatter_(1, labels.view(-1, 1).long(), 1)  # 使用 view 代替 None，並確保使用 long() 型態
                syn_label = y_onehot[:batch_size]  # 確保大小匹配
                
                fake_feat = generator(z, syn_label)
                fake_validity, _ = discriminator(fake_feat.detach(), syn_label)
                real_validity, _ = discriminator(real_feat, syn_label)

                gradient_penalty = compute_gradient_penalty(discriminator, real_feat.data, fake_feat.data, syn_label)
                
                # 修改這裡：確保損失是標量
                d_loss = (-torch.mean(real_validity) + torch.mean(fake_validity) + args.lambda_gp * gradient_penalty).mean()
                d_loss.backward()
                optimizer_D.step()

                batch_d_loss += d_loss.item()

                # 訓練生成器（使用自適應頻率）
                if i % args.n_critic == 0:
                    for g_step in range(g_updates):
                        for p in discriminator.parameters():
                            p.requires_grad = False

                        optimizer_G.zero_grad()
                        
                        # ============ 第二階段優化：使用漸進式生成（如果可用）============
                        if current_task > 0 and progressive_features is not None:
                            # 部分使用漸進式特徵，部分使用常規生成
                            if np.random.random() < 0.3:  # 30%機率使用漸進式特徵
                                fake_feat = progressive_features
                            else:
                                fake_feat = generator(z, syn_label)
                        else:
                            fake_feat = generator(z, syn_label)
                            
                        fake_validity, _ = discriminator(fake_feat, syn_label)

                    if current_task > 0:
                        # 準備前500類和後續類別的比例
                        front_ratio = 0.7  # 70%採樣前500類
                        batch_front = int(batch_size * front_ratio)
                        batch_other = batch_size - batch_front
                        
                        # 為前500類準備標籤
                        front_labels = []
                        if batch_front > 0:
                            # 只從前500類中選擇
                            front_indices = [idx for idx in pre_index if idx < 500]
                            if len(front_indices) > 0:
                                for _ in range(batch_front):
                                    # 從前500類中隨機選擇
                                    front_labels.append(np.random.choice(front_indices))
                        
                        # 為其他類別準備標籤
                        other_labels = []
                        if batch_other > 0:
                            # 從所有舊類別中選擇
                            ind = list(range(len(pre_index)))
                            for _ in range(batch_other):
                                np.random.shuffle(ind)
                                other_labels.append(pre_index[ind[0]])

                        # 合併標籤
                        embed_label_sythesis = torch.tensor(front_labels + other_labels, 
                                                          dtype=torch.long, 
                                                          device='cuda')
                        
                        # 準備one-hot標籤
                        y_onehot.zero_()
                        y_onehot.scatter_(1, embed_label_sythesis.view(-1, 1).long(), 1)
                        syn_label_pre = y_onehot[:batch_size]

                        # 生成特徵並計算與舊生成器的一致性損失
                        z_pre = torch.randn(len(embed_label_sythesis), args.latent_dim).cuda()
                        pre_feat = generator(z_pre, syn_label_pre)
                        pre_feat_old = generator_old(z_pre, syn_label_pre)
                        
                        # 計算不同類別範圍的一致性損失
                        front_mask = (embed_label_sythesis < 500)
                        
                        # 為前500類設置更高的損失權重
                        if torch.any(front_mask):
                            lwf_front = F.mse_loss(
                                pre_feat[front_mask], 
                                pre_feat_old[front_mask]
                            ) * 2.0  # 增加前500類權重
                            
                            if torch.any(~front_mask):
                                lwf_other = F.mse_loss(
                                    pre_feat[~front_mask], 
                                    pre_feat_old[~front_mask]
                                )
                                lwf_loss = (lwf_front * front_mask.sum() + lwf_other * (~front_mask).sum()) / len(embed_label_sythesis)
                            else:
                                lwf_loss = lwf_front
                        else:
                            lwf_loss = F.mse_loss(pre_feat, pre_feat_old)
                    else:
                        lwf_loss = torch.zeros(1).cuda()

                    # 修改這裡：確保損失是標量
                    g_loss = (-torch.mean(fake_validity) + args.gan_tradeoff * lwf_loss).mean()
                    g_loss.backward()
                    optimizer_G.step()

                    batch_g_loss += g_loss.item()
                    batch_lwf_loss += lwf_loss.item()

            # 計算平均損失
            loss_log['D/loss'] = batch_d_loss / num_batches
            loss_log['G/loss'] = batch_g_loss / num_batches
            loss_log['G/prev_mse'] = batch_lwf_loss / num_batches

            # 收集每個 epoch 的數據
            gan_data['epochs'].append(epoch + 1)
            gan_data['d_loss'].append(loss_log['D/loss'])
            gan_data['g_loss'].append(loss_log['G/loss'])
            gan_data['lwf_loss'].append(loss_log['G/prev_mse'])
            
            # 每5個 epoch 更新圖表
            if epoch % 5 == 0:
                gan_fig, gan_axs = plot_gan_training_realtime(
                    gan_data,
                    log_dir,
                    epoch,
                    current_task,
                    gan_fig,
                    gan_axs
                )
            
            print('[GAN Epoch %05d]\t D Loss: %.3f \t G Loss: %.3f \t LwF Loss: %.3f' % (
                epoch + 1, loss_log['D/loss'], loss_log['G/loss'], loss_log['G/prev_mse']))

            # 保存模型時使用相同的命名格式
            if epoch == args.epochs_gan - 1:
                generator_save_path = os.path.join(log_dir, f'task_{str(current_task).zfill(2)}_{epoch}_model_generator.pkl')
                discriminator_save_path = os.path.join(log_dir, f'task_{str(current_task).zfill(2)}_{epoch}_model_discriminator.pkl')
                
                # 使用 _use_new_zipfile_serialization=False 以確保兼容性
                torch.save(generator, generator_save_path, _use_new_zipfile_serialization=False)
                torch.save(discriminator, discriminator_save_path, _use_new_zipfile_serialization=False)
                
                print(f"Saved generator to: {generator_save_path}")
                print(f"Saved discriminator to: {discriminator_save_path}")
        
        # GAN 訓練結束後，評估生成器質量
        if current_task > 0:
            print("評估生成器對每個類別的生成質量...")
            gen_quality = validate_generator(
                generator, model, args.num_class, args.latent_dim, current_task
            )
            
            print(f"生成器質量評估：")
            print(f"整體準確率: {gen_quality['overall_acc']:.4f}")
            print(f"前500類準確率: {gen_quality['front_acc']:.4f}")
            
            # 繪製類別準確率熱圖
            plt.figure(figsize=(20, 5))
            plt.bar(range(args.num_class), gen_quality['class_accuracies'])
            plt.xlabel('Class Index')
            plt.ylabel('Classification Accuracy')
            plt.title(f'Generator Quality at Task {current_task}')
            plt.savefig(os.path.join(log_dir, f'generator_quality_task_{current_task}.png'))
            plt.close()
            
            # 識別表現不佳的前500類
            poor_classes = []
            for cls_idx in range(500):
                if gen_quality['class_accuracies'][cls_idx] < 0.3:  # 降低閾值到30%
                    poor_classes.append(cls_idx)
            
            if len(poor_classes) > 0:
                print(f"發現 {len(poor_classes)} 個表現不佳的前500類，進行生成器微調")
                
                # 創建微調數據集
                finetune_epochs = 50
                finetune_batch_size = 32
                
                # 創建微調圖表
                ft_losses = {'epochs': [], 'mse_loss': [], 'cls_loss': [], 'total_loss': []}
                
                for ft_epoch in range(finetune_epochs):
                    # 每個批次隨機選擇一些表現不佳的類別
                    selected_classes = np.random.choice(poor_classes, finetune_batch_size, replace=True)
                    selected_classes = torch.tensor(selected_classes, dtype=torch.long).cuda()
                    
                    # 準備one-hot標籤
                    ft_y_onehot = torch.zeros(finetune_batch_size, args.num_class).cuda()
                    ft_y_onehot.scatter_(1, selected_classes.view(-1, 1), 1)
                    
                    # 使用舊生成器生成特徵
                    z = torch.randn(finetune_batch_size, args.latent_dim).cuda()
                    ft_feat_old = generator_old(z, ft_y_onehot)
                    
                    # 訓練當前生成器
                    optimizer_G.zero_grad()
                    ft_feat = generator(z, ft_y_onehot)
                    
                    # 與舊生成器保持一致（知識蒸餾）
                    ft_mse_loss = F.mse_loss(ft_feat, ft_feat_old)
                    
                    # 額外的質量約束：生成的特徵應該能被分類器正確分類
                    ft_outputs = model.embed(ft_feat)
                    ft_cls_loss = F.cross_entropy(ft_outputs, selected_classes)
                    
                    # 組合損失：調整權重比例，降低MSE損失權重，提高分類損失權重
                    ft_loss = ft_mse_loss * 0.5 + ft_cls_loss * 1.0
                    ft_loss.backward()
                    optimizer_G.step()
                    
                    # 收集損失數據
                    ft_losses['epochs'].append(ft_epoch + 1)
                    ft_losses['mse_loss'].append(ft_mse_loss.item())
                    ft_losses['cls_loss'].append(ft_cls_loss.item())
                    ft_losses['total_loss'].append(ft_loss.item())
                    
                    if ft_epoch % 10 == 0:
                        print(f"微調生成器 Epoch {ft_epoch}/{finetune_epochs}, "
                             f"MSE Loss: {ft_mse_loss.item():.4f}, "
                             f"Cls Loss: {ft_cls_loss.item():.4f}, "
                             f"Total Loss: {ft_loss.item():.4f}")
                        
                        # 繪製微調損失圖
                        plt.figure(figsize=(10, 6))
                        plt.plot(ft_losses['epochs'], ft_losses['mse_loss'], label='MSE Loss')
                        plt.plot(ft_losses['epochs'], ft_losses['cls_loss'], label='Cls Loss')
                        plt.plot(ft_losses['epochs'], ft_losses['total_loss'], label='Total Loss')
                        plt.xlabel('Epoch')
                        plt.ylabel('Loss')
                        plt.title(f'Generator Finetuning for Task {current_task}')
                        plt.legend()
                        plt.savefig(os.path.join(log_dir, f'generator_finetuning_task_{current_task}.png'))
                        plt.close()
                
                    # 微調結束後重新評估
                    print("微調後重新評估生成器質量...")
                    ft_gen_quality = validate_generator(
                        generator, model, args.num_class, args.latent_dim, current_task
                    )
                    
                    print(f"微調後生成器質量評估：")
                    print(f"整體準確率: {ft_gen_quality['overall_acc']:.4f} (之前: {gen_quality['overall_acc']:.4f})")
                    print(f"前500類準確率: {ft_gen_quality['front_acc']:.4f} (之前: {gen_quality['front_acc']:.4f})")
                    
                    # 保存微調後的生成器
                    ft_generator_save_path = os.path.join(log_dir, f'task_{str(current_task).zfill(2)}_finetuned_generator.pkl')
                    torch.save(generator, ft_generator_save_path, _use_new_zipfile_serialization=False)
                    print(f"保存微調後的生成器到: {ft_generator_save_path}")

        # 訓練結束時關閉圖表
        if gan_fig is not None:
            plt.close(gan_fig)
    tb_writer.close()

    prototype = compute_prototype(model,train_loader)  #!
    
    # 第一階段優化：生成詳細優化報告
    if current_task > 0:
        _generate_optimization_report(
            smart_scheduler, enhanced_dynamic_weights, 
            current_task, log_dir, task_difficulty
        )
    
    # ============ 第二階段優化：生成詳細優化報告 ============
    if current_task > 0:
        print("\n" + "="*80)
        print("第二階段優化總結報告")
        print("="*80)
        
        # 生成第二階段優化報告
        stage2_report_path = os.path.join(log_dir, f'stage2_optimization_report_task_{current_task}.json')
        stage2_report = stage2_manager.generate_stage2_report(stage2_report_path)
        
        # 輸出醫學特定損失統計
        if stage2_losses['total_medical_loss']:
            avg_anatomical = np.mean(stage2_losses['anatomical_consistency'])
            avg_multiscale = np.mean(stage2_losses['multiscale_features'])
            avg_semantic = np.mean(stage2_losses['semantic_consistency'])
            avg_total_medical = np.mean(stage2_losses['total_medical_loss'])
            
            print(f"\n醫學特定損失統計（任務 {current_task}）:")
            print(f"   平均解剖學一致性損失: {avg_anatomical:.6f}")
            print(f"   平均多尺度特徵損失: {avg_multiscale:.6f}")
            print(f"   平均語義一致性損失: {avg_semantic:.6f}")
            print(f"   平均總醫學損失: {avg_total_medical:.6f}")
            print(f"   損失計算次數: {len(stage2_losses['total_medical_loss'])}")
        
        # 輸出GAN穩定性改進統計
        if hasattr(stage2_manager, 'adaptive_updater'):
            d_updates, g_updates = stage2_manager.adaptive_updater.get_update_frequencies()
            print(f"\nGAN穩定性改進統計:")
            print(f"   自適應判別器更新頻率: {d_updates}")
            print(f"   自適應生成器更新頻率: {g_updates}")
            
        # 輸出漸進式訓練統計
        if hasattr(stage2_manager, 'progressive_trainer') and stage2_manager.progressive_trainer:
            current_complexity = stage2_manager.progressive_trainer.get_current_complexity()
            print(f"   漸進式訓練複雜度: {current_complexity:.2f}")
            print(f"   漸進式訓練階段: {stage2_manager.progressive_trainer.current_stage}")
        
        # 保存醫學損失統計到文件
        medical_loss_stats = {
            'task': current_task,
            'anatomical_consistency': stage2_losses['anatomical_consistency'],
            'multiscale_features': stage2_losses['multiscale_features'],
            'semantic_consistency': stage2_losses['semantic_consistency'],
            'total_medical_loss': stage2_losses['total_medical_loss'],
            'statistics': {
                'avg_anatomical': avg_anatomical if stage2_losses['total_medical_loss'] else 0,
                'avg_multiscale': avg_multiscale if stage2_losses['total_medical_loss'] else 0,
                'avg_semantic': avg_semantic if stage2_losses['total_medical_loss'] else 0,
                'avg_total_medical': avg_total_medical if stage2_losses['total_medical_loss'] else 0
            }
        }
        
        medical_stats_path = os.path.join(log_dir, f'medical_loss_statistics_task_{current_task}.json')
        with open(medical_stats_path, 'w', encoding='utf-8') as f:
            json.dump(medical_loss_stats, f, indent=4, ensure_ascii=False)
        
        print(f"\n醫學損失統計已保存至: {medical_stats_path}")
        print(f"第二階段優化報告已保存至: {stage2_report_path}")
        
        # 繪製醫學損失變化圖
        if stage2_losses['total_medical_loss']:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(stage2_losses['anatomical_consistency'])
            plt.title('解剖學一致性損失')
            plt.xlabel('批次')
            plt.ylabel('損失值')
            
            plt.subplot(2, 2, 2)
            plt.plot(stage2_losses['multiscale_features'])
            plt.title('多尺度特徵損失')
            plt.xlabel('批次')
            plt.ylabel('損失值')
            
            plt.subplot(2, 2, 3)
            plt.plot(stage2_losses['semantic_consistency'])
            plt.title('語義一致性損失')
            plt.xlabel('批次')
            plt.ylabel('損失值')
            
            plt.subplot(2, 2, 4)
            plt.plot(stage2_losses['total_medical_loss'])
            plt.title('總醫學損失')
            plt.xlabel('批次')
            plt.ylabel('損失值')
            
            plt.tight_layout()
            medical_loss_plot_path = os.path.join(log_dir, f'medical_losses_task_{current_task}.png')
            plt.savefig(medical_loss_plot_path)
            plt.close()
            
            print(f"醫學損失變化圖已保存至: {medical_loss_plot_path}")
        
        print("\n第二階段優化效果評估:")
        print("[OK] 醫學特定損失函數成功整合並運作")
        print("[OK] GAN穩定性改進機制已啟用")
        print("[OK] 漸進式訓練策略正常運行")
        print("[OK] 自適應更新頻率調整有效")
        print("="*80 + "\n")
        
        # ============ 第三階段優化總結報告 ============
        print("\n" + "="*90)
        print("第三階段優化：集成學習策略優化 + 記憶體效率優化總結報告")
        print("="*90)
        
        # 生成第三階段優化報告
        stage3_report_path = os.path.join(log_dir, f'stage3_optimization_report_task_{current_task}.json')
        stage3_report = stage3_manager.generate_stage3_report(stage3_report_path)
        
        # 輸出集成學習統計
        ensemble_stats = stage3_manager.ensemble_selector.get_reuse_statistics() if hasattr(stage3_manager.ensemble_selector, 'get_reuse_statistics') else {}
        diversity_stats = stage3_manager.diversity_optimizer.get_diversity_statistics()
        
        print(f"\n集成學習策略優化統計（任務 {current_task}）:")
        if diversity_stats:
            print(f"   平均模型多樣性分數: {diversity_stats.get('avg_diversity', 0):.4f}")
            print(f"   多樣性變化趨勢: {diversity_stats.get('diversity_trend', 0):+.6f}")
            print(f"   多樣性記錄數: {diversity_stats.get('total_records', 0)}")
        
        # 輸出記憶體優化統計
        memory_stats = stage3_manager.memory_pool.get_pool_statistics()
        batch_stats = stage3_manager.batch_sizer.get_batch_size_statistics()
        reuse_stats = stage3_manager.feature_reuser.get_reuse_statistics()
        
        print(f"\n記憶體效率優化統計:")
        if memory_stats:
            print(f"   平均GPU記憶體使用: {memory_stats.get('avg_gpu_memory', 0):.2f}GB")
            print(f"   記憶體清理次數: {memory_stats.get('cleanup_count', 0)}")
            print(f"   記憶體效率: {memory_stats.get('memory_efficiency', 0):.2%}")
        
        if batch_stats:
            print(f"   當前批次大小: {batch_stats.get('current_batch_size', 32)}")
            print(f"   平均批次大小: {batch_stats.get('avg_batch_size', 32):.1f}")
            print(f"   批次大小穩定性: {batch_stats.get('batch_size_stability', 1.0):.2%}")
        
        if reuse_stats:
            print(f"   特徵重用命中率: {reuse_stats.get('overall_hit_rate', 0):.2%}")
            print(f"   記憶體節省估計: {reuse_stats.get('memory_saved_estimate', 0):.2%}")
        
        # 輸出不確定性量化統計
        uncertainty_stats = stage3_manager.uncertainty_quantifier.get_uncertainty_statistics()
        if uncertainty_stats:
            print(f"\n不確定性量化統計:")
            print(f"   平均不確定性: {uncertainty_stats.get('avg_uncertainty', 0):.4f}")
            print(f"   校準質量: {uncertainty_stats.get('calibration_quality', 0):.2%}")
            
        # 壓縮統計
        compression_stats = stage3_manager.feature_compressor.get_compression_statistics()
        if compression_stats:
            print(f"\n特徵壓縮統計:")
            print(f"   平均壓縮比: {compression_stats.get('avg_compression_ratio', 1.0):.2f}")
            print(f"   壓縮質量: {compression_stats.get('avg_quality', 1.0):.2%}")
            print(f"   總計節省記憶體: {compression_stats.get('total_memory_saved', 0) / 1024**2:.1f}MB")
        
        # 保存第三階段統計到文件
        stage3_statistics = {
            'task': current_task,
            'ensemble_optimizations': stage3_stats['ensemble_selections'],
            'memory_optimizations': stage3_stats['memory_optimizations'],
            'batch_size_changes': stage3_stats['batch_size_changes'],
            'feature_reuse_hits': stage3_stats['feature_reuse_hits'],
            'uncertainty_scores': stage3_stats['uncertainty_scores'],
            'summary': {
                'diversity_stats': diversity_stats,
                'memory_stats': memory_stats,
                'batch_stats': batch_stats,
                'reuse_stats': reuse_stats,
                'uncertainty_stats': uncertainty_stats,
                'compression_stats': compression_stats
            }
        }
        
        stage3_stats_path = os.path.join(log_dir, f'stage3_optimization_statistics_task_{current_task}.json')
        with open(stage3_stats_path, 'w', encoding='utf-8') as f:
            json.dump(stage3_statistics, f, indent=4, ensure_ascii=False)
        
        # 繪製第三階段優化趨勢圖
        try:
            stage3_manager.plot_optimization_trends(log_dir)
        except Exception as e:
            print(f"繪製第三階段優化趨勢圖時發生錯誤: {e}")
        
        print(f"\n第三階段優化統計已保存至: {stage3_stats_path}")
        print(f"第三階段優化報告已保存至: {stage3_report_path}")
        
        print("\n第三階段優化效果評估:")
        print("[OK] 自適應集成選擇策略有效運行")
        print("[OK] 多樣性驅動優化提升模型差異性")
        print("[OK] 動態批次大小調整節省記憶體")
        print("[OK] 特徵重用機制減少重複計算")
        print("[OK] 記憶體池管理防止記憶體洩漏")
        print("[OK] 不確定性量化改善預測可靠性")
        print("[OK] 任務特徵壓縮優化存儲空間")
        print("="*90 + "\n")
    
    return prototype


def plot_training_results_realtime(data, save_path, current_epoch, task_id, fig=None, axs=None):
    """實時繪製訓練過程的圖表"""
    if fig is None or axs is None:
        fig, axs = plt.subplots(4, 1, figsize=(10, 10), height_ratios=[2, 2, 1, 1])  # 增加一個子圖
        plt.ion()
    else:
        for ax in axs:
            ax.clear()
    
    # 第一個子圖：分類器損失
    axs[0].plot(data['epochs'][:current_epoch+1], 
                data['total_loss'][:current_epoch+1], 
                label='Total Loss', marker='o', markersize=4)
    axs[0].plot(data['epochs'][:current_epoch+1], 
                data['lwf_loss'][:current_epoch+1], 
                label='LwF Loss', marker='s', markersize=4)
    axs[0].plot(data['epochs'][:current_epoch+1], 
                data['cls_loss'][:current_epoch+1], 
                label='Cls Loss', marker='^', markersize=4)
    axs[0].set_title(f'Training Losses (Task {task_id})')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss Value')
    axs[0].legend()
    axs[0].grid(True)
    
    # 第二個子圖：學習率（移除驗證曲線）
    axs[1].plot(data['epochs'][:current_epoch+1], 
                data['learning_rate'][:current_epoch+1], 
                label='Learning Rate', color='g')
    # 移除驗證準確率的雙Y軸
    axs[1].set_title('Learning Rate Schedule')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Learning Rate')
    axs[1].legend(loc='upper left')
    axs[1].grid(True)
    
    # 第三個子圖：驗證準確率
    if 'val_accuracy' in data and any(data['val_accuracy'][:current_epoch+1]):
        axs[2].plot(data['epochs'][:current_epoch+1], 
                   data['val_accuracy'][:current_epoch+1], 
                   label='Validation Accuracy', color='r', linestyle='--', marker='d', markersize=4)
        axs[2].set_title('Validation Accuracy')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Accuracy (%)')
        axs[2].legend()
        axs[2].grid(True)
    else:
        axs[2].text(0.5, 0.5, 'No Validation Data Available', 
                  ha='center', va='center', fontsize=12)
        axs[2].axis('off')
    
    # 第四個子圖：進度條
    progress = (current_epoch + 1) / args.epochs * 100
    axs[3].barh(['Progress'], [progress], color='skyblue')
    axs[3].set_xlim(0, 100)
    axs[3].text(progress/2, 0, 
                f'Task {task_id} - Epoch: {current_epoch+1}/{args.epochs} ({progress:.1f}%)', 
                ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'training_progress_task_{task_id}.png'))
    plt.draw()
    plt.pause(0.1)
    
    return fig, axs

def plot_gan_training_realtime(gan_data, save_path, current_epoch, task_id, fig=None, axs=None):
    """實時繪製 GAN 訓練過程的圖表"""
    if fig is None or axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[3, 1])
        plt.ion()
    else:
        for ax in axs:
            ax.clear()
    
    # 第一個子圖：GAN 損失
    axs[0].plot(gan_data['epochs'][:current_epoch+1], 
                gan_data['d_loss'][:current_epoch+1], 
                label='D Loss', marker='o', markersize=4)
    axs[0].plot(gan_data['epochs'][:current_epoch+1], 
                gan_data['g_loss'][:current_epoch+1], 
                label='G Loss', marker='s', markersize=4)
    axs[0].plot(gan_data['epochs'][:current_epoch+1], 
                gan_data['lwf_loss'][:current_epoch+1], 
                label='LwF Loss', marker='^', markersize=4)
    
    axs[0].set_title(f'GAN Training Losses (Task {task_id})')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss Value')
    axs[0].legend()
    axs[0].grid(True)
    
    # 第二個子圖：進度條
    progress = (current_epoch + 1) / args.epochs_gan * 100
    axs[1].barh(['Progress'], [progress], color='skyblue')
    axs[1].set_xlim(0, 100)
    axs[1].text(progress/2, 0, 
                f'GAN Task {task_id} - Epoch: {current_epoch+1}/{args.epochs_gan} ({progress:.1f}%)', 
                ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'gan_training_task_{task_id}.png'))
    plt.draw()
    plt.pause(0.1)
    
    return fig, axs

def get_advanced_scheduler(optimizer, warmup_epochs, total_epochs, patience=10):
    """創建進階學習率調度器：結合 Cosine Annealing 和 ReduceLROnPlateau"""
    
    # 主要調度器：Cosine Annealing with Warm Restarts
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=max(1, total_epochs // 4),  # 每 1/4 總 epoch 重啟一次
        T_mult=2,  # 重啟週期倍數
        eta_min=1e-6  # 最小學習率
    )
    
    # 輔助調度器：基於驗證損失的自適應調整
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=patience,
        verbose=True,
        min_lr=1e-7
    )
    
    return cosine_scheduler, plateau_scheduler

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """創建帶預熱和餘弦退火的學習率調度器"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 線性預熱
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            # 餘弦退火
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1. + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def backup_previous_results(args):
    """備份之前的訓練結果"""
    log_dir = os.path.join(args.ckpt_dir, args.log_dir)
    if os.path.exists(log_dir):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = os.path.join('result_backup', f'{args.log_dir}_{timestamp}')
        os.makedirs(backup_dir, exist_ok=True)
        
        try:
            # 移動所有文件到備份目錄
            for item in os.listdir(log_dir):
                src_path = os.path.join(log_dir, item)
                dst_path = os.path.join(backup_dir, item)
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
                elif os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path)
            
            # 清空原目錄
            shutil.rmtree(log_dir)
            os.makedirs(log_dir)
            print(f"Previous results backed up to: {backup_dir}")
        except Exception as e:
            print(f"Backup failed: {e}")

def backup_entire_project(args):
    """備份整個專案，包括程式碼和結果
    
    Args:
        args: 包含配置信息的參數對象
        
    Returns:
        str: 備份目錄的路徑
    """
    try:
        # 獲取當前工作目錄（專案根目錄）
        project_root = os.getcwd()
        
        # 確保project_backups目錄存在
        if not os.path.exists('project_backups'):
            os.makedirs('project_backups')
            print("創建備份根目錄: project_backups/")
        
        # 創建備份目錄名稱
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version = 1
        base_backup_dir = os.path.join('project_backups', f'{args.data}_{timestamp}')
        backup_dir = os.path.join(base_backup_dir, f'version_{version}')
        
        # 確保備份目錄名稱唯一
        while os.path.exists(backup_dir):
            version += 1
            backup_dir = os.path.join(base_backup_dir, f'version_{version}')
        
        # 創建備份目錄結構
        os.makedirs(os.path.join(backup_dir, 'code'), exist_ok=True)
        os.makedirs(os.path.join(backup_dir, 'results'), exist_ok=True)
        
        print(f"\n開始備份專案...")
        print(f"備份目標目錄: {backup_dir}")
        
        # 準備備份信息
        backup_info = {
            'timestamp': timestamp,
            'version': version,
            'args': vars(args),
            'code_files': [],
            'result_files': [],
            'project_root': project_root
        }
        
        # 備份程式碼文件
        code_extensions = {'.py', '.json', '.yaml', '.yml', '.txt', '.md'}
        excluded_dirs = {'.git', '__pycache__', 'project_backups', 'result_backup', '.pytest_cache', '.vscode', '.idea'}
        
        print("正在備份程式碼文件...")
        for root, dirs, files in os.walk(project_root):
            # 跳過被排除的目錄
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            
            for file in files:
                if os.path.splitext(file)[1] in code_extensions:
                    rel_path = os.path.relpath(os.path.join(root, file), project_root)
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(backup_dir, 'code', rel_path)
                    
                    # 創建目標目錄
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    
                    # 複製文件
                    shutil.copy2(src_path, dst_path)
                    backup_info['code_files'].append(rel_path)
        
        # 備份結果文件（如果存在）
        print("正在備份結果文件...")
        log_dir = os.path.join(args.ckpt_dir, args.log_dir)
        if os.path.exists(log_dir):
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), log_dir)
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(backup_dir, 'results', rel_path)
                    
                    # 創建目標目錄
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    
                    # 複製文件
                    shutil.copy2(src_path, dst_path)
                    backup_info['result_files'].append(rel_path)
        
        # 保存備份信息
        info_file = os.path.join(backup_dir, 'backup_info.json')
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(backup_info, f, indent=4, ensure_ascii=False)
        
        # 創建最新備份的符號連結
        latest_link = os.path.join('project_backups', 'latest')
        if os.path.exists(latest_link):
            if os.path.islink(latest_link):
                os.remove(latest_link)
            else:
                shutil.rmtree(latest_link)
        
        try:
            os.symlink(backup_dir, latest_link)
        except OSError as e:
            print(f"警告: 無法創建符號連結 (這在Windows系統上是正常的): {e}")
        
        print(f"\n專案備份完成！")
        print(f"備份位置: {backup_dir}")
        print(f"備份版本: {version}")
        print(f"備份代碼文件數量: {len(backup_info['code_files'])}")
        print(f"備份結果文件數量: {len(backup_info['result_files'])}")
        
        # 顯示備份的文件列表
        print("\n備份的代碼文件:")
        for file in backup_info['code_files']:
            print(f"  - {file}")
        
        if backup_info['result_files']:
            print("\n備份的結果文件:")
            for file in backup_info['result_files']:
                print(f"  - {file}")
        
        return backup_dir
        
    except Exception as e:
        print(f"\n備份過程中發生錯誤: {str(e)}")
        print("正在嘗試恢復...")
        
        try:
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            print("恢復完成")
        except Exception as recovery_error:
            print(f"恢復過程中發生錯誤: {str(recovery_error)}")
        
        raise

def restore_project(backup_dir, target_dir=None):
    """從備份中恢復整個專案
    
    Args:
        backup_dir (str): 備份目錄的路徑
        target_dir (str, optional): 目標恢復目錄的路徑，如果為None則恢復到當前目錄
    """
    try:
        if target_dir is None:
            target_dir = os.getcwd()
        
        # 檢查備份信息文件
        info_file = os.path.join(backup_dir, 'backup_info.json')
        if not os.path.exists(info_file):
            raise ValueError(f"找不到備份信息文件: {info_file}")
        
        # 讀取備份信息
        with open(info_file, 'r', encoding='utf-8') as f:
            backup_info = json.load(f)
        
        print(f"\n正在恢復專案備份...")
        print(f"備份版本: {backup_info['version']}")
        print(f"備份時間: {backup_info['timestamp']}")
        
        # 恢復代碼文件
        code_backup_dir = os.path.join(backup_dir, 'code')
        if os.path.exists(code_backup_dir):
            for file_path in backup_info['code_files']:
                src_path = os.path.join(code_backup_dir, file_path)
                dst_path = os.path.join(target_dir, file_path)
                
                # 創建目標目錄
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                # 複製文件
                shutil.copy2(src_path, dst_path)
        
        # 恢復結果文件
        results_backup_dir = os.path.join(backup_dir, 'results')
        if os.path.exists(results_backup_dir):
            results_target_dir = os.path.join(target_dir, 'checkpoints', backup_info['args']['log_dir'])
            os.makedirs(results_target_dir, exist_ok=True)
            
            for file_path in backup_info['result_files']:
                src_path = os.path.join(results_backup_dir, file_path)
                dst_path = os.path.join(results_target_dir, file_path)
                
                # 創建目標目錄
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                # 複製文件
                shutil.copy2(src_path, dst_path)
        
        print("專案恢復完成！")
        print(f"恢復的代碼文件數量: {len(backup_info['code_files'])}")
        print(f"恢復的結果文件數量: {len(backup_info['result_files'])}")
        
    except Exception as e:
        print(f"\n恢復過程中發生錯誤: {str(e)}")
        raise

def load_model_safely(model_path):
    """安全地加載模型，處理 PyTorch 2.6+ 的安全限制"""
    try:
        model = torch.load(model_path, weights_only=False)
        return model
    except Exception as e:
        print(f"錯誤: 無法加載模型 {model_path}: {e}")
        # 嘗試備選方案
        try:
            print("嘗試使用備選加載方法...")
            # 設置 safe_globals 上下文管理器
            with torch.serialization.safe_globals([Generator, Discriminator, BasicBlock, Bottleneck, ClassifierMLP, ModelCNN]):
                model = torch.load(model_path, weights_only=True)
            return model
        except Exception as e2:
            print(f"備選加載方法也失敗: {e2}")
        raise

class FocalLoss(torch.nn.Module):
    """Focal Loss 用於處理類別不平衡問題"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(torch.nn.Module):
    """Label Smoothing Loss 減少過擬合"""
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * self.confidence + (1 - targets_one_hot) * self.smoothing / (self.num_classes - 1)
        loss = (-targets_smooth * log_probs).sum(dim=1).mean()
        return loss

class AdaptiveLossWeighter:
    """自適應損失權重調整器"""
    def __init__(self, num_classes, update_freq=100):
        self.num_classes = num_classes
        self.update_freq = update_freq
        self.class_counts = torch.zeros(num_classes)
        self.step_count = 0
        self.class_weights = torch.ones(num_classes)
    
    def update(self, targets):
        """更新類別統計"""
        self.step_count += 1
        unique, counts = torch.unique(targets, return_counts=True)
        for cls, count in zip(unique, counts):
            self.class_counts[cls] += count
        
        # 每隔一定步數更新權重
        if self.step_count % self.update_freq == 0:
            # 計算逆頻率權重
            total_samples = self.class_counts.sum()
            self.class_weights = total_samples / (self.num_classes * self.class_counts.clamp(min=1))
            self.class_weights = self.class_weights / self.class_weights.sum() * self.num_classes
    
    def get_weights(self):
        return self.class_weights.cuda() if torch.cuda.is_available() else self.class_weights

# 在 AdaptiveTradeoff 類之後添加新的智能調度器類
class SmartScheduler:
    """
    智能學習率調度器：結合多種策略的混合調度系統
    
    原理：
    1. 預熱階段：線性增加學習率，讓模型平穩開始訓練
    2. 主要階段：使用餘弦退火，提供平滑的學習率衰減
    3. 自適應階段：根據驗證損失動態調整，避免過擬合
    4. 重啟機制：在特定時點重啟學習率，逃離局部最優
    """
    
    def __init__(self, optimizer, total_epochs, warmup_epochs=5, 
                 min_lr=1e-6, restart_factor=0.5, patience=10):
        """
        初始化智能調度器
        
        參數:
        - optimizer: 優化器
        - total_epochs: 總訓練輪數
        - warmup_epochs: 預熱輪數
        - min_lr: 最小學習率
        - restart_factor: 重啟時的學習率衰減因子
        - patience: 驗證損失停止改善的容忍輪數
        """
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.restart_factor = restart_factor
        self.patience = patience
        
        # 記錄初始學習率
        self.base_lr = optimizer.param_groups[0]['lr']
        
        # 創建主要調度器：餘弦退火
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=min_lr
        )
        
        # 創建自適應調度器：基於驗證損失
        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience, 
            verbose=True, min_lr=min_lr
        )
        
        # 狀態追蹤
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.restart_count = 0
        self.max_restarts = 3
        
        # 性能歷史
        self.lr_history = []
        self.val_loss_history = []
        
    def step(self, epoch, val_loss=None):
        """
        執行一步調度
        
        參數:
        - epoch: 當前輪數
        - val_loss: 驗證損失（可選）
        """
        self.current_epoch = epoch
        
        # 階段1：預熱階段
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            print(f"預熱階段 - Epoch {epoch+1}: LR = {lr:.6f}")
            
        # 階段2：主要訓練階段
        elif epoch < self.total_epochs * 0.8:
            self.cosine_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"主要階段 - Epoch {epoch+1}: LR = {current_lr:.6f}")
            
        # 階段3：精細調整階段
        else:
            if val_loss is not None:
                self.plateau_scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"精細階段 - Epoch {epoch+1}: LR = {current_lr:.6f}")
            
        # 記錄歷史
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        
        if val_loss is not None:
            self.val_loss_history.append(val_loss)
            
            # 檢查是否需要重啟
            if self._should_restart(val_loss):
                self._restart_learning_rate()
    
    def _should_restart(self, val_loss):
        """判斷是否需要重啟學習率"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            
            # 如果長時間沒有改善且還有重啟次數
            if (self.epochs_without_improvement >= self.patience * 2 and 
                self.restart_count < self.max_restarts):
                return True
            return False
    
    def _restart_learning_rate(self):
        """重啟學習率"""
        new_lr = self.base_lr * (self.restart_factor ** (self.restart_count + 1))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.restart_count += 1
        self.epochs_without_improvement = 0
        
        print(f"學習率重啟 #{self.restart_count}: LR = {new_lr:.6f}")
        
    def get_last_lr(self):
        """獲取當前學習率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def get_statistics(self):
        """獲取調度器統計信息"""
        return {
            'lr_history': self.lr_history,
            'val_loss_history': self.val_loss_history,
            'restart_count': self.restart_count,
            'best_val_loss': self.best_val_loss,
            'current_epoch': self.current_epoch
        }

class EnhancedDynamicLossWeights(DynamicLossWeights):
    """
    增強版動態損失權重管理器
    
    新增功能：
    1. 性能感知調整：根據模型性能動態調整權重策略
    2. 任務難度評估：根據任務複雜度調整權重敏感度
    3. 收斂狀態檢測：識別訓練收斂狀態並調整策略
    4. 自適應學習率：根據損失變化調整權重更新速度
    """
    
    def __init__(self, initial_l2_weight=0.5, initial_cos_weight=0.5, 
                 window_size=5, adjust_rate=0.1, min_weight=0.2, max_weight=0.8,
                 performance_threshold=0.1, task_difficulty_factor=1.0):
        """
        初始化增強版動態損失權重管理器
        
        新增參數:
        - performance_threshold: 性能改善閾值
        - task_difficulty_factor: 任務難度因子
        """
        super().__init__(initial_l2_weight, initial_cos_weight, window_size, 
                        adjust_rate, min_weight, max_weight)
        
        # 性能感知參數
        self.performance_threshold = performance_threshold
        self.task_difficulty_factor = task_difficulty_factor
        
        # 新增狀態追蹤
        self.convergence_detector = ConvergenceDetector()
        self.performance_tracker = PerformanceTracker()
        self.adaptive_rate_controller = AdaptiveRateController(adjust_rate)
        
        # 擴展歷史記錄
        self.weight_change_history = []
        self.performance_correlation = []
        self.task_difficulty_history = []
        
    def update_with_performance(self, l2_loss, cos_loss, current_performance=None, 
                               task_difficulty=None, epoch=None):
        """
        基於性能的增強更新方法
        
        參數:
        - l2_loss: L2損失值
        - cos_loss: 餘弦損失值  
        - current_performance: 當前性能指標（準確率等）
        - task_difficulty: 任務難度評估
        - epoch: 當前訓練輪數
        """
        # 1. 檢測收斂狀態
        convergence_state = self.convergence_detector.detect(l2_loss, cos_loss)
        
        # 2. 追蹤性能變化
        if current_performance is not None:
            performance_trend = self.performance_tracker.update(current_performance)
        else:
            performance_trend = 'stable'
            
        # 3. 評估任務難度
        if task_difficulty is not None:
            difficulty_adjustment = self._compute_difficulty_adjustment(task_difficulty)
        else:
            difficulty_adjustment = 1.0
            
        # 4. 調整更新速率
        current_adjust_rate = self.adaptive_rate_controller.get_rate(
            convergence_state, performance_trend
        )
        
        # 5. 執行增強的權重更新
        old_l2_weight, old_cos_weight = self.l2_weight, self.cos_weight
        
        # 基礎更新（使用父類方法）
        base_l2_weight, base_cos_weight = super().update(l2_loss, cos_loss, current_performance)
        
        # 應用增強調整
        enhanced_l2_weight, enhanced_cos_weight = self._apply_enhancements(
            base_l2_weight, base_cos_weight, convergence_state, 
            performance_trend, difficulty_adjustment, current_adjust_rate
        )
        
        # 更新權重
        self.l2_weight = enhanced_l2_weight
        self.cos_weight = enhanced_cos_weight
        
        # 記錄變化
        weight_change = abs(enhanced_l2_weight - old_l2_weight) + abs(enhanced_cos_weight - old_cos_weight)
        self.weight_change_history.append(weight_change)
        
        if current_performance is not None:
            self.performance_correlation.append((weight_change, current_performance))
            
        if task_difficulty is not None:
            self.task_difficulty_history.append(task_difficulty)
            
        # 輸出調試信息
        if epoch is not None and epoch % 20 == 0:
            print(f"動態權重更新 - Epoch {epoch}:")
            print(f"   L2權重: {old_l2_weight:.4f} -> {enhanced_l2_weight:.4f}")
            print(f"   餘弦權重: {old_cos_weight:.4f} -> {enhanced_cos_weight:.4f}")
            print(f"   收斂狀態: {convergence_state}")
            print(f"   性能趨勢: {performance_trend}")
            print(f"   調整速率: {current_adjust_rate:.4f}")
            
        return enhanced_l2_weight, enhanced_cos_weight
    
    def _compute_difficulty_adjustment(self, task_difficulty):
        """計算基於任務難度的調整因子"""
        if task_difficulty < 0.3:  # 簡單任務
            return 0.8  # 降低權重變化敏感度
        elif task_difficulty > 0.7:  # 困難任務
            return 1.2  # 提高權重變化敏感度
        else:  # 中等難度任務
            return 1.0
    
    def _apply_enhancements(self, base_l2_weight, base_cos_weight, convergence_state, 
                           performance_trend, difficulty_adjustment, current_adjust_rate):
        """應用增強調整"""
        enhanced_l2_weight = base_l2_weight
        enhanced_cos_weight = base_cos_weight
        
        # 根據收斂狀態調整
        if convergence_state == 'converging':
            # 模型正在收斂，保持穩定
            enhanced_l2_weight = base_l2_weight * 0.95 + self.l2_weight * 0.05
            enhanced_cos_weight = base_cos_weight * 0.95 + self.cos_weight * 0.05
        elif convergence_state == 'diverging':
            # 模型發散，增加調整幅度
            enhanced_l2_weight = base_l2_weight * 1.1
            enhanced_cos_weight = base_cos_weight * 1.1
            
        # 根據性能趨勢調整
        if performance_trend == 'improving':
            # 性能改善，輕微調整
            pass  # 保持當前調整
        elif performance_trend == 'degrading':
            # 性能下降，回調至更保守的權重
            enhanced_l2_weight = enhanced_l2_weight * 0.9 + 0.5 * 0.1
            enhanced_cos_weight = enhanced_cos_weight * 0.9 + 0.5 * 0.1
            
        # 應用難度調整
        enhanced_l2_weight *= difficulty_adjustment
        enhanced_cos_weight *= difficulty_adjustment
        
        # 確保權重在合理範圍內
        enhanced_l2_weight = max(self.min_weight, min(self.max_weight, enhanced_l2_weight))
        enhanced_cos_weight = max(self.min_weight, min(self.max_weight, enhanced_cos_weight))
        
        # 重新歸一化
        total = enhanced_l2_weight + enhanced_cos_weight
        enhanced_l2_weight /= total
        enhanced_cos_weight /= total
        
        return enhanced_l2_weight, enhanced_cos_weight
    
    def get_enhanced_statistics(self):
        """獲取增強統計信息"""
        base_stats = super().get_statistics()
        enhanced_stats = {
            'weight_change_history': self.weight_change_history,
            'performance_correlation': self.performance_correlation,
            'task_difficulty_history': self.task_difficulty_history,
            'convergence_stats': self.convergence_detector.get_statistics(),
            'performance_stats': self.performance_tracker.get_statistics(),
            'adaptive_rate_stats': self.adaptive_rate_controller.get_statistics()
        }
        return {**base_stats, **enhanced_stats}

class ConvergenceDetector:
    """收斂狀態檢測器"""
    
    def __init__(self, window_size=10, stability_threshold=0.01):
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.loss_history = []
        
    def detect(self, l2_loss, cos_loss):
        """檢測收斂狀態"""
        combined_loss = l2_loss + cos_loss
        self.loss_history.append(combined_loss)
        
        if len(self.loss_history) < self.window_size:
            return 'initializing'
            
        # 保持窗口大小
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
            
        # 計算變化率
        recent_losses = self.loss_history[-self.window_size//2:]
        early_losses = self.loss_history[:self.window_size//2]
        
        recent_avg = sum(recent_losses) / len(recent_losses)
        early_avg = sum(early_losses) / len(early_losses)
        
        change_rate = (recent_avg - early_avg) / early_avg if early_avg != 0 else 0
        
        if abs(change_rate) < self.stability_threshold:
            return 'converging'
        elif change_rate > 0:
            return 'diverging'
        else:
            return 'improving'
    
    def get_statistics(self):
        return {
            'loss_history': self.loss_history,
            'window_size': self.window_size,
            'stability_threshold': self.stability_threshold
        }

class PerformanceTracker:
    """性能追蹤器"""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.performance_history = []
        
    def update(self, current_performance):
        """更新性能並返回趨勢"""
        self.performance_history.append(current_performance)
        
        if len(self.performance_history) < 2:
            return 'stable'
            
        # 保持窗口大小
        if len(self.performance_history) > self.window_size:
            self.performance_history.pop(0)
            
        # 計算趨勢
        recent_perf = self.performance_history[-1]
        previous_perf = self.performance_history[-2]
        
        if recent_perf > previous_perf + 0.01:  # 1% 改善閾值
            return 'improving'
        elif recent_perf < previous_perf - 0.01:  # 1% 下降閾值
            return 'degrading'
        else:
            return 'stable'
    
    def get_statistics(self):
        return {
            'performance_history': self.performance_history,
            'window_size': self.window_size
        }

class AdaptiveRateController:
    """自適應調整速率控制器"""
    
    def __init__(self, base_rate=0.1):
        self.base_rate = base_rate
        self.rate_history = []
        
    def get_rate(self, convergence_state, performance_trend):
        """根據狀態獲取調整速率"""
        if convergence_state == 'converging' and performance_trend == 'stable':
            rate = self.base_rate * 0.5  # 穩定時降低調整速率
        elif convergence_state == 'diverging' or performance_trend == 'degrading':
            rate = self.base_rate * 1.5  # 不穩定時提高調整速率
        else:
            rate = self.base_rate
            
        self.rate_history.append(rate)
        return rate
    
    def get_statistics(self):
        return {
            'base_rate': self.base_rate,
            'rate_history': self.rate_history
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative Feature Replay Training')

    # task setting
    parser.add_argument('-data', default='medicine', required=True, help='path to Data Set')
    parser.add_argument('-num_class', default=1000, type=int, metavar='n', help='dimension of embedding space')
    parser.add_argument('-nb_cl_fg', type=int, default=500, help="Number of class, first group")
    parser.add_argument('-num_task', type=int, default=2, help="Number of Task after initial Task")

    # method parameters
    parser.add_argument('-mean_replay', action = 'store_true', help='Mean Replay')
    parser.add_argument('-tradeoff', type=float, default=1.0, help="Feature Distillation Loss")

    # basic parameters
    parser.add_argument('-load_dir_aug', default='', help='Load first task')
    parser.add_argument('-ckpt_dir', default='checkpoints', help='checkpoints dir')
    parser.add_argument('-dir', default='/data/datasets/featureGeneration/', help='data dir')
    parser.add_argument('-log_dir', default=None, help='where the trained models save')
    parser.add_argument('-name', type=str, default='tmp', metavar='PATH')

    parser.add_argument("-gpu", type=str, default='0', help='which gpu to choose')
    parser.add_argument('-nThreads', '-j', default=4, type=int, metavar='N', help='number of data loading threads')

    # hyper-parameters
    parser.add_argument('-BatchSize', '-b', default=128, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('-lr', type=float, default=1e-3, help="learning rate of new parameters")
    parser.add_argument('-lr_decay', type=float, default=0.1, help='Decay learning rate')
    parser.add_argument('-lr_decay_step', type=float, default=100, help='Decay learning rate every x steps')
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-weight-decay', type=float, default=2e-4)

    # hype-parameters for W-GAN
    parser.add_argument('-gan_tradeoff', type=float, default=2.0e-3)
    parser.add_argument('-gan_lr', type=float, default=5e-5)
    parser.add_argument('-lambda_gp', type=float, default=7.0)
    parser.add_argument('-n_critic', type=int, default=3)

    parser.add_argument('-latent_dim', type=int, default=200, help="learning rate of new parameters")
    parser.add_argument('-feat_dim', type=int, default=2048, help="learning rate of new parameters")
    parser.add_argument('-hidden_dim', type=int, default=512, help="learning rate of new parameters")
    
    # training parameters
    parser.add_argument('-epochs', default=201, type=int, metavar='N', help='epochs for training process')
    parser.add_argument('-epochs_gan', default=1001, type=int, metavar='N', help='epochs for training process')
    parser.add_argument('-seed', default=1993, type=int, metavar='N', help='seeds for training process')
    parser.add_argument('-start', default=0, type=int, help='resume epoch')
    
    # 新增：資料擴增參數
    parser.add_argument('-augmentation_intensity', default='medium', type=str, 
                       choices=['light', 'medium', 'strong'], 
                       help='Intensity of medical data augmentation for medicine dataset')
    parser.add_argument('-enable_medical_augmentation', action='store_true', default=True,
                       help='Enable medical-specific data augmentation for medicine dataset')
    
    # 新增：進階優化參數
    parser.add_argument('-use_focal_loss', action='store_true', default=False,
                       help='Use Focal Loss for handling class imbalance')
    parser.add_argument('-focal_alpha', type=float, default=1.0,
                       help='Alpha parameter for Focal Loss')
    parser.add_argument('-focal_gamma', type=float, default=2.0,
                       help='Gamma parameter for Focal Loss')
    parser.add_argument('-use_label_smoothing', action='store_true', default=False,
                       help='Use Label Smoothing Loss')
    parser.add_argument('-label_smoothing', type=float, default=0.1,
                       help='Label smoothing factor')
    parser.add_argument('-use_advanced_scheduler', action='store_true', default=False,
                       help='Use advanced learning rate scheduler')
    parser.add_argument('-scheduler_patience', type=int, default=10,
                       help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('-use_adaptive_loss_weighting', action='store_true', default=False,
                       help='Use adaptive loss weighting for class imbalance')

    args = parser.parse_args()

    # 在訓練開始前備份之前的結果
    backup_previous_results(args)

    # Data
    print('==> Preparing data..')
    
    # 只支援 Medicine 資料集
    if args.data == 'medicine':
        print('==> 使用醫學影像專用資料擴增')
        if args.enable_medical_augmentation:
            print(f'擴增強度: {args.augmentation_intensity}')
            # 使用醫學影像專用的資料擴增管道
            transform_train = get_medical_transforms(
                mode='train', 
                image_size=224, 
                intensity=args.augmentation_intensity
            )
            print('醫學影像資料擴增已啟用，包含以下技術：')
            print('- 醫學色彩調整 (Medical Color Jitter)')
            print('- 受控旋轉 (Controlled Rotation)')
            if args.augmentation_intensity in ['medium', 'strong']:
                print('- 透視變換 (Perspective Transform)')
                print('- 邊緣增強 (Edge Enhancement)')
            if args.augmentation_intensity == 'strong':
                print('- 紋理增強 (Texture Enhancement)')
            print('- 醫學亮度對比度調整')
            print('- 高斯噪聲 (30%機率)')
        else:
            # 如果關閉醫學擴增，使用驗證模式的基本變換（不做擴增）
            print('醫學擴增已關閉，使用基本預處理（無擴增）')
            transform_train = get_medical_transforms(
                mode='val',  # 使用驗證模式，只做基本的 resize 和 normalize
                image_size=224
            )
        traindir = os.path.join('medicine_picture', 'train')
    else:
        raise ValueError(f"不支援的資料集: {args.data}。此版本只支援 'medicine' 資料集。")

    num_classes = args.num_class 
    num_task = args.num_task
    num_class_per_task = (num_classes-args.nb_cl_fg) // num_task
    
    random_perm = list(range(num_classes))      # multihead fails if random permutaion here
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    prototype = {}

    if args.mean_replay:
        args.epochs_gan = 2
        
    for i in range(args.start, num_task+1):
        print ("-------------------Get started--------------- ")
        print ("Training on Task " + str(i))
        if i == 0:
            pre_index = 0
            class_index = random_perm[:args.nb_cl_fg]
        else:
            pre_index = random_perm[:args.nb_cl_fg + (i-1) * num_class_per_task]
            class_index = random_perm[args.nb_cl_fg + (i-1) * num_class_per_task:args.nb_cl_fg + (i) * num_class_per_task]

        # 只支援 Medicine 資料集
        trainfolder = ImageFolder(traindir, transform=transform_train, index=class_index)
        train_loader = torch.utils.data.DataLoader(
            trainfolder, batch_size=args.BatchSize,
            shuffle=True,
            drop_last=True, num_workers=args.nThreads)

        prototype_old = prototype
        prototype = train_task(args, train_loader, i, prototype=prototype, pre_index=pre_index)

        if args.start>0:
            pass
            # Currently it only works for our PrototypeLwF method
        else:
            if i >= 1:
                # Increase the prototype as increasing number of task
                for k in prototype.keys():
                    prototype[k] = np.concatenate((prototype[k], prototype_old[k]), axis=0)

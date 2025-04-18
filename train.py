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

from utils import RandomIdentitySampler, mkdir_if_missing, logging, display,truncated_z_sample
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
from CIFAR100 import CIFAR100
import matplotlib.pyplot as plt
from datetime import datetime
import math
import shutil
import json


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
    # One-hot encoding or attribute encoding
    if 'imagenet' in args.data or 'medicine' in args.data:
        model = models.create('resnet18_imagenet', pretrained=False, feat_dim=args.feat_dim,embed_dim=args.num_class)
    elif 'cifar' in args.data:
        model = models.create('resnet18_cifar', pretrained=False, feat_dim=args.feat_dim,embed_dim=args.num_class)


    if current_task > 0:
        try:
            # 加載模型時的路徑
            model_path = os.path.join(log_dir, f'task_{str(current_task - 1).zfill(2)}_{args.epochs - 1}_model.pkl')
            generator_path = os.path.join(log_dir, f'task_{str(current_task - 1).zfill(2)}_{args.epochs_gan - 1}_model_generator.pkl')
            discriminator_path = os.path.join(log_dir, f'task_{str(current_task - 1).zfill(2)}_{args.epochs_gan - 1}_model_discriminator.pkl')
            
            print(f"Loading previous models from:")
            print(f"Model: {model_path}")
            print(f"Generator: {generator_path}")
            print(f"Discriminator: {discriminator_path}")
            
            # 直接加載整個模型
            model = torch.load(model_path)
            model = model.cuda()
            
            model_old = deepcopy(model)
            model_old.eval()
            model_old = freeze_model(model_old)
            
            # 加載生成器和判別器
            generator = torch.load(generator_path)
            discriminator = torch.load(discriminator_path)
            
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
        if 'imagenet' in args.data or 'medicine' in args.data:
            model = models.create('resnet18_imagenet', pretrained=False, feat_dim=args.feat_dim,embed_dim=args.num_class)
        elif 'cifar' in args.data:
            model = models.create('resnet18_cifar', pretrained=False, feat_dim=args.feat_dim,embed_dim=args.num_class)
        model = model.cuda()
        
        generator = Generator(feat_dim=args.feat_dim,latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, class_dim=args.num_class).cuda()
        discriminator = Discriminator(feat_dim=args.feat_dim,hidden_dim=args.hidden_dim, class_dim=args.num_class).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)

    loss_mse = torch.nn.MSELoss(reduction='sum')

    # Loss weight for gradient penalty used in W-GAN
    lambda_gp = args.lambda_gp
    lambda_lwf = args.gan_tradeoff
    # Initialize generator and discriminator
    if current_task == 0:
        generator = Generator(feat_dim=args.feat_dim,latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, class_dim=args.num_class)
        discriminator = Discriminator(feat_dim=args.feat_dim,hidden_dim=args.hidden_dim, class_dim=args.num_class)
    else:
        generator = torch.load(os.path.join(log_dir, 'task_' + str(current_task - 1).zfill(2) + '_%d_model_generator.pkl' % int(args.epochs_gan - 1)))
        discriminator = torch.load(os.path.join(log_dir, 'task_' + str(current_task - 1).zfill(2) + '_%d_model_discriminator.pkl' % int(args.epochs_gan - 1)))
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
    
    # 使用預熱學習率調度器
    warmup_epochs = 5
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs, args.epochs)
    
    # 初始化圖表
    fig, axs = None, None
    
    # 初始化自適應權重調整器
    if current_task > 0:
        adaptive_tradeoff = AdaptiveTradeoff(args.tradeoff)
    
    # 在train_task函數開始時初始化
    dynamic_weights = DynamicLossWeights(
        initial_l2_weight=0.5,
        initial_cos_weight=0.5,
        window_size=5,
        adjust_rate=0.1,
        min_weight=0.2,
        max_weight=0.8
    )
    
    # 創建驗證資料集載入器
    # 使用與藥物圖片測試相同的預處理
    if args.data == 'medicine':
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        transform_val = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                std=std_values)
        ])
        valdir = os.path.join('medicine_picture', 'valid')
        # 使用與訓練相同的class_index
        valfolder = ImageFolder(valdir, transform=transform_val, index=class_index)
        val_loader = torch.utils.data.DataLoader(
            valfolder, batch_size=args.BatchSize,
            shuffle=False, num_workers=args.nThreads)
    elif 'cifar' in args.data:
        # CIFAR驗證資料集
        np.random.seed(args.seed)
        target_transform = np.random.permutation(num_classes)
        valset = CIFAR100(root=traindir, train=False, download=True, 
                         transform=transform_train, 
                         target_transform=target_transform, 
                         index=class_index)
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=args.BatchSize,
            shuffle=False, num_workers=args.nThreads)
    else:
        # 其他類型的驗證資料集
        val_loader = None
    
    for epoch in range(args.epochs):

        loss_log = {'C/loss': 0.0,
                    'C/loss_aug': 0.0,
                    'C/loss_cls': 0.0}
        scheduler.step()
        for i, data in enumerate(train_loader, 0):
            inputs1, labels1 = data
            inputs1, labels1 = inputs1.cuda(), labels1.cuda().long()

            # 初始化各種損失
            loss = torch.zeros(1).cuda()
            loss_cls = torch.zeros(1).cuda()
            loss_aug = torch.zeros(1).cuda()
            optimizer.zero_grad()
            
            inputs, labels = inputs1, labels1
            
            ### Classification loss
            embed_feat = model(inputs)
            if current_task == 0:
                # 第一個任務只計算分類損失
                soft_feat = model.embed(embed_feat)
                loss_cls = torch.nn.CrossEntropyLoss()(soft_feat, labels)
                loss += loss_cls
            else:
                # 後續任務需要計算舊模型的特徵
                embed_feat_old = model_old(inputs)

            ### Feature Extractor Loss
            if current_task > 0:
                # 計算兩種損失
                l2_loss = torch.dist(embed_feat, embed_feat_old, 2)
                cos_loss = 1 - F.cosine_similarity(embed_feat, embed_feat_old).mean()
                
                # 使用動態權重調整器獲取當前的權重
                l2_weight, cos_weight = dynamic_weights.update(
                    l2_loss.item(), 
                    cos_loss.item()
                )
                
                # 組合損失
                loss_aug = l2_weight * l2_loss + cos_weight * cos_loss
                
                # 使用自適應權重調整整體知識蒸餾損失的權重
                current_tradeoff = adaptive_tradeoff.update(loss_aug.item())
                loss += current_tradeoff * loss_aug * old_task_factor
            
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
                    for _ in range(args.BatchSize):
                        np.random.shuffle(ind)
                        embed_label_sythesis.append(pre_index[ind[0]])
                    
                    # 確保標籤在 GPU 上並且是正確的類型
                    embed_label_sythesis = torch.tensor(embed_label_sythesis, 
                                                      dtype=torch.long, 
                                                      device='cuda')
                    
                    y_onehot.zero_()
                    # 確保標籤是 int64 類型
                    y_onehot.scatter_(1, embed_label_sythesis.view(-1, 1).long(), 1)
                    syn_label_pre = y_onehot.cuda()

                    z = torch.randn(args.BatchSize, args.latent_dim).cuda()
                    
                    embed_sythesis = generator(z, syn_label_pre)
                
                # 確保所有張量都在 GPU 上
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
            torch.save(model, model_save_path)
            print(f"Saved model to: {model_save_path}")

        # 收集數據
        current_lr = optimizer.param_groups[0]['lr']
        training_data['epochs'].append(epoch + 1)
        training_data['total_loss'].append(loss_log['C/loss'])
        training_data['lwf_loss'].append(loss_log['C/loss_aug'])
        training_data['cls_loss'].append(loss_log['C/loss_cls'])
        training_data['learning_rate'].append(current_lr)
        
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

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                batch_size = inputs.size(0)
                num_batches += 1

                # 訓練判別器
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

                # 訓練生成器
                if i % args.n_critic == 0:
                    for p in discriminator.parameters():
                        p.requires_grad = False

                    optimizer_G.zero_grad()
                    fake_feat = generator(z, syn_label)
                    fake_validity, _ = discriminator(fake_feat, syn_label)

                    if current_task > 0:
                        ind = list(range(len(pre_index)))
                        embed_label_sythesis = []
                        for _ in range(batch_size):
                            np.random.shuffle(ind)
                            embed_label_sythesis.append(pre_index[ind[0]])

                        embed_label_sythesis = torch.tensor(embed_label_sythesis).cuda().long()
                        y_onehot.zero_()
                        # 確保標籤是 int64 類型
                        y_onehot.scatter_(1, embed_label_sythesis.view(-1, 1).long(), 1)
                        syn_label_pre = y_onehot[:batch_size]

                        pre_feat = generator(z, syn_label_pre)
                        pre_feat_old = generator_old(z, syn_label_pre)
                        lwf_loss = torch.nn.MSELoss()(pre_feat, pre_feat_old)
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
                
                torch.save(generator, generator_save_path)
                torch.save(discriminator, discriminator_save_path)
                
                print(f"Saved generator to: {generator_save_path}")
                print(f"Saved discriminator to: {discriminator_save_path}")

        # 訓練結束時關閉圖表
        if gan_fig is not None:
            plt.close(gan_fig)
    tb_writer.close()

    prototype = compute_prototype(model,train_loader)  #!
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
    parser.add_argument('-gan_tradeoff', type=float, default=1.2e-3)
    parser.add_argument('-gan_lr', type=float, default=5e-5)
    parser.add_argument('-lambda_gp', type=float, default=7.0)
    parser.add_argument('-n_critic', type=int, default=3)

    parser.add_argument('-latent_dim', type=int, default=200, help="learning rate of new parameters")
    parser.add_argument('-feat_dim', type=int, default=512, help="learning rate of new parameters")
    parser.add_argument('-hidden_dim', type=int, default=512, help="learning rate of new parameters")
    
    # training parameters
    parser.add_argument('-epochs', default=201, type=int, metavar='N', help='epochs for training process')
    parser.add_argument('-epochs_gan', default=1001, type=int, metavar='N', help='epochs for training process')
    parser.add_argument('-seed', default=1993, type=int, metavar='N', help='seeds for training process')
    parser.add_argument('-start', default=0, type=int, help='resume epoch')

    args = parser.parse_args()

    # 在訓練開始前備份之前的結果
    backup_previous_results(args)

    # Data
    print('==> Preparing data..')
    
    if args.data == 'imagenet_sub' or args.data == 'imagenet_full':
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            #transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values)
        ])
        traindir = os.path.join(args.dir, 'ILSVRC12_256', 'train')
    elif args.data == 'medicine':
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values)
        ])
        traindir = os.path.join('medicine_picture', 'train')
    elif args.data == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        traindir = args.dir + '/cifar'

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

        if args.data == 'cifar100':
            np.random.seed(args.seed)
            target_transform = np.random.permutation(num_classes)
            trainset = CIFAR100(root=traindir, train=True, download=True, transform=transform_train, target_transform = target_transform, index = class_index)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.BatchSize, shuffle=True, num_workers=args.nThreads,drop_last=True)
        elif args.data == 'medicine':
            trainfolder = ImageFolder(traindir, transform=transform_train, index=class_index)
            train_loader = torch.utils.data.DataLoader(
                trainfolder, batch_size=args.BatchSize,
                shuffle=True,
                drop_last=True, num_workers=args.nThreads)
        else:
            trainfolder = ImageFolder(traindir, transform_train, index=class_index)
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試醫學影像資料擴增功能
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from medical_data_augmentation import get_medical_transforms, MedicalImageAugmentation

def test_medical_augmentation():
    """測試醫學影像資料擴增功能"""
    print("開始測試醫學影像資料擴增功能...")
    
    # 檢查是否有測試圖片
    test_image_path = None
    if os.path.exists('medicine_picture/train'):
        # 嘗試找到一張真實的藥物圖片
        for root, dirs, files in os.walk('medicine_picture/train'):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    test_image_path = os.path.join(root, file)
                    break
            if test_image_path:
                break
    
    if test_image_path and os.path.exists(test_image_path):
        print(f"使用真實藥物圖片進行測試: {test_image_path}")
        try:
            original_img = Image.open(test_image_path).convert('RGB')
            # 調整圖片大小到224x224
            original_img = original_img.resize((224, 224))
        except Exception as e:
            print(f"讀取圖片失敗: {e}")
            print("使用合成測試圖片")
            original_img = create_synthetic_medicine_image()
    else:
        print("未找到真實藥物圖片，使用合成測試圖片")
        original_img = create_synthetic_medicine_image()
    
    # 測試不同強度的擴增
    intensities = ['light', 'medium', 'strong']
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('醫學影像資料擴增效果測試', fontsize=16)
    
    # 第一行：原圖和三種強度的概覽
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('原始圖片')
    axes[0, 0].axis('off')
    
    for i, intensity in enumerate(intensities):
        transform = get_medical_transforms(mode='train', intensity=intensity)
        try:
            # 轉換並反標準化以供顯示
            augmented_tensor = transform(original_img)
            augmented_img = tensor_to_pil(augmented_tensor)
            
            axes[0, i+1].imshow(augmented_img)
            axes[0, i+1].set_title(f'{intensity.upper()} 強度擴增')
            axes[0, i+1].axis('off')
        except Exception as e:
            axes[0, i+1].text(0.5, 0.5, f'錯誤: {str(e)}', ha='center', va='center')
            axes[0, i+1].set_title(f'{intensity.upper()} 強度擴增')
    
    # 第二到四行：展示 medium 強度的多種變化
    medium_transform = get_medical_transforms(mode='train', intensity='medium')
    
    for row in range(1, 4):
        for col in range(4):
            try:
                augmented_tensor = medium_transform(original_img)
                augmented_img = tensor_to_pil(augmented_tensor)
                
                axes[row, col].imshow(augmented_img)
                axes[row, col].set_title(f'Medium 擴增 #{(row-1)*4 + col + 1}')
                axes[row, col].axis('off')
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'錯誤: {str(e)}', ha='center', va='center')
                axes[row, col].set_title(f'Medium 擴增 #{(row-1)*4 + col + 1}')
    
    plt.tight_layout()
    plt.savefig('medical_augmentation_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 測試各種擴增技術的統計信息
    print("\n=== 測試各種擴增技術 ===")
    
    augmentation_techniques = [
        ('原始圖片', lambda x: x),
        ('醫學色彩調整', lambda x: MedicalImageAugmentation(
            medical_color_jitter=True,
            controlled_rotation=False,
            perspective_transform=False,
            medical_brightness_contrast=False,
            edge_enhancement=False,
            texture_enhancement=False
        )(x)),
        ('受控旋轉', lambda x: MedicalImageAugmentation(
            medical_color_jitter=False,
            controlled_rotation=True,
            perspective_transform=False,
            medical_brightness_contrast=False,
            edge_enhancement=False,
            texture_enhancement=False
        )(x)),
        ('完整擴增管道', get_medical_transforms('train', intensity='medium'))
    ]
    
    # 計算統計信息
    for name, transform in augmentation_techniques:
        try:
            if name == '原始圖片':
                img_array = np.array(original_img)
            else:
                if callable(transform):
                    result = transform(original_img)
                    if isinstance(result, torch.Tensor):
                        img_array = tensor_to_numpy(result)
                    else:
                        img_array = np.array(result)
                else:
                    img_array = np.array(original_img)
            
            mean_val = np.mean(img_array)
            std_val = np.std(img_array)
            print(f"{name}: 均值={mean_val:.3f}, 標準差={std_val:.3f}")
        except Exception as e:
            print(f"{name}: 測試失敗 - {str(e)}")
    
    print("\n測試完成！結果已保存至 'medical_augmentation_test_results.png'")

def create_synthetic_medicine_image():
    """創建一個合成的藥物測試圖片"""
    # 創建一個模擬藥片的圖片
    img = Image.new('RGB', (224, 224), color=(240, 240, 240))
    
    # 使用PIL繪製一個簡單的圓形藥片
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # 藥片主體（橢圓形）
    draw.ellipse([50, 80, 174, 144], fill=(255, 200, 200), outline=(200, 150, 150))
    
    # 藥片上的刻線
    draw.line([112, 80, 112, 144], fill=(180, 120, 120), width=2)
    
    # 一些紋理點
    for i in range(20):
        x = np.random.randint(60, 164)
        y = np.random.randint(90, 134)
        draw.point([x, y], fill=(220, 180, 180))
    
    return img

def tensor_to_pil(tensor):
    """將標準化的tensor轉換回PIL圖片"""
    # 反標準化
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    tensor_denorm = tensor.clone()
    for t, m, s in zip(tensor_denorm, mean, std):
        t.mul_(s).add_(m)
    
    # 轉換為PIL
    tensor_denorm = torch.clamp(tensor_denorm, 0, 1)
    img_array = tensor_denorm.permute(1, 2, 0).numpy()
    img_array = (img_array * 255).astype(np.uint8)
    
    return Image.fromarray(img_array)

def tensor_to_numpy(tensor):
    """將tensor轉換為numpy陣列"""
    if len(tensor.shape) == 3:  # C, H, W
        tensor = tensor.permute(1, 2, 0)  # H, W, C
    
    # 反標準化（如果需要）
    if tensor.min() < 0:  # 可能已經標準化
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        for t, m, s in zip(tensor.unbind(-1), mean, std):
            t.mul_(s).add_(m)
    
    tensor = torch.clamp(tensor, 0, 1)
    return (tensor.numpy() * 255).astype(np.uint8)

if __name__ == "__main__":
    test_medical_augmentation() 
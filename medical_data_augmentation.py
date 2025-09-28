import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
from torchvision.transforms import functional as F
import warnings
warnings.filterwarnings('ignore')

# 嘗試載入 OpenCV，如果沒有安裝則跳過相關功能
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("警告：未安裝 OpenCV，紋理增強功能將被停用。如需完整功能，請執行：pip install opencv-python")

class MedicalColorJitter:
    """
    醫學影像專用的顏色調整
    原理：藥物影像在不同照明條件下會有色彩變化，這個變換模擬真實拍攝環境
    """
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, img):
        # 醫學影像通常需要保持較小的色彩變化，避免影響診斷特徵
        brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)
        
        img = F.adjust_brightness(img, brightness_factor)
        img = F.adjust_contrast(img, contrast_factor)
        img = F.adjust_saturation(img, saturation_factor)
        img = F.adjust_hue(img, hue_factor)
        
        return img

class ControlledRotation:
    """
    受控的旋轉變換
    原理：藥物影像需要保持方向特徵，只進行小角度旋轉模擬手持拍攝的輕微晃動
    """
    def __init__(self, max_angle=15):
        self.max_angle = max_angle
    
    def __call__(self, img):
        # 限制旋轉角度，保持藥物的基本形狀特徵
        angle = random.uniform(-self.max_angle, self.max_angle)
        return F.rotate(img, angle, expand=False, fill=0)

class PerspectiveTransform:
    """
    透視變換
    原理：模擬不同角度拍攝藥物的效果，增加視角多樣性
    """
    def __init__(self, distortion_scale=0.2):
        self.distortion_scale = distortion_scale
    
    def __call__(self, img):
        width, height = img.size
        
        # 計算透視變換的四個角點
        half_height = height // 2
        half_width = width // 2
        topleft = [
            int(random.uniform(0, self.distortion_scale * half_width)),
            int(random.uniform(0, self.distortion_scale * half_height))
        ]
        topright = [
            int(random.uniform(width - self.distortion_scale * half_width, width)),
            int(random.uniform(0, self.distortion_scale * half_height))
        ]
        bottomright = [
            int(random.uniform(width - self.distortion_scale * half_width, width)),
            int(random.uniform(height - self.distortion_scale * half_height, height))
        ]
        bottomleft = [
            int(random.uniform(0, self.distortion_scale * half_width)),
            int(random.uniform(height - self.distortion_scale * half_height, height))
        ]
        
        startpoints = [[0, 0], [width, 0], [width, height], [0, height]]
        endpoints = [topleft, topright, bottomright, bottomleft]
        
        return F.perspective(img, startpoints, endpoints)

class GaussianNoise:
    """
    高斯噪聲
    原理：模擬相機感測器噪聲，提高模型對噪聲的魯棒性
    """
    def __init__(self, mean=0, std=0.02):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        if isinstance(tensor, Image.Image):
            tensor = F.to_tensor(tensor)
        
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0, 1)

class MedicalBrightnessContrast:
    """
    醫學影像專用亮度對比度調整
    原理：模擬不同照明條件和相機設定，保持藥物特徵可見性
    """
    def __init__(self, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def __call__(self, img):
        # 轉換為numpy陣列進行處理
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            img_array = img
        
        # 亮度調整
        brightness_factor = random.uniform(*self.brightness_range)
        img_array = np.clip(img_array * brightness_factor, 0, 255)
        
        # 對比度調整
        contrast_factor = random.uniform(*self.contrast_range)
        mean_val = np.mean(img_array)
        img_array = np.clip((img_array - mean_val) * contrast_factor + mean_val, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))

class EdgeEnhancement:
    """
    邊緣增強
    原理：突出藥物的形狀特徵，幫助模型學習到更好的邊界資訊
    """
    def __init__(self, enhancement_factor=0.3):
        self.enhancement_factor = enhancement_factor
    
    def __call__(self, img):
        if random.random() < 0.5:  # 50%機率應用邊緣增強
            # 應用邊緣增強濾鏡
            enhanced = img.filter(ImageFilter.EDGE_ENHANCE)
            # 與原圖混合
            return Image.blend(img, enhanced, self.enhancement_factor)
        return img

class TextureEnhancement:
    """
    紋理增強
    原理：使用CLAHE（對比度限制自適應直方圖均衡化）增強藥物表面紋理
    """
    def __init__(self, clip_limit=2.0):
        self.clip_limit = clip_limit
    
    def __call__(self, img):
        if not CV2_AVAILABLE:
            # 如果沒有 OpenCV，返回原圖
            return img
            
        if random.random() < 0.4:  # 40%機率應用紋理增強
            try:
                # 轉換為OpenCV格式
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                # 轉換到LAB顏色空間
                lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # 對L通道應用CLAHE
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
                l_enhanced = clahe.apply(l)
                
                # 合併通道並轉換回RGB
                enhanced_lab = cv2.merge([l_enhanced, a, b])
                enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                
                return Image.fromarray(enhanced_rgb)
            except Exception as e:
                print(f"紋理增強失敗，返回原圖: {e}")
                return img
        return img

class MedicalImageAugmentation:
    """
    醫學影像增強主類
    整合多種增強技術，專門針對藥物影像分類任務
    """
    def __init__(self, enable_all=True, **kwargs):
        self.transforms = []
        
        if enable_all or kwargs.get('medical_color_jitter', True):
            self.transforms.append(MedicalColorJitter())
        
        if enable_all or kwargs.get('controlled_rotation', True):
            self.transforms.append(ControlledRotation())
        
        if enable_all or kwargs.get('perspective_transform', True):
            self.transforms.append(PerspectiveTransform())
        
        if enable_all or kwargs.get('medical_brightness_contrast', True):
            self.transforms.append(MedicalBrightnessContrast())
        
        if enable_all or kwargs.get('edge_enhancement', True):
            self.transforms.append(EdgeEnhancement())
        
        if enable_all or kwargs.get('texture_enhancement', True):
            self.transforms.append(TextureEnhancement())
    
    def __call__(self, img):
        # 隨機選擇1-3個變換應用
        num_transforms = random.randint(1, min(3, len(self.transforms)))
        selected_transforms = random.sample(self.transforms, num_transforms)
        
        for transform in selected_transforms:
            img = transform(img)
        
        return img

class MedicalTransformPipeline:
    """
    完整的醫學影像變換管道
    包含基本變換和醫學特定增強
    """
    def __init__(self, mode='train', image_size=224, intensity='medium'):
        """
        Args:
            mode: 'train' 或 'val'
            image_size: 目標影像大小
            intensity: 增強強度 'light', 'medium', 'strong'
        """
        self.mode = mode
        self.image_size = image_size
        self.intensity = intensity
        
        # 基本變換
        if mode == 'train':
            self.basic_transforms = [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.3),  # 降低水平翻轉機率，保持藥物方向
            ]
        else:
            self.basic_transforms = [
                transforms.Resize(int(image_size * 1.143)),  # 256 for 224
                transforms.CenterCrop(image_size),
            ]
        
        # 醫學增強（僅訓練時使用）
        if mode == 'train':
            if intensity == 'light':
                self.medical_aug = MedicalImageAugmentation(
                    medical_color_jitter=True,
                    controlled_rotation=True,
                    perspective_transform=False,
                    medical_brightness_contrast=True,
                    edge_enhancement=False,
                    texture_enhancement=False
                )
            elif intensity == 'medium':
                self.medical_aug = MedicalImageAugmentation(
                    medical_color_jitter=True,
                    controlled_rotation=True,
                    perspective_transform=True,
                    medical_brightness_contrast=True,
                    edge_enhancement=True,
                    texture_enhancement=False
                )
            else:  # strong
                self.medical_aug = MedicalImageAugmentation(enable_all=True)
        else:
            self.medical_aug = None
        
        # 噪聲增強（僅訓練時使用）
        self.gaussian_noise = GaussianNoise() if mode == 'train' else None
        
        # 最終變換
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        self.final_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values, std=std_values)
        ]
    
    def __call__(self, img):
        # 基本變換
        for transform in self.basic_transforms:
            img = transform(img)
        
        # 醫學增強（訓練時）
        if self.medical_aug and random.random() < 0.8:  # 80%機率應用醫學增強
            img = self.medical_aug(img)
        
        # 轉換為Tensor
        img_tensor = transforms.ToTensor()(img)
        
        # 噪聲增強（訓練時）
        if self.gaussian_noise and random.random() < 0.3:  # 30%機率添加噪聲
            img_tensor = self.gaussian_noise(img_tensor)
        
        # 標準化
        img_tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(img_tensor)
        
        return img_tensor

def get_medical_transforms(mode='train', image_size=224, intensity='medium'):
    """
    獲取醫學影像變換管道的便捷函數
    
    Args:
        mode: 'train' 或 'val'
        image_size: 目標影像大小
        intensity: 增強強度 'light', 'medium', 'strong'
    
    Returns:
        Transform pipeline
    """
    return MedicalTransformPipeline(mode=mode, image_size=image_size, intensity=intensity)

# 測試函數
def test_augmentation():
    """測試增強功能"""
    import matplotlib.pyplot as plt
    
    # 創建一個測試影像
    test_img = Image.new('RGB', (224, 224), color='white')
    
    # 測試各種變換
    transforms_to_test = [
        ('Original', lambda x: x),
        ('Medical Color Jitter', MedicalColorJitter()),
        ('Controlled Rotation', ControlledRotation()),
        ('Perspective Transform', PerspectiveTransform()),
        ('Medical Brightness/Contrast', MedicalBrightnessContrast()),
        ('Edge Enhancement', EdgeEnhancement()),
        ('Texture Enhancement', TextureEnhancement()),
        ('Complete Pipeline', get_medical_transforms('train'))
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (name, transform) in enumerate(transforms_to_test):
        try:
            transformed = transform(test_img)
            if isinstance(transformed, torch.Tensor):
                # 反標準化顯示
                transformed = transformed.permute(1, 2, 0)
                transformed = transformed * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                transformed = torch.clamp(transformed, 0, 1)
                transformed = transformed.numpy()
            
            axes[i].imshow(transformed)
            axes[i].set_title(name)
            axes[i].axis('off')
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
            axes[i].set_title(name)
    
    plt.tight_layout()
    plt.savefig('medical_augmentation_test.png')
    plt.show()

if __name__ == "__main__":
    test_augmentation() 
# -*- coding: utf-8 -*-
"""
進階優化配置文件
專門為醫學影像分類任務設計的優化策略
"""

# 學習率優化配置
LEARNING_RATE_CONFIGS = {
    'conservative': {
        'base_lr': 1e-4,
        'warmup_epochs': 5,
        'scheduler_patience': 15,
        'min_lr': 1e-7
    },
    'aggressive': {
        'base_lr': 3e-4,
        'warmup_epochs': 3,
        'scheduler_patience': 8,
        'min_lr': 1e-6
    },
    'adaptive': {
        'base_lr': 2e-4,
        'warmup_epochs': 5,
        'scheduler_patience': 10,
        'min_lr': 1e-6
    }
}

# 損失函數配置
LOSS_CONFIGS = {
    'balanced_dataset': {
        'use_focal_loss': False,
        'use_label_smoothing': True,
        'label_smoothing': 0.1
    },
    'imbalanced_dataset': {
        'use_focal_loss': True,
        'focal_alpha': 1.0,
        'focal_gamma': 2.0,
        'use_label_smoothing': True,
        'label_smoothing': 0.05
    },
    'highly_imbalanced': {
        'use_focal_loss': True,
        'focal_alpha': 2.0,
        'focal_gamma': 3.0,
        'use_adaptive_loss_weighting': True,
        'use_label_smoothing': False
    }
}

# 資料擴增配置
AUGMENTATION_CONFIGS = {
    'light_augmentation': {
        'augmentation_intensity': 'light',
        'enable_medical_augmentation': True
    },
    'medium_augmentation': {
        'augmentation_intensity': 'medium',
        'enable_medical_augmentation': True
    },
    'strong_augmentation': {
        'augmentation_intensity': 'strong',
        'enable_medical_augmentation': True
    }
}

# 推薦的組合配置
RECOMMENDED_CONFIGS = {
    'high_accuracy': {
        'learning_rate': 'conservative',
        'loss_function': 'balanced_dataset',
        'augmentation': 'medium_augmentation',
        'use_advanced_scheduler': True,
        'description': '追求高準確度的穩定配置'
    },
    'fast_training': {
        'learning_rate': 'aggressive',
        'loss_function': 'balanced_dataset',
        'augmentation': 'light_augmentation',
        'use_advanced_scheduler': False,
        'description': '快速訓練的配置'
    },
    'robust_generalization': {
        'learning_rate': 'adaptive',
        'loss_function': 'imbalanced_dataset',
        'augmentation': 'strong_augmentation',
        'use_advanced_scheduler': True,
        'description': '強化泛化能力的配置'
    }
}

def get_config(config_name):
    """獲取推薦配置"""
    if config_name not in RECOMMENDED_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}")
    
    config = RECOMMENDED_CONFIGS[config_name]
    
    # 組合各部分配置
    result = {}
    result.update(LEARNING_RATE_CONFIGS[config['learning_rate']])
    result.update(LOSS_CONFIGS[config['loss_function']])
    result.update(AUGMENTATION_CONFIGS[config['augmentation']])
    result['use_advanced_scheduler'] = config['use_advanced_scheduler']
    result['description'] = config['description']
    
    return result

def print_config(config_name):
    """打印配置詳情"""
    config = get_config(config_name)
    print(f"\n=== {config_name.upper()} 配置 ===")
    print(f"描述: {config['description']}")
    print("\n參數設定:")
    for key, value in config.items():
        if key != 'description':
            print(f"  {key}: {value}")
    print()

if __name__ == '__main__':
    # 展示所有推薦配置
    for config_name in RECOMMENDED_CONFIGS.keys():
        print_config(config_name) 
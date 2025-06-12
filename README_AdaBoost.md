# AdaBoost 集成學習用於持續學習模型

## 概述

本專案實現了基於AdaBoost算法的集成學習方法，專門用於整合您訓練的11個持續學習模型。這個方法將每個專門的模型作為弱學習器，通過AdaBoost的權重計算機制來優化整體預測性能。

## 核心原理

### 1. AdaBoost 在持續學習中的應用

傳統的AdaBoost算法主要用於訓練一系列弱學習器，而我們的實現將其應用於已經訓練好的專門模型：

```
模型 1: 負責類別 0-499   (500個類別)
模型 2: 負責類別 500-549 (50個類別)
模型 3: 負責類別 550-599 (50個類別)
...
模型11: 負責類別 950-999 (50個類別)
```

### 2. 權重計算機制

#### 傳統AdaBoost權重公式：
```
α_m = 0.5 * ln((1 - ε_m) / ε_m)
```
其中：
- `α_m` 是第m個模型的權重
- `ε_m` 是第m個模型在驗證集上的錯誤率

#### 權重正規化：
```
w_m = α_m / Σ(α_i)
```

### 3. 集成預測策略

實現了三種集成策略：

1. **加權投票 (Weighted Voting)**：
   ```
   prediction = argmax(Σ(w_i * P_i))
   ```

2. **自適應選擇 (Adaptive Selection)**：
   - 優先選擇信心度最高的模型
   - 檢查預測類別是否在該模型的專業範圍內
   - 如果不在範圍內，退回到加權投票

3. **信心度投票 (Confidence Voting)**：
   - 使用模型預測的信心度作為動態權重
   - 適合處理模型性能變化的情況

## 使用方法

### 1. 基本使用

```bash
python adaboost_ensemble.py \
    -data medicine \
    -models_dir checkpoints/your_log_dir \
    -num_class 1000 \
    -nb_cl_fg 500 \
    -num_task 10 \
    -epochs 200 \
    -batch_size 128 \
    -gpu 0
```

### 2. 進階參數設定

```bash
python adaboost_ensemble.py \
    -data medicine \
    -models_dir checkpoints/your_log_dir \
    -weight_method adaboost \
    -ensemble_strategy weighted_voting \
    -save_dir my_adaboost_results
```

### 3. 參數說明

#### 基本參數：
- `-data`: 資料集類型 (`medicine`, `cifar100`)
- `-models_dir`: 包含11個模型檔案的目錄
- `-num_class`: 總類別數 (預設: 1000)
- `-nb_cl_fg`: 第一個模型的類別數 (預設: 500)
- `-num_task`: 後續任務數 (預設: 10)

#### AdaBoost特定參數：
- `-weight_method`: 權重計算方法
  - `adaboost`: 傳統AdaBoost權重計算
  - `accuracy_based`: 基於準確率的權重
  - `confidence_based`: 基於信心度的權重

- `-ensemble_strategy`: 集成策略
  - `weighted_voting`: 加權投票
  - `adaptive_selection`: 自適應選擇
  - `confidence_voting`: 信心度投票

## 輸出結果

執行完成後會在指定目錄生成以下文件：

### 1. 性能報告
- `performance_report.json`: 詳細的數據報告
- `performance_summary.txt`: 可讀的摘要報告

### 2. 視覺化圖表
- `overall_performance.png`: 整體性能比較圖
- `class_wise_performance.png`: 類別級別性能分析
- `model_weights.png`: 模型權重分布圖
- `confusion_matrix.png`: 混淆矩陣（前100類）

### 3. 範例輸出

```
模型權重計算完成：
模型  1 (類別   0-499): 錯誤率=0.1234, 權重=0.2876
模型  2 (類別 500-549): 錯誤率=0.2345, 權重=0.0987
...

集成預測完成！
集成準確率: 0.8765
模型  1 準確率: 0.8543
模型  2 準確率: 0.7654
...

AdaBoost 集成學習完成！
集成準確率: 0.8765
最佳個別模型: 0.8543
改善幅度: 0.0222
```

## 理論優勢

### 1. 相比於Stacking的優勢

| 特性 | Stacking | AdaBoost |
|------|----------|----------|
| 模型選擇 | 動態選擇單一最佳模型 | 加權組合所有模型 |
| 錯誤處理 | 依賴meta-learner | 基於錯誤率的自動權重調整 |
| 魯棒性 | 可能過度依賴某個模型 | 分散風險，更加穩定 |
| 理論基礎 | 經驗性 | 有嚴格的理論保證 |

### 2. 持續學習中的特殊優勢

1. **災難性遺忘緩解**：通過加權組合，保護舊知識
2. **專業化與泛化平衡**：每個模型專注特定類別，集成提供泛化能力
3. **自適應性**：根據各模型表現動態調整權重
4. **可解釋性**：清楚展示每個模型的貢獻程度

## 實驗建議

### 1. 超參數調優

1. **權重計算方法比較**：
   ```bash
   # 測試不同權重計算方法
   for method in adaboost accuracy_based confidence_based; do
       python adaboost_ensemble.py -weight_method $method
   done
   ```

2. **集成策略比較**：
   ```bash
   # 測試不同集成策略
   for strategy in weighted_voting adaptive_selection confidence_voting; do
       python adaboost_ensemble.py -ensemble_strategy $strategy
   done
   ```

### 2. 性能分析

1. **檢查模型權重分布**：關注是否有模型權重過低
2. **分析類別級別性能**：識別哪些類別從集成中受益最大
3. **比較改善幅度**：評估AdaBoost相對於最佳單一模型的提升

### 3. 故障排除

如果遇到以下問題：

1. **模型加載失敗**：
   - 檢查模型文件路徑是否正確
   - 確認PyTorch版本兼容性

2. **記憶體不足**：
   - 減少batch_size
   - 使用較少的工作進程

3. **權重異常**：
   - 檢查驗證集是否包含所有類別
   - 嘗試不同的權重計算方法

## 擴展功能

### 1. 動態權重調整

可以實現在線學習，根據新數據動態調整模型權重：

```python
# 示例：動態更新權重
ensemble.update_weights(new_validation_data)
```

### 2. 模型剪枝

自動移除表現過差的模型：

```python
# 移除權重低於閾值的模型
ensemble.prune_models(threshold=0.01)
```

### 3. 混合策略

結合多種集成策略：

```python
# 根據類別動態選擇策略
ensemble.predict_hybrid(test_loader)
```

這個AdaBoost實現為您的持續學習系統提供了一個強大而靈活的集成框架，能夠有效整合11個專門模型的優勢，同時保持良好的可解釋性和擴展性。 
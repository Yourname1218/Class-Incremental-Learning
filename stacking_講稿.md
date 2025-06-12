# Stacking 集成學習方法講稿

## 1. Stacking流程圖 - 方法

### 🎯 核心執行流程

Stacking方法的執行分為**兩個主要階段**和**兩種執行模式**：

#### **階段一：基礎模型準備**
```
載入11個基礎模型 → 搜尋task_XX_200_model.pkl → Task 0-10專精模型
```
- **Task 0**: 負責前500個基礎類別，經過200輪分類器訓練
- **Task 1-10**: 各自負責50個增量類別，專精於特定範圍
- **模型特點**: 每個模型在其專精領域具有最強表現能力

#### **階段二：執行模式分支**

**🔄 訓練模式流程**：
```
準備訓練數據 → 收集基礎模型預測 → 分割數據集(70:20:10) → 
創建元模型 → 訓練30 epochs → 保存最佳模型 → 評估性能
```

**📊 評估模式流程**：
```
載入已訓練元模型 → 準備測試數據 → 收集預測 → 
元模型組合 → 性能評估 → 輸出結果
```

### 🏗️ 技術實現架構

**數據處理管線**：
- **圖像預處理**: 224×224 CenterCrop + ImageNet標準正規化
- **批次處理**: 32張圖像/批次，優化GPU記憶體使用
- **並行推理**: 11個模型同時處理，提升計算效率

**預測收集機制**：
```python
# 每個模型輸出1000維logits
features = model(images)
logits = model.embed(features)
# 形狀: [batch_size, 1000]
```

**權重學習策略**：
```python
# 元模型權重矩陣 [11 × 1000]
self.weights = nn.Parameter(torch.ones(num_models, num_classes))
# Task 0前500類權重 = 1.5
# Task 1-10各自範圍權重 = 3.0
```

---

## 2. Stacking - 概念

### 🧠 核心理論基礎

Stacking是一種**元學習(Meta-Learning)集成方法**，其核心思想是：
> "訓練一個元模型來學習如何最佳地組合多個基礎模型的預測結果"

### 🎨 設計哲學

#### **多專家協作模式**
- **專業分工**: 每個Task模型在特定類別範圍內是專家
- **知識互補**: Task 0提供基礎知識，Task 1-10提供專精能力
- **智能組合**: 元模型學習在不同情況下信任哪個專家

#### **自適應權重學習**
```
傳統加權平均：固定權重，無法適應複雜場景
Stacking方法：動態權重，基於數據學習最佳組合
```

### 🔬 與其他方法的核心差異

| 比較維度 | Stacking | 傳統投票 | AdaBoost |
|---------|----------|----------|----------|
| **權重來源** | 數據驅動學習 | 人工設定或均等 | 錯誤率計算 |
| **適應能力** | 高度自適應 | 固定不變 | 基於錯誤率 |
| **複雜度** | 需要訓練階段 | 簡單直接 | 中等複雜 |
| **精確度** | 通常最高 | 基準線 | 較好 |

### 🎯 針對持續學習的特殊設計

**災難性遺忘解決方案**：
- **基礎知識保護**: Task 0模型權重確保前500類知識不丟失
- **增量知識增強**: Task 1-10模型權重提升新類別學習效果
- **平衡機制**: 元模型動態平衡新舊知識的重要性

**專業範圍劃分**：
```
Task 0: 類別 0-499 (基礎領域，覆蓋率50%)
Task 1: 類別 500-549 (專精領域1，覆蓋率5%)
Task 2: 類別 550-599 (專精領域2，覆蓋率5%)
...
Task 10: 類別 950-999 (專精領域10，覆蓋率5%)
```

---

## 3. Stacking - 元模型訓練

### 🏋️ 訓練架構設計

#### **StackingMetaModel 類別結構**
```python
class StackingMetaModel(nn.Module):
    def __init__(self, num_models=11, num_classes=1000):
        # 核心權重矩陣：每個模型對每個類別的重要性
        self.weights = nn.Parameter(torch.ones(num_models, num_classes))
```

#### **智能權重初始化策略**
```python
# Task 0：基礎模型，前500類適中權重
self.weights[0, :500] = 1.5

# Task 1-10：專精模型，各自範圍高權重
for i in range(1, num_models):
    start_idx = 500 + (i-1) * 50
    end_idx = start_idx + 50
    self.weights[i, start_idx:end_idx] = 3.0
```

### 🎛️ 訓練過程詳解

#### **第一步：預測收集階段**
```python
def collect_predictions(models, dataloader):
    # 收集11個模型對所有樣本的預測
    # 輸出格式：List[Tensor] 長度=11，每個Tensor形狀=[樣本數, 1000]
```

**關鍵技術特點**：
- **記憶體優化**: 預測結果移到CPU，節省GPU記憶體
- **批次處理**: 逐批次收集，避免記憶體溢出
- **格式統一**: 確保所有預測格式一致，便於後續處理

#### **第二步：數據分割策略**
```python
# 70%用於元模型訓練
# 20%用於驗證和調整
# 10%用於最終測試
train_split = 0.7
val_ratio_in_train = 0.2
```

#### **第三步：元模型訓練循環**

**優化器配置**：
```python
optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(mode='min', factor=0.5, patience=5)
criterion = nn.CrossEntropyLoss()
```

**訓練過程**：
```python
# 前向傳播：線性加權組合
weighted_preds = (predictions * self.weights).sum(dim=1)

# 反向傳播：更新權重矩陣
loss = criterion(weighted_preds, labels)
loss.backward()
optimizer.step()
```

### 📊 關鍵性能指標

#### **訓練監控指標**
- **訓練損失**: 每個epoch的CrossEntropyLoss
- **驗證準確率**: 用於提早停止和模型選擇
- **學習率調整**: 基於驗證損失的自動調整

#### **權重分析**
```python
# 監控權重變化趨勢
print(f"Task 0平均權重: {meta_model.weights[0].mean():.4f}")
print(f"Task 1-10平均權重: {meta_model.weights[1:].mean():.4f}")
```

#### **最佳模型保存機制**
```python
if val_acc > best_acc:
    best_acc = val_acc
    best_model_state = meta_model.state_dict().copy()
    print(f'新的最佳驗證準確率: {best_acc:.2f}%')
```

### 🎯 訓練策略優化

**防止過擬合措施**：
- **學習率調度**: ReduceLROnPlateau自動降低學習率
- **提早停止**: 基於驗證準確率選擇最佳模型
- **批次大小**: 32的適中批次大小平衡訓練穩定性

**記憶體管理**：
- **GPU/CPU分配**: 預測在CPU處理，訓練在GPU執行
- **批次處理**: 避免一次性載入所有數據
- **動態清理**: 及時釋放不需要的張量

### 📈 預期訓練效果

**權重學習目標**：
- Task 0模型在前500類獲得較高權重
- Task 1-10模型在各自專精範圍獲得最高權重
- 跨領域類別的權重分配趨於平衡

**性能提升預期**：
- 整體準確率：目標超越51.21% (最佳單一模型)
- 前500類：保持或提升Task 0的75.75%準確率
- 後500類：顯著提升Task 1-10的低準確率問題

---

## 🎉 總結

Stacking方法通過**兩階段學習**和**智能權重分配**，成功解決了持續學習中的災難性遺忘問題。元模型的訓練過程體現了現代機器學習中**數據驅動**和**自適應學習**的核心思想，為醫學影像分類的持續學習提供了有效的解決方案。 
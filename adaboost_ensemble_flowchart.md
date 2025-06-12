# AdaBoost Ensemble 集成學習流程圖

## 核心AdaBoost方法流程

```mermaid
flowchart TD
    A[開始] --> B[載入11個基礎模型<br/>搜尋 task_XX_199_model.pkl<br/>Task 0-10 弱學習器]
    B --> C[定義類別範圍<br/>Task 0: 類別 0-499<br/>Task 1-10: 各50類]
    C --> D[準備驗證數據<br/>計算模型權重用]
    
    D --> E[計算每個模型錯誤率<br/>只在專業類別範圍內評估<br/>避免跨領域干擾]
    E --> F[應用AdaBoost權重公式<br/>weight = 0.5 * ln1-error/error<br/>錯誤率越低權重越高]
    F --> G[範圍樣本比例調整<br/>adjusted_weight = weight * sqrt_ratio<br/>平衡樣本數量差異]
    
    G --> H[準備測試數據<br/>執行集成預測]
    H --> I{選擇集成策略}
    
    I -->|加權投票| J[Weighted Voting<br/>加權求和所有模型預測<br/>final = Sum_weight_i * pred_i]
    I -->|自適應選擇| K[Adaptive Selection<br/>根據類別範圍和信心度<br/>選擇最適合的模型]
    I -->|信心度投票| L[Confidence Voting<br/>基於預測信心度<br/>動態分配權重]
    
    J --> M[輸出最終預測結果]
    K --> M
    L --> M
    
    M --> N[性能分析與視覺化<br/>準確率/混淆矩陣/權重分布]
```

## AdaBoost核心概念

```mermaid
flowchart LR
    A[Task 0 弱學習器<br/>專業: 類別 0-499<br/>基礎模型 500類] --> D[AdaBoost集成器<br/>錯誤率導向權重分配<br/>專業範圍內評估]
    B[Task 1-10 弱學習器<br/>各專業: 50類範圍<br/>增量學習模型] --> D
    C[11個專業模型<br/>覆蓋全部1000類<br/>避免跨領域干擾] --> D
    
    D --> E[錯誤率評估<br/>error = 錯誤樣本數/總樣本數<br/>僅在專業範圍內計算]
    E --> F[AdaBoost權重公式<br/>alpha = 0.5 * ln1-epsilon/epsilon<br/>epsilon為錯誤率]
    F --> G[範圍調整權重<br/>考慮樣本數量比例<br/>平方根調整機制]
    
    H[權重計算方法] --> I[AdaBoost經典公式<br/>基於錯誤率計算]
    I --> J[Accuracy Based<br/>基於準確率平方]
    J --> K[Confidence Based<br/>基於平均信心度]
```

## 權重計算過程

```mermaid
flowchart TD
    A[收集所有模型預測<br/>11個模型對驗證集的預測<br/>shape: 11 * samples * 1000] --> B[提取專業範圍預測<br/>每個模型只在其專業類別範圍內<br/>避免跨領域評估偏差]
    
    B --> C[計算專業範圍錯誤率<br/>mask = true_labels in class_range<br/>error = mean predictions != labels]
    C --> D[應用AdaBoost公式<br/>確保 1e-10 <= error <= 0.999<br/>避免除零和權重爆炸]
    D --> E[計算基礎權重<br/>weight = 0.5 * ln1-error/error<br/>錯誤率越低權重越高]
    
    E --> F[樣本比例調整<br/>ratio = 專業範圍樣本數/總樣本數<br/>調整係數 = sqrt_ratio]
    F --> G[最終權重<br/>final_weight = weight * adjustment<br/>平衡不同模型的樣本數差異]
    
    G --> H[權重正規化<br/>確保所有權重為正數<br/>便於後續加權投票]
```

## 集成預測策略

### 1. 加權投票策略

```mermaid
flowchart TD
    A[測試樣本輸入<br/>224*224 RGB圖像<br/>CenterCrop + Normalize] --> B[11個模型並行預測<br/>每個輸出1000維機率分布<br/>softmax歸一化]
    
    B --> C[應用AdaBoost權重<br/>weighted_sum = Sum_w_i * p_i<br/>w_i為第i個模型權重]
    C --> D[選擇最大機率類別<br/>final_class = argmax weighted_sum<br/>輸出最終預測結果]
```

### 2. 自適應選擇策略

```mermaid
flowchart TD
    A[獲取所有模型預測和信心度<br/>predictions: 11 * samples * 1000<br/>confidences: 11 * samples] --> B[找出最高信心度模型<br/>best_model = argmax confidences<br/>predicted_class = argmax prediction]
    
    B --> C{預測類別是否在<br/>該模型專業範圍內?}
    C -->|是| D[使用該模型預測<br/>充分利用專業優勢<br/>提高預測準確性]
    C -->|否| E[回退到加權投票<br/>避免跨領域預測錯誤<br/>使用集成智慧]
    
    D --> F[輸出最終結果]
    E --> F
```

### 3. 信心度投票策略

```mermaid
flowchart TD
    A[收集每個樣本的<br/>模型預測和信心度<br/>動態權重分配] --> B[計算信心度權重<br/>norm_conf_i = conf_i / Sum_conf_j<br/>歸一化信心度作為權重]
    
    B --> C[信心度加權求和<br/>weighted_pred = Sum_norm_conf_i * pred_i<br/>動態調整模型貢獻]
    C --> D[選擇最大機率類別<br/>adaptive_class = argmax weighted_pred<br/>基於實時信心度的決策]
```

## 方法特點與技術優勢

**專業範圍評估**:
- 每個模型只在其專業類別範圍內評估
- Task 0: 500個基礎類別 (0-499)
- Task 1-10: 各50個增量類別
- 避免跨領域評估造成的偏差

**錯誤率導向權重分配**:
- AdaBoost經典公式: α = 0.5 × ln((1-ε)/ε)
- 錯誤率範圍限制: [1e-10, 0.999]
- 樣本比例平方根調整機制
- 動態權重正規化

**多策略集成**:
- 加權投票: 穩定的基礎策略
- 自適應選擇: 專業範圍內優先策略
- 信心度投票: 動態調整權重策略
- 支援運行時策略切換

**記憶體優化**:
- 批次大小: 128 (可調整)
- CPU預測收集節省GPU記憶體
- 並行模型推理提升效率
- 分批處理大型數據集

## 性能分析與評估指標

**整體性能比較**:
- 集成準確率 vs 最佳個別模型
- 11個模型的個別準確率分析
- 權重分布視覺化
- 改善幅度量化 (集成-最佳個別)

**類別級別分析**:
- 前500類 vs 後500類性能對比
- 每個類別的詳細準確率
- 專業範圍內的模型表現
- 跨領域預測失誤分析

**模型權重分析**:
- AdaBoost權重分布圖
- 權重與錯誤率的關係
- 專業範圍樣本數影響
- 權重調整效果驗證

**混淆矩陣分析**:
- 前100類的詳細混淆矩陣
- 類別間的預測混淆模式
- 專業模型的預測偏好
- 集成效果的視覺化驗證

## 核心參數配置總結

| 參數項目 | 數值/設定 | 說明 |
|---------|-----------|------|
| 弱學習器數量 | 11個 | Task 0-10 專業模型 |
| 總類別數 | 1000 | 完整醫學影像分類 |
| Task 0 類別 | 500 (0-499) | 基礎類別範圍 |
| 增量Task類別 | 50/每個 | Task 1-10 各負責50類 |
| 權重計算方法 | adaboost | 經典AdaBoost公式 |
| 錯誤率下限 | 1e-10 | 避免除零錯誤 |
| 錯誤率上限 | 0.999 | 防止權重爆炸 |
| 樣本調整係數 | sqrt(ratio) | 平方根調整機制 |
| 批次大小 | 128 | 記憶體優化 |
| 集成策略 | weighted_voting | 預設加權投票 |
| 圖像尺寸 | 224×224 | 標準輸入格式 |
| 正規化參數 | ImageNet標準 | mean=[0.485,0.456,0.406] |

## 輸出文件結構

```
AdaBoost_Results/
├── AdaBoost_adaboost_weighted_voting/
│   └── run_YYYYMMDD_HHMMSS/
│       ├── performance_report.json      # 詳細性能數據
│       ├── performance_summary.txt      # 文字摘要報告
│       ├── overall_performance.png      # 整體性能比較圖
│       ├── class_wise_performance.png   # 類別級別分析圖
│       ├── model_weights.png           # 模型權重分布圖
│       └── confusion_matrix.png        # 混淆矩陣熱力圖
```

## AdaBoost vs Stacking 比較

| 特徵比較 | AdaBoost | Stacking |
|---------|----------|----------|
| 核心理念 | 錯誤率導向權重分配 | 元模型學習最佳組合 |
| 權重計算 | 基於專業範圍錯誤率 | 可訓練權重矩陣 |
| 複雜度 | 相對簡單，公式明確 | 較複雜，需要額外訓練 |
| 適應性 | 固定權重，運行時高效 | 動態學習，更靈活 |
| 專業性 | 強調專業範圍評估 | 全局最佳化組合 |
| 計算開銷 | 低，僅權重計算 | 中等，需元模型訓練 |
``` 
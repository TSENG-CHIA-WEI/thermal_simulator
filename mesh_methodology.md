# Thermal Solver 網格生成方法論 (Mesh Generation Methodology)

## 目錄

1. [概述](#1-概述)
2. [Icepak 網格核心技術](#2-icepak-網格核心技術)
3. [網格加密準則](#3-網格加密準則)
4. [先進網格生成算法](#4-先進網格生成算法)
5. [Icepak 特殊技術](#5-icepak-特殊技術)
6. [推薦網格方案](#6-推薦網格方案)
7. [網格生成流程](#7-網格生成流程)
8. [實現規範](#8-實現規範)
9. [參考文獻](#9-參考文獻)

---

## 1. 概述

### 1.1 網格生成的挑戰

在熱流體模擬中，網格質量直接決定了模擬結果的精度和計算效率。對於電子散熱應用（如 Icepak），網格生成面臨以下挑戰：

| 挑戰 | 描述 | 影響 |
|-----|------|-----|
| 幾何複雜性 | 鰭片、熱源、散熱器的複雜幾何形狀 | 需要靈活的網格類型 |
| 多尺度特徵 | 從微米級界面到厘米級晶片 | 需要局部加密 |
| 熱邊界層 | 溫度梯度在固體-流體界面最大 | 需要細化網格 |
| 計算效率 | 工程師需要快速迭代 | 需要平衡精度與速度 |

### 1.2 網格生成目標

```
┌─────────────────────────────────────────────────────────────────┐
│                        網格生成目標                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   精度 (Accuracy)     ═══════════════════════════              │
│                        ↑                                        │
│                        │                                        │
│                        │         ╔═══════════════════╗         │
│                        │         ║   最佳平衡點      ║         │
│                        │         ╚═══════════════════╝         │
│                        │                                        │
│                        │                                        │
│                        └───────────────────────────────→        │
│                                  計算成本                         │
│                                                                  │
│   原則：                                                         │
│   - 在需要精度的地方加密 (熱源、鰭片、邊界層)                   │
│   - 在遠場使用粗網格 (節省計算資源)                             │
│   - 保持網格漸變 (避免數值振盪)                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 本文件範圍

本文檔描述適用於 3D 瞬態熱求解器的網格生成方法論，涵蓋：

- Icepak 採用的核心網格技術
- 自適應網格加密 (AMR) 策略
- 多層級網格管理
- 幾何與物理驅動的網格生成
- 網格質量評估標準
- 推薦的實現方案

---

## 2. Icepak 網格核心技術

### 2.1 自適應網格加密 (Adaptive Mesh Refinement, AMR)

自適應網格加密是 Icepak 最重要的核心技術之一。它根據求解結果動態調整網格密度，在保持計算效率的同時提高關鍵區域的精度。

```
┌─────────────────────────────────────────────────────────────────┐
│                    Icepak AMR 流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Step 1: 初始粗網格                                             │
│   ┌───┬───┬───┬───┬───┬───┬───┬───┐                            │
│   │   │   │   │   │   │   │   │   │                            │
│   ├───┼───┼───┼───┼───┼───┼───┼───┤                            │
│   │   │   │   │   │   │   │   │   │                            │
│   ├───┼───┼───┼───┼───┼───┼───┼───┤                            │
│   │   │   │   │   │   │   │   │   │                            │
│   ├───┼───┼───┼───┼───┼───┼───┼───┤                            │
│   │   │   │   │   │   │   │   │   │                            │
│   ├───┼───┼───┼───┼───┼───┼───┼───┤                            │
│   │   │   │   │   │   │   │   │   │                            │
│   └───┴───┴───┴───┴───┴───┴───┴───┘                            │
│                                                                  │
│   Step 2: 首次求解 + 誤差評估                                    │
│   → 計算溫度梯度、功率密度、熱流方向                            │
│                                                                  │
│   Step 3: 根據準則加密                                           │
│   ┌───┬─────┬───┬─────┬───┬─────┬───┬─────┐                   │
│   │   │█████│   │█████│   │█████│   │█████│  ← 熱點區域加密 │
│   ├───┼█████┼───┼█████┼───┼█████┼───┼█████┼                   │
│   │   │█████│   │█████│   │█████│   │█████│                   │
│   ├───┼─────┼───┼─────┼───┼─────┼───┼─────┤                   │
│   │   │     │   │     │   │     │   │     │                   │
│   ├───┼─────┼───┼─────┼───┼─────┼───┼─────┤                   │
│   │   │     │   │     │   │     │   │     │                   │
│   └───┴─────┴───┴─────┴───┴─────┴───┴─────┘                   │
│                                                                  │
│   Step 4: 迭代直到收斂                                           │
│   → 重複求解 → 誤差評估 → 局部加密                               │
│   → 直到最大加密層數或誤差閾值達到                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 多層級網格策略 (Multi-Level Hierarchy)

Icepak 採用多層級網格策略，在不同精度級別之間進行嵌套求解，提高收斂速度。

```
┌─────────────────────────────────────────────────────────────────┐
│                    多層級網格策略                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Level 0 (粗網格):  16×16×4    →  快速預測全局溫度場          │
│                                                                  │
│   Level 1 (中網格):  32×32×8    →  細化熱流路徑                │
│                                                                  │
│   Level 2 (細網格):  64×64×16   →  精確邊界層                  │
│                                                                  │
│   Level 3 (極細):   128×128×32 →  最終高精度                  │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │    Level 0 → Level 1 → Level 2 → Level 3              │   │
│   │       │          │          │          │                │   │
│   │       │          ▼          │          ▼                │   │
│   │       │    ┌─────────┐      │    ┌─────────┐           │   │
│   │       │    │ AMG     │      │    │ 嵌套網格 │           │   │
│   │       │    │ 求解器  │      │    │ 插值    │           │   │
│   │       │    └─────────┘      │    └─────────┘           │   │
│   │       │                     │                           │   │
│   │       └─────────────────────┘                           │   │
│   │                                                          │   │
│   │   V-cycle (多重網格方法):                               │   │
│   │   ┌─────────┐                                          │   │
│   │   │ Level 3 │  限制 → 求解 → 延伸                      │   │
│   │   └────┬────┘                                          │   │
│   │        │ 限制                                           │   │
│   │   ┌────┴────┐                                          │   │
│   │   │ Level 2 │  限制 → 求解 → 延伸                      │   │
│   │   └────┬────┘                                          │   │
│   │        │ 限制                                           │   │
│   │   ┌────┴────┐                                          │   │
│   │   │ Level 1 │  限制 → 求解 → 延伸                      │   │
│   │   └────┬────┘                                          │   │
│   │        │ 限制                                           │   │
│   │   ┌────┴────┐                                          │   │
│   │   │ Level 0 │  求解 (粗網格，快速收斂)                  │   │
│   │   └─────────┘                                          │   │
│   │                                                          │   │
│   └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│   優勢：                                                          │
│   - 計算複雜度從 O(N³) 降至 O(N² log N)                         │
│   - 消除低頻誤差                                                   │
│   - 加速收斂 5-10 倍                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 混合網格類型

Icepak 支援多種網格類型的混合使用，以平衡精度與效率。

| 網格類型 | 描述 | 優點 | 缺點 | 適用場景 |
|--------|------|-----|------|---------|
| **結構化 (Structured)** | 規則網格，指數索引 | 記憶體效率高、計算快 | 幾何靈活性差 | 主體區域 |
| **非結構化 (Unstructured)** | 任意連接，拓撲靈活 | 幾何適應性強 | 記憶體大、計算慢 | 複雜幾何 |
| **笛卡爾切割 (Cut-Cell)** | 結構化網格 + 切割 | 保持結構化效率 | 邊界處理複雜 | 鰭片陣列 |
| **八叉樹 (Octree)** | 自適應樹狀結構 | 自然局部加密 | 數據結構複雜 | 自適應求解 |
| **邊界層 (Prismatic)** | 垂直邊界的棱柱網格 | 精確捕捉邊界層 | 需要搭配其他網格 | 鰭片/壁面 |

```
┌─────────────────────────────────────────────────────────────────┐
│                    混合網格示意圖                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   ═══════╦══════╦══════╦══════╦══════╦══════          │   │
│   │   ═╤═════╬══════╬══════╬══════╬══════╬═════╤═         │   │
│   │   ═╧═════╩══════╬══════╬══════╬══════╩═════╧═         │   │
│   │   ════════════════════════════════════════════════      │   │
│   │                                                          │   │
│   │   ════════════════════════════════════════════════      │   │
│   │   ════════════════════════════════════════════════      │   │
│   │   ════════════════════════════════════════════════      │   │
│   │                                                          │   │
│   │   ════════════════════════════════════════════════      │   │
│   │   ════════════════════════════════════════════════      │   │
│   │   ════════════════════════════════════════════════      │   │
│   │                                                          │   │
│   │   ┌───────────┐  ┌───────────┐  ┌───────────┐        │   │
│   │   │ ┌───────┐ │  │ ┌───────┐ │  │ ┌───────┐ │        │   │
│   │   │ │Prism │ │  │ │Prism │ │  │ │Prism │ │        │   │
│   │   │ └───────┘ │  │ └───────┘ │  │ └───────┘ │        │   │ ← 邊界層
│   │   │           │  │           │  │           │        │   │
│   │   │  Hex    │ │  │  Hex    │ │  │  Hex    │ │        │   │ ← 結構化
│   │   │           │  │           │  │           │        │   │
│   │   └───────────┘  └───────────┘  └───────────┘        │   │
│   │                                                          │   │
│   │   ════════════════════════════════════════════════      │   │
│   │   ════════════════════════════════════════════════      │   │
│   │                                                          │   │
│   └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│   圖例：                                                          │
│   ═══ : 鰭片 (Hex + Prism 邊界層)                               │
│   ═══ : 基底 (結構化 Hex)                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 網格加密準則

### 3.1 幾何自適應 (Geometry-Based)

幾何自適應根據模型的几何特徵自動調整網格密度。

```python
# Icepak 網格尺寸計算公式
class GeometryBasedRefinement:
    """
    根據幾何特徵計算網格尺寸
    """
    
    def __init__(self, geometry):
        self.geometry = geometry
        
    def calculate_cell_size(self, location):
        """
        計算指定位置的網格尺寸
        
        Parameters:
        -----------
        location : Point
            查詢點位置
            
        Returns:
        --------
        h : float
            建議的網格尺寸
        """
        h_min = self.geometry.min_feature_size      # 最小特徵尺寸
        h_max = self.geometry.max_feature_size       # 最大特徵尺寸
        
        # 1. 基於曲率半徑
        curvature = self.get_local_curvature(location)
        h_curvature = min(h_max, max(h_min, 
                           curvature * self.refinement_factor))
        
        # 2. 基於最小特徵
        local_feature = self.get_nearest_feature(location)
        h_feature = max(h_min, local_feature.size * 0.5)
        
        # 3. 漸變控制
        h_neighbor = self.get_neighbor_size(location)
        h_growth = min(h_curvature, h_neighbor * self.max_growth_rate)
        
        return min(h_curvature, h_feature, h_growth)
    
    def get_local_curvature(self, point):
        """
        獲取局部曲率
        
        曲率越大（幾何越彎曲），需要的網格越細
        """
        # 對於曲面，曲率 κ = 1/R
        radius = self.geometry.get_curvature_radius(point)
        return radius
```

| 幾何準則 | 描述 | 加密強度 | 適用場景 |
|---------|------|---------|---------|
| 曲率半徑 | 曲面局部曲率 | 高 | 彎曲表面 |
| 特徵長度 | 最小几何特徵 | 中 | 鰭片、狹窄通道 |
| 寬高比 | 局部網格寬高比 | 中 | 一般幾何 |
| 漸變率 | 相鄰網格尺寸變化率 | 低 | 網格平滑 |

### 3.2 物理自適應 (Physics-Based)

物理自適應根據求解結果（溫度梯度、熱流等）動態加密網格。

```
┌─────────────────────────────────────────────────────────────────┐
│                    物理驅動網格加密                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   熱流方向加密：                                                   │
│   ┌─────────────────────────────────────────┐                   │
│   │    ═══════╗  熱流 ║                  │                   │
│   │    ╔══════╝       ║                  │                   │
│   │    ╚══════╗       ║ ← 高梯度區加密     │                   │
│   │           ║       ║                  │                   │
│   │           ╚═══════╝                  │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                  │
│   加密評估準則：                                                   │
│                                                                  │
│   Γ = |∇T| / |∇T|_max                                           │
│                                                                  │
│   if Γ > Γ_threshold:                                            │
│       h_new = h_old / refinement_ratio                            │
│   elif Γ < Γ_low_threshold:                                       │
│       h_new = h_old * coarsening_ratio                            │
│                                                                  │
│   常見閾值：                                                       │
│   ┌───────────────────┬─────────────────────┐                   │
│   │ 參數              │ 建議範圍            │                   │
│   ├───────────────────┼─────────────────────┤                   │
│   │ 溫度梯度 Γ_max    │ 0.01 ~ 0.1         │                   │
│   │ 熱流密度 Γ_q      │ 0.05 ~ 0.2         │                   │
│   │ 加密閾值 Γ_refine │ 0.1 ~ 0.3          │                   │
│   │ 粗化閾值 Γ_coarse │ 0.01 ~ 0.05        │                   │
│   │ 加密比率           │ 1.5 ~ 2.0          │                   │
│   │ 粗化比率           │ 1.2 ~ 1.5          │                   │
│   └───────────────────┴─────────────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 網格質量指標

網格質量直接影響求解的精度和穩定性。Icepak 建議以下質量標準：

```python
class MeshQuality:
    """
    網格質量評估類
    參考 Icepak 建議的質量標準
    """
    
    # Icepak 建議的質量閾值
    AR_THRESHOLD = 20          # 長寬比 (Aspect Ratio)
    SKEW_THRESHOLD = 0.85      # Skewness (0=完美, 1=最大變形)
    ORTH_THRESHOLD = 0.2       # 正交性 (0=差, 1=好)
    GROWTH_THRESHOLD = 1.5     # 生長率
    
    def __init__(self, mesh):
        self.mesh = mesh
        
    def aspect_ratio(self, cell):
        """
        長寬比 (Aspect Ratio)
        
        AR = 最大邊長 / 最小邊長
        
        理想值: 1 (正方形/立方體)
        可接受: < 20
        警告: > 3
        """
        edges = cell.get_edge_lengths()
        return max(edges) / min(edges)
    
    def skewness(self, cell):
        """
        Skewness (變形度)
        
        表示網格偏離理想形狀的程度
        
        理想值: 0
        可接受: < 0.85
        警告: > 0.9
        """
        # 對於四面體
        ideal_volume = cell.get_ideal_volume()
        actual_volume = cell.volume
        
        # Skewness = 1 - (actual / ideal) for degenerate cells
        if ideal_volume > 0:
            return max(0, 1 - actual_volume / ideal_volume)
        return 1.0
    
    def orthogonality(self, cell):
        """
        正交性 (Cell Orthogonality)
        
        面法向量與連接向量間的角度
        
        理想值: 1 (90°)
        可接受: > 0.2
        """
        face_normals = cell.get_face_normals()
        centroid_vectors = cell.get_centroid_vectors()
        
        cos_theta = [np.dot(fn, cv) / (np.linalg.norm(fn) * np.linalg.norm(cv))
                    for fn, cv in zip(face_normals, centroid_vectors)]
        
        # 返回最小餘弦值（最差正交性）
        return min(abs(c) for c in cos_theta)
    
    def growth_rate(self, cell, neighbor):
        """
        相鄰網格生長率
        
        防止網格尺寸突變
        """
        return neighbor.h / cell.h
    
    def check_quality(self):
        """
        綜合質量檢查
        
        Returns:
        --------
        report : dict
            質量報告，包含所有違反閾值的情況
        """
        violations = {
            'high_ar': [],    # 長寬比過高
            'high_skew': [],  # Skewness 過高
            'low_orth': [],   # 正交性不足
            'high_growth': [] # 生長率過高
        }
        
        for cell in self.mesh.cells:
            # 檢查長寬比
            ar = self.aspect_ratio(cell)
            if ar > self.AR_THRESHOLD:
                violations['high_ar'].append({
                    'cell_id': cell.id,
                    'value': ar
                })
            
            # 檢查 skewness
            skew = self.skewness(cell)
            if skew > self.SKEW_THRESHOLD:
                violations['high_skew'].append({
                    'cell_id': cell.id,
                    'value': skew
                })
            
            # 檢查正交性
            orth = self.orthogonality(cell)
            if orth < self.ORTH_THRESHOLD:
                violations['low_orth'].append({
                    'cell_id': cell.id,
                    'value': orth
                })
        
        return violations
    
    def quality_summary(self):
        """
        生成質量摘要
        """
        violations = self.check_quality()
        
        total_cells = len(self.mesh.cells)
        
        return {
            'total_cells': total_cells,
            'high_ar_count': len(violations['high_ar']),
            'high_skew_count': len(violations['high_skew']),
            'low_orth_count': len(violations['low_orth']),
            'overall_quality': 1.0 - (
                len(violations['high_ar']) +
                len(violations['high_skew']) +
                len(violations['low_orth'])
            ) / (3 * total_cells)
        }
```

### 3.4 質量標準摘要

| 指標 | 公式 | 理想值 | Icepak 建議 | 警告閾值 |
|-----|------|-------|------------|---------|
| 長寬比 (AR) | max(edges) / min(edges) | 1 | < 20 | > 20 |
| Skewness | 1 - V_actual / V_ideal | 0 | < 0.85 | > 0.9 |
| 正交性 | min(cos θ) | 1 | > 0.2 | < 0.1 |
| 生長率 | h₂ / h₁ | 1 | < 1.5 | > 2.0 |

---

## 4. 先進網格生成算法

### 4.1 八叉樹網格 (Octree / Quadtree)

八叉樹是一種自適應空間分割數據結構，特別適合局部加密。

```
┌─────────────────────────────────────────────────────────────────┐
│                    Octree 網格結構                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                        Level 0                                   │
│                    ┌───────┐                                     │
│                    │       │                                    │
│                    │   0   │                                    │
│                    │       │                                    │
│                    └───┬───┘                                     │
│                        │                                         │
│            ┌───────────┼───────────┐                            │
│            ▼           ▼           ▼                            │
│        ┌───────┐ ┌───────┐ ┌───────┐                          │
│        │   0   │ │   0   │ │   0   │  Level 1                │
│        └───┬───┘ └───┬───┘ └───┬───┘                          │
│            │         │         │                                │
│            ▼         ▼         ▼                                │
│        ┌───────┐ ┌───────┐ ┌───────┐                          │
│        │   0   │ │ 加密   │ │   0   │  Level 2                │
│        │       │ │ 區域   │ │       │                          │
│        └───────┘ └───────┘ └───────┘                          │
│                                                                  │
│   節點編碼：                                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0  1  │  2  3  │  4  5  │  6  7                      │   │
│   │(左下前)│(右下前)│(左上前)│(右上前)                      │   │
│   │  8  9  │ 10 11  │ 12 13  │ 14 15                      │   │
│   │(左下後)│(右下後)│(左上後)│(右上後)                      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   優勢：                                                          │
│   - 局部加密靈活（只加密需要的區域）                              │
│   - 記憶體效率高（未加密區域保持粗網格）                         │
│   - 適合自適應求解（根據誤差動態調整）                          │
│   - 實現相對簡單                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
class OctreeMeshGenerator:
    """
    八叉樹網格生成器
    
    實現基於八叉樹的自適應網格生成
    """
    
    def __init__(self, domain, max_depth=6):
        """
        Parameters:
        -----------
        domain : Box
            計算域邊界
        max_depth : int
            最大加密深度
        """
        self.domain = domain
        self.max_depth = max_depth
        self.root = OctreeNode(domain, depth=0)
        
    def refine_based_on_error(self, error_field, threshold):
        """
        根據誤差場加密
        
        Parameters:
        -----------
        error_field : callable
            返回每個位置的誤差函數
        threshold : float
            加密閾值
        """
        def recursive_refine(node):
            if node.depth >= self.max_depth:
                return
                
            # 計算節點中心的誤差
            error = error_field(node.center)
            
            if error > threshold:
                node.refine()  # 分割節點
                for child in node.children:
                    recursive_refine(child)
                    
        recursive_refine(self.root)
        
    def refine_based_on_geometry(self, geometry, feature_threshold):
        """
        根據幾何特徵加密
        """
        def recursive_refine(node):
            if node.depth >= self.max_depth:
                return
                
            # 計算節點與幾何的相交
            if geometry.intersects(node.bounds):
                distance = geometry.min_distance_to_surface(node.center)
                
                if distance < feature_threshold * (2 ** (self.max_depth - node.depth)):
                    node.refine()
                    for child in node.children:
                        recursive_refine(child)
                        
        recursive_refine(self.root)
        
    def to_unstructured_mesh(self):
        """
        將八叉樹轉換為非結構化網格（用於 FVM 求解）
        """
        cells = []
        
        def traverse(node):
            if node.is_leaf():
                # 生成四面體或六面體單元
                cell = self.node_to_cell(node)
                cells.append(cell)
            else:
                for child in node.children:
                    traverse(child)
                    
        traverse(self.root)
        return UnstructuredMesh(cells)


class OctreeNode:
    """
    八叉樹節點
    """
    
    def __init__(self, bounds, depth):
        self.bounds = bounds      # 節點邊界
        self.depth = depth        # 當前深度
        self.children = None      # 子節點（8 個）
        self.data = None          # 存儲的數據（溫度、誤差等）
        
    def refine(self):
        """分割節點為 8 個子節點"""
        if self.children is not None:
            return
            
        # 計算子節點邊界
        min_corner = self.bounds.min
        max_corner = self.bounds.max
        center = (min_corner + max_corner) / 2
        
        # 創建 8 個子節點
        self.children = []
        for i in range(8):
            child_bounds = self._get_child_bounds(i, min_corner, max_corner, center)
            child = OctreeNode(child_bounds, self.depth + 1)
            self.children.append(child)
    
    def _get_child_bounds(self, index, min_c, max_c, center):
        """計算第 index 個子節點的邊界"""
        # 根據索引確定子節點的位置
        x_sign = 1 if index & 1 else -1
        y_sign = 1 if index & 2 else -1
        z_sign = 1 if index & 4 else -1
        
        child_min = center.copy()
        child_max = center.copy()
        
        if x_sign > 0:
            child_max[0] = max_c[0]
        else:
            child_min[0] = min_c[0]
            
        if y_sign > 0:
            child_max[1] = max_c[1]
        else:
            child_min[1] = min_c[1]
            
        if z_sign > 0:
            child_max[2] = max_c[2]
        else:
            child_min[2] = min_c[2]
            
        return Box(child_min, child_max)
```

### 4.2 笛卡爾切割法 (Cartesian Cut-Cell)

笛卡爾切割法保持結構化網格的高效性，同時能處理複雜幾何。

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cut-Cell 方法                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   原始笛卡爾網格：          切割後：                              │
│                                                                  │
│   ┌───┬───┬───┬───┐       ┌───┬───┬───┬───┐                   │
│   │   │   │   │   │       │   │   │   │ ╲│                    │
│   ├───┼───┼───┼───┤       ├───┼───┼───┤ ╱ │ ← 切割邊          │
│   │   │ ╲ │   │   │       │   │╲  │   │   │                   │
│   ├───┼───╋───┼───┤       ├───┼───╋───┼───┤                   │
│   │   │ ╱ │   │   │       │   │ ╲ │   │   │                   │
│   ├───┼───┼───┼───┤       ├───┼───┼───┼───┤                   │
│   │   │   │   │   │       │   │   │   │   │                   │
│   └───┴───┴───┴───┘       └───┴───┴───┴───┘                   │
│                                                                  │
│   切割原則：                                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  1. 識別與物體邊界相交的網格                              │   │
│   │     - 網格與幾何求交                                      │   │
│   │     - 計算相交體積/面積                                   │   │
│   │                                                          │   │
│   │  2. 計算切割面                                           │   │
│   │     - 幾何表面方程                                        │   │
│   │     - 切割邊界的多邊形/多面體                            │   │
│   │                                                          │   │
│   │  3. 創建切割網格的離散格式                                │   │
│   │     - 計算有效面積/體積                                  │   │
│   │     - 計算切割面的法向量                                  │   │
│   │                                                          │   │
│   │  4. 保持物理守恆                                         │   │
│   │     - 切割後的通量必須守恆                                │   │
│   │     - 使用子面/子體積方法                                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
class CutCellMeshGenerator:
    """
    笛卡爾切割網格生成器
    """
    
    def __init__(self, grid_size):
        """
        Parameters:
        -----------
        grid_size : tuple (nx, ny, nz)
            結構化網格尺寸
        """
        self.nx, self.ny, self.nz = grid_size
        
    def generate(self, geometry):
        """
        生成切割網格
        
        Parameters:
        -----------
        geometry : list of SolidRegion
            幾何實體列表
            
        Returns:
        --------
        mesh : CutCellMesh
            切割後的網格
        """
        mesh = CutCellMesh(self.nx, self.ny, self.nz)
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    cell = self.create_cell(i, j, k, geometry)
                    mesh.add_cell(cell)
                    
        return mesh
    
    def create_cell(self, i, j, k, geometry):
        """
        創建單個切割網格
        """
        cell = Cell()
        cell.index = (i, j, k)
        cell.bounds = self.get_cell_bounds(i, j, k)
        
        # 計算與每個實體的相交
        for entity in geometry:
            intersection = entity.intersect(cell.bounds)
            if intersection.is_valid():
                cell.add_intersection(entity, intersection)
                
        return cell
```

### 4.3 邊界層網格 (Boundary Layer / Inflation)

邊界層網格用於精確捕捉近壁面區域的溫度和速度邊界層。

```
┌─────────────────────────────────────────────────────────────────┐
│                    邊界層網格加密                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────┐          │
│   │╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲│ ← 鰭片 (Solid)     │
│   │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│               │
│   │ ╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲│               │
│   │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│               │
│   │╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲│               │
│   ├─────────────────────────────────────────────────┤          │
│   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ ← 基底 (Substrate)  │
│   └─────────────────────────────────────────────────┘          │
│                                                                  │
│   邊界層參數：                                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 參數                    │ 公式/建議                       │   │
│   ├─────────────────────────────────────────────────────────┤   │
│   │ 第一層厚度 h₁            │ h₁ = h_base / growth_ratio     │   │
│   │ 增長率                  │ 1.1 ~ 1.3 (推薦 1.15)           │   │
│   │ 總層數                  │ 5 ~ 20 (推薦 10)                │   │
│   │ 增長類型                │ geometric / arithmetic           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   y+ 控制（流體模擬）：                                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ y+ = (u_τ · Δy) / ν                                    │   │
│   │                                                          │   │
│   │ 其中：                                                    │   │
│   │ - u_τ = 摩擦速度 = √(τ_w / ρ)                           │   │
│   │ - Δy = 第一層網格厚度                                   │   │
│   │ - ν = 運動粘度                                          │   │
│   │                                                          │   │
│   │ 熱模擬建議：                                               │   │
│   │ - 第一層網格足夠細以捕捉溫度邊界層                        │   │
│   │ - Δy ≈ δ_t / 10 (δ_t = 熱邊界層厚度)                   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
class BoundaryLayerGenerator:
    """
    邊界層網格生成器
    """
    
    def __init__(self, n_layers, growth_ratio, first_layer_height=None):
        """
        Parameters:
        -----------
        n_layers : int
            邊界層總層數
        growth_ratio : float
            網格增長率 (推薦 1.1 ~ 1.3)
        first_layer_height : float, optional
            第一層厚度，如果未指定則自動計算
        """
        self.n_layers = n_layers
        self.growth_ratio = growth_ratio
        self.first_layer_height = first_layer_height
        
    def calculate_layer_heights(self, total_thickness):
        """
        計算各層厚度
        
        Parameters:
        -----------
        total_thickness : float
            邊界層總厚度
            
        Returns:
        --------
        heights : list
            各層厚度列表
        """
        if self.first_layer_height is None:
            # 根據總厚度和增長率計算第一層
            # 幾何級數求和: h₁ * (1 - rⁿ) / (1 - r) = total
            # => h₁ = total * (1 - r) / (1 - rⁿ)
            r = self.growth_ratio
            n = self.n_layers
            self.first_layer_height = total_thickness * (1 - r) / (1 - r ** n)
            
        heights = []
        for i in range(self.n_layers):
            h = self.first_layer_height * (self.growth_ratio ** i)
            heights.append(h)
            
        return heights
    
    def generate(self, surface_mesh, total_thickness):
        """
        生成邊界層網格
        
        Parameters:
        -----------
        surface_mesh : SurfaceMesh
            壁面網格
        total_thickness : float
            邊界層總厚度
            
        Returns:
        --------
        prism_layer : PrismMesh
            棱柱邊界層網格
        """
        layer_heights = self.calculate_layer_heights(total_thickness)
        
        prism_layer = []
        
        for layer_idx, h in enumerate(layer_heights):
            # 創建該層的網格
            layer_mesh = self.create_layer(surface_mesh, h, layer_idx)
            prism_layer.append(layer_mesh)
            
        return prism_layer
```

---

## 5. Icepak 特殊技術

### 5.1 智能網格建議器 (Smart Mesh Advisor)

Icepak 的智能網格建議器能自動分析幾何並建議最適合的網格策略。

```
┌─────────────────────────────────────────────────────────────────┐
│              Icepak 智能網格建議流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Step 1: 載入幾何                                                 │
│          │                                                        │
│          ▼                                                        │
│   Step 2: 自動識別特徵                                             │
│   ┌─────────────────────────────────────┐                        │
│   │  鰭片 (Fins):    8 個, 間距 1mm      │ ← 自動檢測          │
│   │  熱源 (Sources): 4 個, 位置已標記    │                      │
│   │  散熱器 (Heatsink): 1 個, 高度 10mm │                      │
│   │  邊界 (BC):      6 個面             │                      │
│   └─────────────────────────────────────┘                        │
│          │                                                        │
│          ▼                                                        │
│   Step 3: 分析網格敏感區域                                         │
│   ┌─────────────────────────────────────┐                        │
│   │  區域名        │ 敏感度 │ 網格建議   │                        │
│   ├─────────────────────────────────────┤                        │
│   │  鰭片區域      │ 高     │ 極細 0.1mm │ ← 鰭片優先加密        │
│   │  熱源附近      │ 高     │ 細 0.2mm  │ ← 熱點加密            │
│   │  基底區域      │ 中     │ 中 0.5mm  │ ← 主體                │
│   │  遠場          │ 低     │ 粗 2.0mm  │ ← 粗網格節省資源      │
│   └─────────────────────────────────────┘                        │
│          │                                                        │
│          ▼                                                        │
│   Step 4: 建議網格策略                                            │
│   ┌─────────────────────────────────────┐                        │
│   │  全域基礎網格:  1.0mm × 1.0mm × 0.5mm                      │
│   │  鰭片區域加密:  0.1mm × 0.1mm × 0.1mm                      │
│   │  邊界層:       10 層, 增長率 1.15                          │
│   │  最大網格數:   < 500,000                                    │
│   └─────────────────────────────────────┘                        │
│          │                                                        │
│          ▼                                                        │
│   Step 5: 用戶確認 / 調整                                          │
│          │                                                        │
│          ▼                                                        │
│   Step 6: 生成網格                                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
class SmartMeshAdvisor:
    """
    智能網格建議器
    
    自動分析幾何並建議最適合的網格策略
    """
    
    def __init__(self):
        self.geometry_analysis = GeometryAnalyzer()
        self.mesh_suggester = MeshSuggester()
        
    def analyze_and_suggest(self, geometry, physics_settings):
        """
        分析幾何並建議網格策略
        
        Parameters:
        -----------
        geometry : list of GeometricEntity
            幾何實體列表
        physics_settings : dict
            物理設定（熱功率、流速等）
            
        Returns:
        --------
        suggestion : MeshSuggestion
            網格建議結果
        """
        # Step 1: 分析幾何特徵
        features = self.geometry_analysis.analyze(geometry)
        
        # Step 2: 識別網格敏感區域
        sensitive_regions = self.identify_sensitive_regions(
            features, physics_settings)
        
        # Step 3: 生成網格建議
        suggestion = self.mesh_suggester.suggest(
            features, sensitive_regions)
        
        return suggestion


class GeometryAnalyzer:
    """幾何分析器"""
    
    def analyze(self, geometry):
        """分析幾何並提取特徵"""
        features = {
            'fins': [],
            'heat_sources': [],
            'heat_sinks': [],
            'interfaces': [],
            'min_feature_size': float('inf'),
            'max_feature_size': 0
        }
        
        for entity in geometry:
            if entity.type == 'fin':
                features['fins'].append(self.analyze_fin(entity))
                features['min_feature_size'] = min(
                    features['min_feature_size'],
                    entity.thickness)
                    
            elif entity.type == 'heat_source':
                features['heat_sources'].append(entity)
                
            elif entity.type == 'heat_sink':
                features['heat_sinks'].append(entity)
                
            features['max_feature_size'] = max(
                features['max_feature_size'],
                entity.max_dimension)
                
        return features
    
    def analyze_fin(self, fin):
        """分析鰭片特徵"""
        return {
            'height': fin.height,
            'thickness': fin.thickness,
            'spacing': fin.spacing,
            'count': fin.count,
            'orientation': fin.orientation,  # 'horizontal', 'vertical'
            'array_type': fin.array_type     # 'linear', 'staggered', 'parallel'
        }
```

### 5.2 自動鰭片識別

```python
class FinDetector:
    """Icepak 鰭片自動識別算法"""
    
    def __init__(self, geometry):
        self.geometry = geometry
        
    def detect_fins(self):
        """
        自動識別鰭片
        
        Returns:
        --------
        fins : list of Fin
            識別出的鰭片列表
        """
        # 1. 幾何分割 - 按高度分組
        height_groups = self.geometry.segment_by_height()
        
        # 2. 高寬比分析
        candidate_fins = []
        for group in height_groups:
            for entity in group:
                aspect_ratio = entity.height / entity.thickness
                if aspect_ratio > 3:  # 典型的鰭片高寬比
                    if self.is_fin_shaped(entity):
                        candidate_fins.append(entity)
        
        # 3. 間距分析 - 識別鰭片陣列
        fins = self.group_into_fin_arrays(candidate_fins)
        
        return fins
    
    def suggest_mesh(self, fin_info):
        """
        建議鰭片區域網格
        
        Parameters:
        -----------
        fin_info : dict
            鰭片信息
            
        Returns:
        --------
        mesh_params : dict
            網格參數建議
        """
        h_fin = fin_info['thickness']  # 鰭片厚度
        h_min = h_fin / 4  # 鰭片間最小網格
        
        # Icepak 建議：鰭片區域至少 8 層
        n_layers = 8
        growth = 1.15
        
        return {
            'base_size': h_min,
            'layers': n_layers,
            'growth_ratio': growth,
            'first_layer_height': h_min / (growth ** (n_layers - 1)),
            'type': 'prismatic',  # 邊界層網格
            'growth_type': 'geometric'
        }
```

### 5.3 多區域網格匹配

```
┌─────────────────────────────────────────────────────────────────┐
│                    多區域網格連續性                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┬─────────┬─────────┬─────────┐                    │
│   │  Zone A │  Zone B │  Zone C │  Zone D │                   │
│   │ (鰭片)  │  (基底) │  (TIM)  │ (散熱蓋)│                   │
│   ├─────────┼─────────┼─────────┼─────────┤                    │
│   │ h=0.1mm │ h=1.0mm │ h=0.05mm│ h=0.5mm │  ← 不同區域網格   │
│   └────┬────┴────┬────┴────┬────┴────┬────┘                    │
│        │         │         │         │                          │
│        └────┬────┴────┬────┴────┬────┘                          │
│             │         │         │                                │
│             ▼         ▼         ▼                                │
│        ┌─────────────────────────────────┐                      │
│        │     網格匹配層 (Matching Layer)  │ ← 確保連續性         │
│        └─────────────────────────────────┘                      │
│                                                                  │
│   匹配原則：                                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  1. 面對面的兩個區域網格尺寸比 ≤ 2:1                    │   │
│   │  2. 節點位置盡量對齊                                     │   │
│   │  3. 不匹配時使用 1:2 或 1:4 漸變                        │   │
│   │  4. 界面熱阻區域需要更細的網格                           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 推薦網格方案

### 6.1 三層次混合網格策略

對於我們的 3D 熱求解器，推薦採用三層次混合網格策略：

```
┌─────────────────────────────────────────────────────────────────┐
│                    混合網格策略                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                         │   │
│   │   Level 1: 笛卡爾結構網格（高效）                        │   │
│   │   ┌───┬───┬───┬───┬───┬───┬───┬───┐                   │   │
│   │   │   │   │   │   │   │   │   │   │                   │   │
│   │   ├───┼───┼───┼───┼───┼───┼───┼───┤                   │   │
│   │   │   │   │   │   │   │   │   │   │                   │   │
│   │   ├───┼───┼───┼───┼───┼───┼───┼───┤                   │   │
│   │   │   │   │   │   │   │   │   │   │                   │   │
│   │   └───┴───┴───┴───┴───┴───┴───┴───┘                   │   │
│   │   → 用於主體區域                                        │   │
│   │                                                         │   │
│   │   Level 2: 局部加密（八叉樹）                           │   │
│   │   ┌───┬───┬───┬───┬───┬───┬───┬───┐                   │   │
│   │   │   │   │   │   │   │   │   │   │                   │   │
│   │   ├───┼───┼───┼───┼───┼───┼───┼───┤                   │   │
│   │   │   │ ██│ ██│   │   │   │   │   │ ← 熱源區域八叉樹加密│   │
│   │   ├───┼█████┼───┼───┼───┼───┼───┤                      │   │
│   │   │   │ ██│ ██│   │   │   │   │   │                   │   │
│   │   └───┴───┴───┴───┴───┴───┴───┴───┘                   │   │
│   │                                                         │   │
│   │   Level 3: 邊界層（鰭片/散熱器）                        │   │
│   │   ═══════╦══════╦══════╦══════                          │   │
│   │   ═╤═════╬══════╬══════╬═════╤═                         │   │
│   │   ═╧═════╩══════╩══════╩═════╧═                         │   │
│   │   ════════════════════════════════  ← 鰭片區域           │   │
│   │                                                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 網格生成優先順序

```
Phase 1: 基礎結構化網格 (2-3 週)
├── 步驟 1.1: 創建全域笛卡爾網格
├── 步驟 1.2: 識別材料區域
├── 步驟 1.3: 實現邊界層生成
└── 步驟 1.4: 切割鰭片區域

Phase 2: 自適應加密 (2-3 週)
├── 步驟 2.1: 實現八叉樹局部加密
├── 步驟 2.2: 基於熱梯度加密
└── 步驟 2.3: 多層次網格匹配

Phase 3: 高級功能 (2-3 週)
├── 步驟 3.1: 網格品質優化
├── 步驟 3.2: 並行網格生成
└── 步驟 3.3: 與求解器整合
```

### 6.3 網格類型比較總結

| 特性 | 笛卡爾 | 八叉樹 | 非結構化 | Icepak 混合 |
|-----|-------|-------|---------|------------|
| **實現難度** | 低 | 中 | 高 | 很高 |
| **記憶體效率** | 高 | 中 | 低 | 中 |
| **幾何靈活性** | 低 | 中 | 高 | 高 |
| **物理捕捉** | 中 | 好 | 很好 | 極好 |
| **適合場景** | IC 封裝 | 一般複雜 | 任意幾何 | 電子散熱 |
| **鰭片處理** | 切割法 | 自然支援 | 困難 | 最佳 |
| **計算效率** | 極好 | 好 | 中 | 好 |

### 6.4 推薦方案詳細比較

| 方案 | 描述 | 優點 | 缺點 | 適合項目 |
|-----|------|-----|------|---------|
| **方案 A** | 笛卡爾 + 切割法 | 簡單高效 | 鰭片處理複雜 | IC 封裝 |
| **方案 B** | 八叉樹 | 靈活優雅 | 數據結構複雜 | 一般熱分析 |
| **方案 C** | 非結構化 | 最通用 | 計算量大 | 複雜幾何 |
| **方案 D** | Icepak 混合 | 最強大 | 開發量大 | 生產級 |

**推薦：方案 B（八叉樹）**

八叉樹方案提供了良好的平衡：
- 足夠的幾何靈活性
- 自然支援局部加密
- 實現複雜度適中
- 便於與自適應求解整合

---

## 7. 網格生成流程

### 7.1 完整流程圖

```
┌─────────────────────────────────────────────────────────────────┐
│                  網格生成流程                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌───────────────┐                                              │
│   │ 1. 載入幾何    │                                              │
│   │   - STL/STEP  │                                              │
│   │   - 參數化模型 │                                              │
│   └───────┬───────┘                                              │
│           │                                                      │
│           ▼                                                      │
│   ┌───────────────────────────┐                                  │
│   │ 2. 識別特徵              │ ← 鰭片、熱源、邊界               │
│   │   - Fin Detector         │                                  │
│   │   - Source Locator       │                                  │
│   │   - Boundary Identifier  │                                  │
│   └───────────┬───────────────┘                                  │
│               │                                                    │
│               ▼                                                      │
│   ┌───────────────────────────────────┐                          │
│   │ 3. 定義網格控制區域 (Control Zones)│                          │
│   │   ┌─────────────────────────┐    │                          │
│   │   │ Zone A: 全體 (粗)       │    │                          │
│   │   │ Zone B: 熱源 (加密)     │    │                          │
│   │   │ Zone C: 鰭片 (極細)     │    │                          │
│   │   │ Zone D: 邊界層 (Prism)  │    │                          │
│   │   └─────────────────────────┘    │                          │
│   └───────────┬───────────────────────┘                          │
│               │                                                    │
│               ▼                                                    │
│   ┌───────────────────────────────────┐                          │
│   │ 4. 選擇網格類型                   │                          │
│   │   ├── 主體: 笛卡爾 (高效)        │                          │
│   │   ├── 鰭片: 邊界層 (Prismatic)  │                          │
│   │   └── 複雜: 八叉樹 (局部)        │                          │
│   └───────────┬───────────────────────┘                          │
│               │                                                    │
│               ▼                                                    │
│   ┌───────────────────────────────────┐                          │
│   │ 5. 估計網格尺寸                   │ ← 基於物理準則           │
│   │                                    │   (熱梯度、幾何特徵)    │
│   └───────────┬───────────────────────┘                          │
│               │                                                    │
│               ▼                                                    │
│   ┌───────────────────────────────────┐                          │
│   │ 6. 生成初始網格                  │                          │
│   │   ├── 八叉樹分割                  │                          │
│   │   ├── 邊界層生成                  │                          │
│   │   └── 區域匹配                   │                          │
│   └───────────┬───────────────────────┘                          │
│               │                                                    │
│               ▼                                                    │
│   ┌───────────────────────────────────┐                          │
│   │ 7. 網格質量檢查                  │                          │
│   │   ├── AR < 20                     │                          │
│   │   ├── Skewness < 0.85            │ ← Icepak 標準          │
│   │   └── Orthogonality > 0.2         │                          │
│   └───────────┬───────────────────────┘                          │
│               │                                                    │
│       ┌───────┴───────┐                                           │
│       ▼               ▼                                           │
│   ┌─────────┐   ┌─────────────┐                                   │
│   │ 通過 ✓  │   │ 失敗 ✗      │                                   │
│   └────┬────┘   └──────┬──────┘                                   │
│        │               ▼                                          │
│        │         ┌─────────────────┐                              │
│        │         │ 改善網格質量     │ ← 加密/平滑化               │
│        │         └────────┬────────┘                              │
│        │                  │                                       │
│        └──────────────────┘                                      │
│                           │                                       │
│                           ▼                                       │
│                   ┌─────────────────┐                             │
│                   │ 8. 輸出網格      │ → FVM 求解器               │
│                   │   - 結構化      │                            │
│                   │   - 非結構化     │                            │
│                   └─────────────────┘                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 核心類別設計

```python
"""
網格生成模組核心類別
"""


class MeshGenerator:
    """
    網格生成器主類
    """
    
    def __init__(self, config):
        """
        Parameters:
        -----------
        config : MeshConfig
            網格配置參數
        """
        self.config = config
        self.geometry = None
        self.mesh = None
        
    def load_geometry(self, geometry_file):
        """
        載入幾何模型
        """
        if geometry_file.endswith('.stl'):
            self.geometry = STLGeometry(geometry_file)
        elif geometry_file.endswith('.step'):
            self.geometry = STEPGeometry(geometry_file)
        else:
            raise ValueError(f"不支持的幾何格式: {geometry_file}")
            
    def generate(self):
        """
        生成網格
        """
        # 1. 分析幾何
        features = self.analyze_geometry()
        
        # 2. 定義控制區域
        zones = self.define_control_zones(features)
        
        # 3. 生成網格
        mesh = self._generate_mesh(zones)
        
        # 4. 質量檢查
        quality = self.check_quality(mesh)
        
        if not quality.is_acceptable():
            mesh = self.improve_quality(mesh, quality)
            
        self.mesh = mesh
        return mesh
        
    def analyze_geometry(self):
        """分析幾何特徵"""
        analyzer = GeometryAnalyzer()
        return analyzer.analyze(self.geometry)


class MeshConfig:
    """
    網格配置類
    """
    
    def __init__(self):
        # 基礎網格尺寸
        self.base_size = 1.0e-3  # 1mm
        
        # 加密區域
        self.refinement_zones = []
        
        # 邊界層
        self.boundary_layer = {
            'enabled': True,
            'n_layers': 10,
            'growth_ratio': 1.15,
            'first_layer_height': None
        }
        
        # 質量閾值
        self.quality_thresholds = {
            'max_aspect_ratio': 20,
            'max_skewness': 0.85,
            'min_orthogonality': 0.2,
            'max_growth_rate': 1.5
        }
        
        # 最大網格數
        self.max_cells = 500000
        
        # 最大加密深度
        self.max_refinement_depth = 4


class OctreeMeshGenerator:
    """
    八叉樹網格生成器
    """
    
    def __init__(self, domain, config):
        self.domain = domain
        self.config = config
        self.root = None
        
    def generate(self, refinement_sources):
        """
        生成八叉樹網格
        
        Parameters:
        -----------
        refinement_sources : list of RefinementSource
            加密源（幾何、熱源等）
        """
        # 創建根節點
        self.root = OctreeNode(self.domain, depth=0)
        
        # 遞歸加密
        self._refine_recursive(self.root, refinement_sources)
        
        # 轉換為網格
        return self._to_mesh()
        
    def _refine_recursive(self, node, sources):
        """
        遞歸加密
        """
        if node.depth >= self.config.max_refinement_depth:
            return
            
        # 計算節點的加密需求
        refinement_level = 0
        for source in sources:
            level = source.get_refinement_level(node)
            refinement_level = max(refinement_level, level)
            
        if refinement_level > 0:
            if node.is_leaf():
                node.refine()
            for child in node.children:
                self._refine_recursive(child, sources)


class BoundaryLayerGenerator:
    """
    邊界層網格生成器
    """
    
    def __init__(self, surface_mesh, config):
        self.surface = surface_mesh
        self.config = config
        
    def generate(self):
        """
        生成邊界層網格
        """
        heights = self._calculate_layer_heights()
        layers = []
        
        for i, h in enumerate(heights):
            layer = self._create_layer(i, h)
            layers.append(layer)
            
        return layers
        
    def _calculate_layer_heights(self):
        """計算各層厚度"""
        n = self.config['n_layers']
        r = self.config['growth_ratio']
        
        if self.config['first_layer_height'] is not None:
            h1 = self.config['first_layer_height']
        else:
            # 根據總厚度計算
            total = self.config['total_thickness']
            h1 = total * (1 - r) / (1 - r ** n)
            
        return [h1 * (r ** i) for i in range(n)]


class MeshQualityChecker:
    """
    網格質量檢查器
    """
    
    def __init__(self, mesh, thresholds):
        self.mesh = mesh
        self.thresholds = thresholds
        
    def check(self):
        """
        執行質量檢查
        
        Returns:
        --------
        report : QualityReport
            質量報告
        """
        violations = {
            'high_aspect_ratio': [],
            'high_skewness': [],
            'low_orthogonality': [],
            'large_growth_rate': []
        }
        
        for cell in self.mesh.cells:
            # 檢查長寬比
            ar = self.aspect_ratio(cell)
            if ar > self.thresholds['max_aspect_ratio']:
                violations['high_aspect_ratio'].append({
                    'cell': cell.id,
                    'value': ar
                })
                
            # 檢查 skewness
            skew = self.skewness(cell)
            if skew > self.thresholds['max_skewness']:
                violations['high_skewness'].append({
                    'cell': cell.id,
                    'value': skew
                })
                
            # 檢查正交性
            orth = self.orthogonality(cell)
            if orth < self.thresholds['min_orthogonality']:
                violations['low_orthogonality'].append({
                    'cell': cell.id,
                    'value': orth
                })
                
        return QualityReport(violations)
```

---

## 8. 實現規範

### 8.1 檔案結構

```
thermal_solver_3d/
│
├── core/
│   ├── __init__.py
│   ├── geometry.py          # 3D 堆疊幾何
│   │   ├── Geometry
│   │   ├── SolidRegion
│   │   └── GeometryAnalyzer
│   │
│   ├── material.py          # 材料（含異向性）
│   │   ├── Material
│   │   ├── SolidMaterial
│   │   └── FluidMaterial
│   │
│   └── mesh.py             # 網格生成
│       ├── MeshGenerator
│       ├── MeshConfig
│       ├── OctreeMeshGenerator
│       ├── CutCellMeshGenerator
│       ├── BoundaryLayerGenerator
│       └── MeshQualityChecker
│
├── sources/
│   ├── __init__.py
│   ├── power_source.py     # 功率源基類
│   ├── point_source.py     # 點熱源
│   ├── area_source.py      # 面熱源
│   └── power_trace.py      # CSV 解析
│
├── doe/
│   ├── __init__.py
│   ├── doe_manager.py      # DOE 主類
│   ├── designs.py          # 實驗設計方法
│   └── analysis.py         # 靈敏度/響應面
│
├── io/
│   ├── __init__.py
│   ├── reader.py           # YAML/CSV 讀取
│   └── writer.py           # VTK/CSV 輸出
│
├── post/
│   ├── __init__.py
│   ├── visualization.py    # 熱圖/動畫
│   └── exporter.py         # 格式匯出
│
├── examples/
│   ├── mesh_example.py     # 網格生成範例
│   ├── fin_array.py        # 鰭片陣列範例
│   └── adaptive_mesh.py    # 自適應加密範例
│
├── tests/
│   ├── test_mesh.py        # 網格單元測試
│   ├── test_quality.py     # 質量測試
│   └── test_refinement.py  # 加密測試
│
├── config/
│   ├── mesh_defaults.yaml  # 預設網格參數
│   └── materials.yaml      # 材料資料庫
│
├── setup.py
├── requirements.txt
└── README.md
```

### 8.2 API 規範

```python
"""
熱求解器網格模組 API 規範

Example Usage:
-------------

# 1. 創建網格配置
config = MeshConfig()
config.base_size = 1e-3  # 1mm
config.boundary_layer = {
    'enabled': True,
    'n_layers': 10,
    'growth_ratio': 1.15
}

# 2. 創建網格生成器
generator = MeshGenerator(config)

# 3. 載入幾何
generator.load_geometry("heat_sink.stl")

# 4. 生成網格
mesh = generator.generate()

# 5. 檢查質量
checker = MeshQualityChecker(mesh, config.quality_thresholds)
report = checker.check()
print(report.summary())

# 6. 輸出網格
mesh.export("mesh.vtk")
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import numpy as np


class MeshConfig:
    """網格配置"""
    
    def __init__(self):
        # 基礎參數
        self.base_size: float = 1.0e-3
        self.min_size: float = 1.0e-4
        self.max_cells: int = 500000
        
        # 邊界層
        self.boundary_layer: Dict = {
            'enabled': True,
            'n_layers': 10,
            'growth_ratio': 1.15,
            'first_layer_height': None,
            'total_thickness': None
        }
        
        # 加密區域
        self.refinement_zones: List[Dict] = []
        
        # 質量閾值
        self.quality_thresholds: Dict = {
            'max_aspect_ratio': 20,
            'max_skewness': 0.85,
            'min_orthogonality': 0.2,
            'max_growth_rate': 1.5
        }
        
        # 最大加密深度
        self.max_refinement_depth: int = 4


class Mesh(ABC):
    """網格基類"""
    
    @abstractmethod
    def n_cells(self) -> int:
        """返回單元數"""
        pass
    
    @abstractmethod
    def n_nodes(self) -> int:
        """返回節點數"""
        pass
    
    @abstractmethod
    def cells(self) -> List:
        """返回單元列表"""
        pass
    
    @abstractmethod
    def nodes(self) -> np.ndarray:
        """返回節點座標"""
        pass
    
    @abstractmethod
    def export(self, filename: str):
        """導出網格"""
        pass


class StructuredMesh(Mesh):
    """結構化網格"""
    
    def __init__(self, nx: int, ny: int, nz: int):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.nodes_array = None
        self.cells = []
        
    def n_cells(self) -> int:
        return (self.nx - 1) * (self.ny - 1) * (self.nz - 1)
    
    def n_nodes(self) -> int:
        return self.nx * self.ny * self.nz


class UnstructuredMesh(Mesh):
    """非結構化網格"""
    
    def __init__(self):
        self.nodes_array = np.array([])  # N x 3
        self.cells = []  # 單元列表
        
    def n_cells(self) -> int:
        return len(self.cells)
    
    def n_nodes(self) -> int:
        return len(self.nodes_array)


class MeshGenerator(ABC):
    """網格生成器基類"""
    
    @abstractmethod
    def generate(self) -> Mesh:
        """生成網格"""
        pass
    
    @abstractmethod
    def refine(self, locations: List):
        """局部加密"""
        pass


class OctreeMeshGenerator(MeshGenerator):
    """八叉樹網格生成器"""
    
    def __init__(self, domain: Tuple[np.ndarray, np.ndarray], config: MeshConfig):
        """
        Parameters:
        -----------
        domain : (min_corner, max_corner)
            計算域邊界
        config : MeshConfig
            網格配置
        """
        self.domain = domain
        self.config = config
        self.root = None
        
    def generate(self) -> UnstructuredMesh:
        """生成八叉樹網格"""
        pass
    
    def refine(self, locations: List[np.ndarray]):
        """根據位置列表加密"""
        pass


class CutCellMeshGenerator(MeshGenerator):
    """切割網格生成器"""
    
    def __init__(self, grid_size: Tuple[int, int, int], config: MeshConfig):
        self.nx, self.ny, self.nz = grid_size
        self.config = config
        
    def generate(self) -> StructuredMesh:
        """生成切割網格"""
        pass


class MeshQualityChecker:
    """網格質量檢查器"""
    
    def __init__(self, mesh: Mesh, thresholds: Dict):
        self.mesh = mesh
        self.thresholds = thresholds
        
    def check(self) -> 'QualityReport':
        """執行質量檢查"""
        pass
    
    def aspect_ratio(self, cell) -> float:
        """計算單元長寬比"""
        pass
    
    def skewness(self, cell) -> float:
        """計算單元 skewness"""
        pass
    
    def orthogonality(self, cell) -> float:
        """計算單元正交性"""
        pass


class QualityReport:
    """質量報告"""
    
    def __init__(self, violations: Dict):
        self.violations = violations
        
    def summary(self) -> str:
        """返回摘要"""
        pass
    
    def is_acceptable(self) -> bool:
        """檢查是否可接受"""
        pass
```

---

## 10. 移動熱源敏感場景的網格策略

### 10.1 場景特點分析

當熱源位置移動且對微小溫差敏感時，傳統的靜態網格策略可能不足。讓我們分析這個特殊場景的需求：

| 特性 | 傳統場景 | 移動熱源敏感場景 |
|-----|---------|-----------------|
| 熱源位置 | 固定 | 多個、可移動 |
| 溫差敏感度 | 一般 | 極高 (<< 1°C) |
| 網格需求 | 靜態 | 需適應熱源位置 |
| 被動層 | 正常網格 | 可大幅簡化 |
| 主動層 | 普通加密 | 極細網格 + 動態調整 |

### 10.2 雙層網格差異化策略

```
┌─────────────────────────────────────────────────────────────────┐
│                    雙層網格差異化策略                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                        3D 堆疊結構                       │   │
│   │                                                          │   │
│   │   ╔═══════════════════════════════════════════════╗   │   │
│   │   ║  主動層 (Active Layer)                         ║   │   │
│   │   ║  ════════════════════════════════════════      ║   │   │
│   │   ║  🔥 🔥 🔥 🔥 🔥 🔥 🔥 🔥 🔥 🔥              ║   │   │
│   │   ║  🔥 🔥 🔥 🔥 🔥 🔥 🔥 🔥 🔥 🔥              ║   │   │
│   │   ║  🔥 🔥 🔥 🔥 🔥 🔥 🔥 🔥 🔥 🔥              ║   │   │
│   │   ║  🔥 🔥 🔥 🔥 🔥 🔥 🔥 🔥 🔥 🔥              ║   │   │
│   │   ║  網格: 10-50 µm (極細)                         ║   │   │
│   │   ║  特點: 每個熱源上方超細加密                      ║   │   │
│   │   ╚═══════════════════════════════════════════════╝   │   │
│   │                        │                                 │
│   │                        │ 熱流向下                         │
│   │   ╔═══════════════════════════════════════════════╗   │   │
│   │   ║  被動層 1 (Passive Layer 1)                   ║   │   │
│   │   ║  ═══════════════════════════════              ║   │   │
│   │   ║                                                 ║   │   │
│   │   ║  網格: 100-500 µm (粗)                        ║   │   │
│   │   ║  特點: 統一粗網格，快速傳遞                    ║   │   │
│   │   ╚═══════════════════════════════════════════════╝   │   │
│   │                        │                                 │
│   │                        │                                 │
│   │   ╔═══════════════════════════════════════════════╗   │   │
│   │   ║  被動層 2 (Passive Layer 2)                   ║   │   │
│   │   ║  ═══════════════════════════════              ║   │   │
│   │   ║                                                 ║   │   │
│   │   ║  網格: 200-1000 µm (極粗)                     ║   │   │
│   │   ║  特點: 極簡化，只需捕捉平均溫度                  ║   │   │
│   │   ╚═══════════════════════════════════════════════╝   │   │
│   │                                                          │   │
│   └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│   原則：                                                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  主動層：極細網格 (10-50 µm)，每個熱源獨立加密           │   │
│   │  被動層：粗網格 (100-1000 µm)，統一處理                   │   │
│   │  界面：熱阻網格，捕捉溫度跳變                            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 10.3 移動熱源的全域加密策略

由於熱源會移動，我們需要在熱源可能出現的所有區域都進行加密：

```
┌─────────────────────────────────────────────────────────────────┐
│                    移動熱源的全域加密策略                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   熱源移動範圍 (Heat Source Movement Envelope):                  │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   ┌─────────────────────────────────────────────────┐  │  │
│   │   │                                                 │  │  │
│   │   │   ┌─────────────────────────────────────────┐   │  │  │
│   │   │   │                                         │   │  │  │
│   │   │   │         ┌─────────────────────┐         │   │  │  │
│   │   │   │         │  熱源最大移動範圍    │         │   │  │  │
│   │   │   │         │  (Movement Envelope)│         │   │  │  │
│   │   │   │         │  ════════════════   │         │   │  │  │
│   │   │   │         │  🔥 → 🔥 → 🔥       │         │   │  │  │
│   │   │   │         │  🔥    🔥           │         │   │  │  │
│   │   │   │         │         🔥 → 🔥     │         │   │  │  │
│   │   │   │         └─────────────────────┘         │   │  │  │
│   │   │   │                                         │   │  │  │
│   │   │   │   熱源只會在這個信封內移動               │   │  │  │
│   │   │   │   對整個信封區域進行加密                 │   │  │  │
│   │   │   └─────────────────────────────────────────┘   │  │  │
│   │   │                                                 │  │  │
│   │   │   ┌─────────────────────────────────────────┐   │  │  │
│   │   │   │                                         │   │  │  │
│   │   │   │   加密區域                               │   │  │  │
│   │   │   │   ════════════════════════════════     │   │  │  │
│   │   │   │   ║███████████████║                   │   │  │  │
│   │   │   │   ║███████████████║                   │   │  │  │
│   │   │   │   ║███████████████║                   │   │  │  │
│   │   │   │   ║███████████████║                   │   │  │  │
│   │   │   │   ╚═════════════════════════════════╝   │  │  │
│   │   │   │                                         │   │  │
│   │   │   │   網格尺寸:                             │  │  │
│   │   │   │   - 熱源區域: 10-20 µm                   │  │  │
│   │   │   │   - 熱源附近: 30-50 µm                  │  │  │
│   │   │   │   - 熱源信封: 50-100 µm                 │  │  │
│   │   │   └─────────────────────────────────────────┘   │  │
│   │   │                                                 │  │
│   │   └─────────────────────────────────────────────────┘  │
│   │                                                          │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│   關鍵步驟：                                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  1. 識別所有熱源的運動軌跡                                │   │
│   │  2. 計算熱源的最大運動範圍 (Envelope)                    │   │
│   │  3. 對整個 Envelope 區域進行加密                        │   │
│   │  4. 保持 Envelope 邊界外的網格粗獷                      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 10.4 熱源敏感區域的網格設計

```python
class HeatSourceSensitiveMeshGenerator:
    """
    針對移動熱源敏感場景的網格生成器
    """
    
    def __init__(self, config):
        self.config = config
        
    def generate_for_mobile_sources(
        self,
        heat_sources: List[HeatSource],
        movement_envelopes: Dict[str, Box],
        passive_layers: List[PassiveLayer]
    ):
        """
        為移動熱源生成敏感網格
        
        Parameters:
        -----------
        heat_sources : List[HeatSource]
            熱源列表
        movement_envelopes : Dict[str, Box]
            每個熱源的運動信封
        passive_layers : List[PassiveLayer]
            被動層列表
        """
        mesh_zones = []
        
        # 1. 主動層網格（極細）
        active_zone = self._create_active_layer_zone(
            heat_sources, 
            movement_envelopes
        )
        mesh_zones.append(active_zone)
        
        # 2. 被動層網格（粗略）
        for layer in passive_layers:
            passive_zone = self._create_passive_layer_zone(layer)
            mesh_zones.append(passive_zone)
            
        # 3. 界面層（熱阻捕捉）
        interface_zone = self._create_interface_zone(
            heat_sources, 
            passive_layers
        )
        mesh_zones.append(interface_zone)
        
        # 4. 生成最終網格
        return self._combine_zones(mesh_zones)
        
    def _create_active_layer_zone(
        self,
        heat_sources: List[HeatSource],
        envelopes: Dict[str, Box]
    ) -> MeshZone:
        """
        創建主動層網格區域
        
        原則：
        - 熱源位置網格: 10-20 µm
        - 熱源信封內: 30-50 µm  
        - 熱源信封外但同層: 100 µm
        """
        zone = MeshZone()
        zone.name = "Active_Layer"
        zone.mesh_type = "uniform_fine"
        zone.base_size = 50e-6  # 50 µm
        
        # 收集所有熱源信封的並集
        envelope_union = self._union_envelopes(list(envelopes.values()))
        
        # 在信封內進行加密
        for source_id, envelope in envelopes.items():
            refinement = RefinementZone()
            refinement.box = envelope
            refinement.mesh_size = 20e-6  # 20 µm
            refinement.type = "absolute"  # 絕對尺寸
            zone.refinements.append(refinement)
            
        # 熱源正上方（Z方向）加密
        for source in heat_sources:
            above_zone = Box(
                min=(source.x_min - 100e-6, source.y_min - 100e-6, source.z_max),
                max=(source.x_max + 100e-6, source.y_max + 100e-6, source.z_max + 10e-6)
            )
            above_refinement = RefinementZone()
            above_refinement.box = above_zone
            above_refinement.mesh_size = 10e-6  # 10 µm - 極細！
            above_refinement.type = "absolute"
            zone.refinements.append(above_refinement)
            
        return zone
        
    def _create_passive_layer_zone(self, layer: PassiveLayer) -> MeshZone:
        """
        創建被動層網格區域
        
        原則：
        - 被動層不需要細網格
        - 只需捕捉整體熱流方向
        - 網格尺寸: 200-1000 µm
        """
        zone = MeshZone()
        zone.name = f"Passive_Layer_{layer.name}"
        zone.mesh_type = "uniform_coarse"
        
        # 被動層可以使用很粗的網格
        zone.base_size = max(layer.thickness * 5, 200e-6)
        
        return zone
        
    def _create_interface_zone(
        self,
        heat_sources: List[HeatSource],
        passive_layers: List[PassiveLayer]
    ) -> MeshZone:
        """
        創建界面層網格（捕捉熱阻）
        """
        zone = MeshZone()
        zone.name = "Interface_Layer"
        zone.mesh_type = "boundary_layer"
        
        # 界面需要邊界層網格
        zone.boundary_layer = {
            'n_layers': 5,
            'growth_ratio': 1.3,
            'first_layer_height': 5e-6  # 5 µm
        }
        
        return zone
```

### 10.5 被動層的簡化策略

對於被動層，由於它們只是傳遞熱量，可以使用大幅簡化的網格策略：

```python
class PassiveLayerSimplifier:
    """
    被動層網格簡化器
    
    原理：
    被動層不需要精確的溫度分佈
    只需要正確傳遞總熱流
    """
    
    def __init__(self):
        self.simplification_rules = {
            'thickness_ratio_threshold': 10,  # 厚度比
            'power_density_variation_threshold': 0.01,  # 功率密度變化
            'min_mesh_size_factor': 0.1  # 最小網格因子
        }
        
    def should_simplify(self, layer, source_power_density):
        """
        判斷是否應該簡化該被動層
        """
        # 規則 1: 如果被動層比熱源層厚很多，可以簡化
        if layer.thickness > source_layer.thickness * self.simplification_rules['thickness_ratio_threshold']:
            return True
            
        # 規則 2: 如果功率密度變化很小，可以簡化
        if self._power_variation_is_low(source_power_density):
            return True
            
        return False
        
    def get_simplified_mesh_size(self, layer, active_mesh_size):
        """
        獲取簡化後的網格尺寸
        
        原則：
        被動層網格可以是主動層的 10-100 倍
        """
        factor = max(
            self.simplification_rules['min_mesh_size_factor'],
            layer.thickness / active_layer_min_thickness
        )
        return active_mesh_size * factor
        
    def apply_simplification(self, layer, mesh):
        """
        應用簡化策略
        """
        simplified_mesh = Mesh()
        
        # 計算簡化後的網格數
        n_cells_z = max(2, int(layer.thickness / self.get_simplified_mesh_size(layer)))
        
        # 使用均勻粗網格
        simplified_mesh.nx = 10  # 被動層 X 方向
        simplified_mesh.ny = 10  # 被動層 Y 方向
        simplified_mesh.nz = n_cells_z  # Z 方向保持最少層數
        
        return simplified_mesh
```

### 10.6 溫差敏感度的網格需求

| 溫差敏感度 | 所需網格精度 | 主動層網格尺寸 | 被動層網格尺寸 |
|-----------|-------------|--------------|---------------|
| ΔT > 5°C | 低 | 50-100 µm | 200-500 µm |
| 1°C < ΔT < 5°C | 中 | 20-50 µm | 100-200 µm |
| 0.1°C < ΔT < 1°C | 高 | 10-20 µm | 50-100 µm |
| ΔT < 0.1°C | 極高 | 5-10 µm | 20-50 µm |

```
┌─────────────────────────────────────────────────────────────────┐
│                    網格精度與溫差關係                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ΔT (°C)                                                        │
│     │                                                           │
│   10 │                      ╭───────────────────                │
│     │                 ╭────╯                                       │
│    5 │            ╭──╯                                            │
│     │       ╭────╯                                                │
│    1 │  ╭────╯                                                    │
│     │ ╱╱                                                          │
│  0.1│╱                                                            │
│     └───────────────────────────────────────────────→ 網格精度     │
│     粗網格 ────────────────────────────────────────→ 細網格        │
│     (100µm)                                    (5µm)            │
│                                                                  │
│   經驗法則：                                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  網格尺寸 ≈ 熱源特徵尺寸 / 10                            │   │
│   │                                                          │   │
│   │  例如：                                                   │   │
│   │  - 熱源 100×100 µm → 網格 10×10 µm                       │   │
│   │  - 熱源 50×50 µm → 網格 5×5 µm                          │   │
│   │  - 熱源 10×10 µm → 網格 1×1 µm                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 10.7 完整配置示例

```yaml
# mobile_heat_sources_config.yaml
# 針對移動熱源敏感場景的完整配置

experiment:
  name: "Mobile_Heat_Sources_Analysis"
  type: "transient_thermal"
  
mesh:
  # 主動層配置（熱源所在層）
  active_layer:
    name: "Active_Die"
    mesh_type: "uniform_fine"
    base_size: 50e-6  # 50 µm
    
    refinements:
      # 熱源區域加密
      - zone: "heat_source_envelope"
        mesh_size: 20e-6  # 20 µm
        type: "absolute"
        
      # 熱源正上方加密
      - zone: "above_sources"
        mesh_size: 10e-6  # 10 µm
        z_range: [source_z_max, source_z_max + 50e-6]
        type: "absolute"
        
  # 被動層配置
  passive_layers:
    - name: "Interposer"
      thickness: 0.5e-3  # 500 µm
      mesh_type: "uniform_coarse"
      base_size: 200e-6  # 200 µm
      simplification_enabled: true
      
    - name: "Heat_Spreader" 
      thickness: 1.0e-3  # 1 mm
      mesh_type: "uniform_coarse"
      base_size: 500e-6  # 500 µm
      simplification_enabled: true
      
    - name: "Heat_Sink_Base"
      thickness: 2.0e-3  # 2 mm
      mesh_type: "uniform_coarse"
      base_size: 1000e-6  # 1 mm
      simplification_enabled: true
      
  # 界面層配置
  interfaces:
    - name: "Active_to_Interposer"
      boundary_layer:
        enabled: true
        n_layers: 5
        growth_ratio: 1.3
        first_layer_height: 5e-6  # 5 µm
        
  # 敏感度配置
  sensitivity:
    min_temperature_difference: 0.1  # 0.1°C
    required_mesh_precision: "high"
    
  # 性能配置
  performance:
    max_total_cells: 1000000
    parallel_mesh_generation: true
```

### 10.8 移動熱源網格策略總結

```
┌─────────────────────────────────────────────────────────────────┐
│            移動熱源敏感場景的網格策略總結                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  主動層策略                                               │  │
│   │  ═══════════════════════════════════════                  │  │
│   │                                                          │  │
│   │  • 計算所有熱源的運動範圍 (Envelope)                     │  │
│   │  • 對整個 Envelope 進行加密                              │  │
│   │  • 熱源正上方使用極細網格 (10-20 µm)                    │  │
│   │  • 保持信封外的網格粗獷                                  │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  被動層策略                                               │  │
│   │  ═══════════════════════════════════════                  │  │
│   │                                                          │  │
│   │  • 被動層大幅簡化 (10-100× 主動層網格)                   │  │
│   │  • 只捕捉整體熱流方向                                     │  │
│   │  • 不需要精確的溫度分佈                                   │  │
│   │  • 使用均匀粗網格                                         │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  界面層策略                                               │  │
│   │  ═══════════════════════════════════════                  │  │
│   │                                                          │  │
│   │  • 捕捉熱阻效應                                          │  │
│   │  • 使用邊界層網格 (Prismatic)                            │  │
│   │  • 第一層 5-10 µm                                        │  │
│   │  • 5-10 層，增长率 1.2-1.3                              │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                  │
│   預期效果：                                                      │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  • 總網格數減少 50-90%（被動層簡化）                    │  │
│   │  • 計算速度提升 5-20 倍                                  │  │
│   │  • 主動層精度保持 (ΔT < 0.1°C)                           │  │
│   │  • 被動層精度足夠（只關注平均溫度）                      │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. 參考文獻

### 11.1 Icepak 技術文檔

1. ANSYS Icepak Documentation
   - 版本: 2023 R1
   - 網格生成章節
   - 自適應網格加密

2. ANSYS Fluent Theory Guide
   - Chapter: Mesh and Mesh Quality
   - Chapter: Adaptive Mesh Refinement

### 11.2 網格生成經典文獻

3. Versteeg, H.K., & Malalasekera, W. (2007)
   - "An Introduction to Computational Fluid Dynamics: The Finite Volume Method"
   - Chapter 4: Mesh Generation

4. Löhner, R. (2008)
   - "Applied Computational Fluid Dynamics Techniques"
   - Chapters on mesh generation

### 11.3 自適應網格

5. Berger, M.J., & Oliger, J. (1984)
   - "Adaptive Mesh Refinement for Hyperbolic Partial Differential Equations"
   - Journal of Computational Physics

6. Löhner, R. (1987)
   - "An Adaptive Finite Element Scheme for Transient Problems in CFD"
   - Computer Methods in Applied Mechanics and Engineering

### 11.4 熱傳導網格

7. Incropera, F.P., & DeWitt, D.P. (2007)
   - "Fundamentals of Heat and Mass Transfer"
   - Chapter 3: Transient Conduction

8. Wang, Z., & Wang, J. (2016)
   - "Efficient Thermal Simulation for 3D-IC with Adaptive Mesh"
   - IEEE Transactions on Components, Packaging and Manufacturing Technology

### 11.5 鰭片散熱網格

9. Bejan, A., & Kraus, A.D. (2003)
   - "Heat Transfer Handbook"
   - Chapter 12: Heat Sinks

10. Bar-Cohen, A., & Kraus, A.D. (1988)
    - "Thermal Analysis and Control of Electronic Equipment"
    - Chapter on extended surfaces

---

## 附錄 A：網格質量標準速查表

| 指標 | 公式 | 理想值 | 可接受 | 警告 |
|-----|------|-------|-------|------|
| 長寬比 | max/L/min | 1 | < 20 | > 20 |
| Skewness | 1 - V_actual/V_ideal | 0 | < 0.85 | > 0.9 |
| 正交性 | min(cos θ) | 1 | > 0.2 | < 0.1 |
| 生長率 | h₂/h₁ | 1 | < 1.5 | > 2.0 |

## 附錄 B：常見鰭片網格參數

| 鰭片厚度 | 建議第一層厚度 | 層數 | 總厚度 |
|---------|---------------|------|-------|
| 0.5 mm | 0.05 mm | 8-10 | 0.5 mm |
| 1.0 mm | 0.1 mm | 8-10 | 1.0 mm |
| 2.0 mm | 0.2 mm | 10-15 | 2.0 mm |

## 附錄 C：邊界層增長公式

**幾何級數增長：**
```
h_i = h_1 × r^(i-1)

其中：
- h_i = 第 i 層厚度
- h_1 = 第一層厚度
- r = 增長率 (1.1 ~ 1.3)
```

**總厚度：**
```
H = h_1 × (1 - r^n) / (1 - r)

其中 n = 總層數
```

---

## 版本信息

- **文件版本**: 1.1
- **創建日期**: 2024-02-12
- **更新日期**: 2024-02-12
- **作者**: Thermal Solver Development Team
- **適用版本**: Thermal Solver v1.0+

---

*本文檔持續更新，最新版本請參考項目文檔庫。*

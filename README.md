# Thermal Simulator (ThermoSim) v10.0

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**ThermoSim** 是一個專為晶片堆疊 (Chip Stacking) 與先進封裝設計的高精度 3D 熱模擬器。它採用 **Feature-Driven Scan-Line Algorithm (FD-SLA)** 結合 **Active/Sparse Representation**，在保證 floorplan 對齊精度的同時，大幅降低記憶體需求。

## ✨ 核心特色

*   **Precision Hex Meshing**: 保證網格與 Floorplan 功能區塊完美對齊。
*   **SmartCells Lite**: 針對超薄層 (如 TIM, Oxide) 的單層網格優化，顯著減少 Z 軸網格數量。
*   **Non-Conformal Meshing (NCM)**: 支援組件間網格解耦，提升局部加密彈性。
*   **GPU 加速**: 支援 CUDA/CuPy 加速，大型矩陣求解速度提升 10x-50x。
*   **雙重度量報告**: 提供 Nodal Max (Peak) 與 Element Max (Avg) 兩種溫度指標，滿足不同設計規範需求。

## 🚀 快速開始

### 環境需求
*   Python 3.10+
*   NVIDIA GPU (可選，推薦用於加速)

### 安裝步驟

1.  **建立虛擬環境**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate   # Windows
    # source venv/bin/activate  # Linux/macOS
    ```

2.  **安裝依賴**
    ```bash
    pip install -r requirements.txt
    ```
    *若需 GPU 加速，請額外安裝:*
    ```bash
    pip install cupy-cuda12x  # 請依據您的 CUDA 版本選擇
    ```

## 💻 使用指南

### 1. 執行模擬 (CLI)
使用 `ThermoSim_NCM.py` 執行模擬：

```powershell
python ThermoSim_NCM.py <sim_config> <params_config> [options]
```

**參數說明**:
*   `sim_config`: 模擬配置檔路徑 (e.g., `projects/chip_stack/test_smart_stack.config`)
*   `params_config`: 材料參數檔路徑 (e.g., `projects/chip_stack/params_stack.config`)
*   `--mesh_size`: (可選) 強制覆寫最大網格尺寸 (m)。
*   `--check`: (可選) 僅檢查網格，不執行求解。
*   `--show`: (可選) 計算後開啟互動式視覺化。

**範例**:
```powershell
python ThermoSim_NCM.py projects/chip_stack/test_smart_stack.config projects/chip_stack/params_stack.config --show
```

### 2. 使用 GUI (ThermoStudio)
啟動圖形介面進行結果分析與切片觀察：
```powershell
python ThermoStudio.py
```

## 🏗️ 專案結構

*   `ThermoSim_NCM.py`: 主程式 (Solver Entry Point)。
*   `fem_engine.py`: 有限單元求解核心。
*   `mesh_core.py` / `mesh_core_ncm.py`: 網格生成核心。
*   `config_parser.py`: 配置與參數解析器。
*   `projects/`: 專案範例與配置檔。
*   `docs/`: 技術文件與開發指南。

## 📄 授權
MIT License

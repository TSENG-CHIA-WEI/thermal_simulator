# Thermal Simulator (ThermoSim) v10.2

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**ThermoSim** 是一個專為晶片堆疊 (Chip Stacking) 與先進封裝設計的高精度 3D 熱模擬器。它採用 **Feature-Driven Scan-Line Algorithm (FD-SLA)** 結合 **Active/Sparse Representation**，在保證 floorplan 對齊精度的同時，大幅降低記憶體需求與計算時間。

## ✨ 核心特色

*   **Precision Hex Meshing**: 保證網格與 Floorplan 功能區塊完美對齊，不遺漏細微特徵。
*   **SmartCells Lite**: 針對超薄層 (如 TIM, Oxide, Bonding) 優化，顯著減少 Z 軸網格數量，避免硬體資源浪費。
*   **Non-Conformal Meshing (NCM)**: 支援組件間網格解耦，可針對發熱熱點 (Hotspots) 進行局部網格加密。
*   **高效能 FEM 運算引擎**: 
    * 採用 **In-Place CSR Assembly**，COO 緩衝記憶體降低 69%。
    * 分析型 Jacobian (Analytical Jacobian) 加速矩陣建立。
    * 支援 **CUDA/CuPy GPU 加速**，大型求解 (700萬+ 網格) 縮短至 20 秒內完成。

---

## 🚀 快速開始 & 環境安裝

### 環境需求
*   作業系統：Windows / Linux / macOS
*   Python 3.10+
*   NVIDIA GPU (可選，強烈推薦用於加速 CUDA 求解)

### 安裝步驟（一鍵完成）

1. **取得最新程式碼**
   ```bash
   git clone https://github.com/TSENG-CHIA-WEI/thermal_simulator.git
   cd thermal_simulator
   ```

2. **執行自動環境設定腳本**
   ```bash
   python setup_env.py
   ```
   此腳本會自動完成以下工作：
   - 建立 Python 虛擬環境 (`venv/`)
   - 安裝所有 CPU 核心套件 (NumPy, SciPy, PyVista, PyQt5)
   - **自動偵測您的 CUDA 版本**，安裝對應的 GPU 加速套件 (CuPy)
     - CUDA 11.x → `cupy-cuda11x`
     - CUDA 12.x → `cupy-cuda12x`
     - 無 CUDA → 跳過（純 CPU 模式，程式仍可正常運行）

3. **啟動虛擬環境**
   ```bash
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

---

## � 詳細使用教學

本模擬器採用 **配置檔 (Config) 驅動**。您只需要定義結構 (幾何) 和材料參數，執行主程式即可獲得結果。

### 1. 準備設定檔 (Configuration Files)

一個完整的模擬需要兩個主要的 `.config` 檔案：
1. **模擬配置檔 (`sim.config`)**: 定義幾何層次 (Layers)、邊界條件 (Boundary Conditions) 以及網格切分設定。
2. **材料參數檔 (`params.config`)**: 定義各個 Block (發熱源或結構) 的熱傳導係數 (k) 與功耗 (Power)。

**範例配置結構 (可參考 `projects/chip_stack/box_sim.config`)**:
```ini
[Meshing]
base_dx = 0.001
base_dy = 0.001
base_dz = 0.001

[Layer:SoC]
z_start = 0.000
z_end = 0.001
boxes = SoC_Core, SoC_L2Cache

[Boundary_Conditions]
Convection_Top = 20.0, 60000.0   # T_inf, h
```

### 2. 執行命令列 (CLI) 模擬

程式的主入口為 `ThermoSim_NCM.py`。

```bash
python ThermoSim_NCM.py <sim_config> <params_config> [options]
```

**命令列參數說明**:
* `<sim_config>`: 幾何與網格配置檔。
* `<params_config>`: 材料與熱源配置檔。
* `--mesh_size <數值>`: (可選) 強制覆寫 `sim.config` 內的全域最大網格尺寸 (m)。數值越小，網格越多（精度越高，計算越久）。
* `--check`: (可選) 僅執行 Mesh 生成與記憶體評估，跳過 FEM 求解矩陣。
* `--show`: (可選) 若有安裝 PyVista 等視覺化模組，可在運算結束後開啟 3D 視窗。

**執行範例 (以晶片堆疊專案為例)**:
```bash
python ThermoSim_NCM.py projects/chip_stack/test_smart_stack.config projects/chip_stack/params_stack.config --mesh_size 0.0004
```

### 3. 查看報告與輸出

模擬完成後，程式會在 `<sim_config>` 同目錄下生成兩個檔案：
1. **`simulation_report.txt`**: 包含各個功能區塊的溫度數據 (Peak Max, Element Avg, Min)。
2. **`output.vtk`**: 包含完整的 3D 溫度分佈。您可以使用開源軟體 **ParaView** (https://www.paraview.org/) 開啟此檔案，進行截面、等溫線 (Isocontours) 的進階分析。

---

## 🏗️ 專案結構說明

* `ThermoSim_NCM.py`: 主程式 (Solver Entry Point)，負責組合流程。
* `fem_engine.py`: 有限單元求解核心，負責 GPU/CPU 矩陣組裝與求解。
* `mesh_core_ncm.py`: 網格生成核心，負責執行佈局剖析與非對齊網格生成。
* `layout_parser.py`: Floorplan 階層解析與熱源分布計算。
* `config_parser.py`: 解析 `.config` 文字檔。
* `projects/`: 內含不同測試案例 (如 `chip_stack`, `demo`)。

---

## ❓ 常見問題 (FAQ)

**Q: 執行時出現記憶體不足 (OOM) 怎麼辦？**
A: 請加大 `--mesh_size` 的數值（例如從 0.0002 改為 0.0005），或者在配置檔中針對厚度大的 Base 層（如 Substrate 或 Heatsink）啟用單層 SmartCells 以減少 Z 軸元素。

**Q: Git Push 失敗，顯示紅字錯誤怎麼辦？**
A: PowerShell 有時會將 Git 正常的輸出 (進度條或提示碼) 當作錯誤文字 (紅字) 來顯示。只要最後出現 `main -> main`，就表示已成功上傳。若要避免紅字，建議改用 Windows 內建的 `cmd` (Command Prompt) 或 Git Bash 執行 `git push`。

**Q: 如何確認我的 GPU 有被正確調用？**
A: 在執行程式的終端輸出中，若看到 `[Solver] Device: GPU (CuPy Detected & Verified)` 則代表成功啟用。若未安裝 CuPy，系統會自動 fallback 退回到 SciPy (CPU) 模式並顯示提醒。

## 📄 授權
MIT License

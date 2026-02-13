# Thermal Simulator v10.0 User Guide (Precision Hex)

Welcome to the **ThermoSim v10.0**. This major update introduces **Precision Hex Meshing**, which guarantees perfect alignment with your floorplans while significantly reducing memory usage.

## 📁 Project Structure

*   `ThermoSim.py`: The core CLI solver and report generator.
*   `ThermoStudio.py`: The Graphical User Interface (GUI) for slicing and analysis.
*   `projects/`: Folder containing model configurations (e.g., `chip_stack`).
*   `mesh_core.py`: The new Conformal "Gap-Filler" meshing kernel.

## 💻 Quick Start & Installation

### 1. Setup
```bash
python -m venv venv
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

---

## 🛠️ Operating Instructions

### 1. Launching the GUI
Visualize and slice your 3D results in real-time:
```powershell
python ThermoStudio.py
```

### 2. Running via Command Line (CLI)
```powershell
python ThermoSim.py <path/to/box_sim.config> <path/to/params.config>
```

---

## 📝 Preparing Input Files

### A. `box_sim.config` (Geometry & BCs)
*   **[Mesh]**: Set `MaxElementSize = 0.3mm`. The Gap-Filler logic will ensure this is the *maximum* size, while automatically refining further at heat source boundaries.
*   **[Box:Name]**:
    *   `Origin` / `Size`: Standard 3D dimensions (meters).
    *   `Priority`: Set lower for background (Mold) and higher for functional layers (Die).
    *   `PowerFace`: Use `Top` or `Bottom` for thin-layer surface heating.
    *   `Floorplan`: Path to your CSV power map.

### B. Power Map CSV
The simulator automatically injects **Midpoint Refinement** into the mesh for every block defined in your CSV, ensuring the absolute peak temperature is captured.

---

## 📊 Understanding the Report (`simulation_report.txt`)

Version 10.0 introduces **Dual-Metric Reporting**:

| Column | Description |
| :--- | :--- |
| **Node Max (Peak)** | The absolute highest temperature detected in the layer. Match this to your design limits. |
| **Elem Max (Average)** | The average temperature across the hottest element. Match this to coarse-mesh tools (e.g., Icepak). |
| **Avg (C)** | The weighted average temperature of the entire component. |

> [!TIP]
> **Anti-Bleeding Logic**: The report uses element-filtering to ensure hotspots from one layer (like a hot Die) do not "bleed" into the reports of adjacent layers (like the Mold Compound), providing 100% accurate component-wise tracking.

---

## 🚀 Performance Optimization (GPU)
If you have an NVIDIA GPU, install `cupy` to enable the high-speed solver:
```powershell
pip install cupy-cuda12x 
```
The simulator will provide a **10x-50x speedup** on the matrix solve step.

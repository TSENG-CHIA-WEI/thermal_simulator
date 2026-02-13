# System Requirements & Environment Adaptation

This document outlines the software and hardware requirements to run **ThermoSim v9.0** and **ThermoStudio**.

## 1. Software Prerequisites

### Base Environment
*   **OS**: Windows 10/11 (Recommended) or Linux (Ubuntu 20.04+).
*   **Python**: Version 3.8 or higher (3.10+ recommended).

### Core Dependencies (CPU Only)
Install these packages to run the standard version of the simulator:

```text
numpy>=1.21.0
scipy>=1.7.0
pyvista>=0.37.0
pyvistaqt>=0.9.0
PyQt5>=5.15.0
matplotlib>=3.5.0
```

### Installation Command
```powershell
pip install numpy scipy pyvista pyvistaqt PyQt5 matplotlib
```

---

## 2. Hardware Acceleration (Optional)

To enable the high-performance **GPU Solver** (50x speedup), you need an NVIDIA GPU.

### GPU Requirements
*   **Hardware**: NVIDIA GPU with Compute Capability 6.0+ (Pascal or newer).
*   **Drivers**: Latest NVIDIA Studio or Game Ready Driver.
*   **Toolkit**: CUDA Toolkit 11.x or 12.x (Must match Cupy version).

### GPU Installation
Install the `cupy` library corresponding to your CUDA version:

*   **CUDA 11.x**: `pip install cupy-cuda11x`
*   **CUDA 12.x**: `pip install cupy-cuda12x`

_Note: If `cupy` is not detected, ThermoSim will automatically fall back to the CPU solver._

---

## 3. Visualization Support
**ThermoStudio** relies on Qt5 and VTK.
*   **OpenGL**: Ensure your graphics drivers support OpenGL 3.3+.
*   **Remote Desktop**: If running via RDP, you may need a software rasterizer if OpenGL fails, or use `Allow-OpenGL-over-RDP` policies.

---

## 4. File Output Storage
*   Simulations generate large VTK files (~50MB - 500MB per run depending on mesh size).
*   Ensure at least **10 GB** of free disk space for the `projects/` directory if running multiple high-res automated sweeps.

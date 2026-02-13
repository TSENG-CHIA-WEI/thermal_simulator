# Technology Support & Feature List (tech.md)

This document provides a technical deep-dive into the features and physical kernels implemented in **ThermoSim v10.0 (Precision Hex)**.

---

## 1. Physical Kernel: 3D Steady-State FEM
The simulator uses a Linear Finite Element Method (FEM) solver to solve the heat conduction equation:
$$\nabla \cdot (k \nabla T) + Q = 0$$

*   **Element Type**: 8-node Hexahedral (Brick) elements with tri-linear shape functions.
*   **Anisotropic Conductors**: Full support for diagonal anisotropic thermal conductivity ($k_x, k_y, k_z$).
*   **Energy Balance**: High-precision reporting (~0.0005% error) of Total Power vs. Total Convective Loss.

---

## 2. Mesh Engine: Conformal Precision Hex (v10.0)
The v10.0 engine moves beyond simple adaptive meshing to a **Hard-Constraint / Gap-Filler** architecture:
*   **Conformal Alignment**: Every box boundary and every floorplan power block is treated as a "Hard Constraint." The mesh engine *strictly* aligns nodes to these features.
*   **Gap-Filler Logic**: Instead of subdividing per-box, the engine identifies gaps between all constraints and subdivides only the necessary regions. This eliminates "Dummy Mesh" and reduces element count by up to 80%.
*   **Ultra-Thin Resolution**: The `min_gap` threshold is set to **1nm**, allowing the simulator to resolve sub-micron layers (e.g., 0.45µm bonding oxide) without numerical merging.
*   **Peak Refinement (FAM)**: Automatically injects mid-point nodes into active heat sources to capture absolute peak temperatures without smearing.

---

## 3. Boundary Condition Matrix
*   **Robin (Convection)**: $q = h(T - T_{ref})$. Applied to any face of any box.
*   **Neumann (Heat Flux/Power)**:
    *   **Volumetric**: Power distributed throughout the 3D volume of a component.
    *   **Surface (PowerFace)**: Top/Bottom surface heat loads, perfect for modeling thin FEOL heating.
*   **Dirichlet (Fixed Temp)**: Penalty-method based implementation for constant temperature boundaries.

---

## 4. Reporting & Validation (v10.0)
*   **Nodal vs. Element Reporting**:
    *   **Node Max (Peak)**: The absolute maximum temperature at any individual node. Essential for hotspot verification.
    *   **Elem Max (Average)**: The maximum element-averaged temperature. Useful for matching Icepak coarse-mesh results.
*   **Interface Bleeding Protection**: Nodal statistics are strictly filtered by component-specific element sets to ensure Mold Compound doesn't "bleed" die-level hotspots into its report.

---

## 5. Solver & Performance
*   **Sparse Computing**: Utilizes CSR (Compressed Sparse Row) matrices.
*   **HPC (GPU) Path**: Automatic detection of **NVIDIA CUDA** via CuPy for 10x-50x speedups.
*   **Solver Logic**: Preconditioned Conjugate Gradient (PCG) for rapid convergence on multi-million element meshes.

---

## 6. Visualization (ThermoStudio)
*   **Gap-Filled View**: Visualization reflects the optimized conformal grid.
*   **Interactive Slicing Engine**: Real-time X/Y/Z plane interpolation.
*   **Reproduction**: `run_meta.json` records all constraints for R&D traceability.

---

## 7. Supported Units
*   **Length**: `m`, `cm`, `mm`, `um`, `nm`.
*   **Temperature**: `C`, `K`.
*   **Power**: `W`.
*   **Conductivity**: `W/mK`.

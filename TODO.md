# Project Roadmap & TODO

## 🔴 Critical / Immediate
- [ ] **Git Configuration**: User needs to set `user.email` and `user.name` to enable commits.
- [ ] **CI/CD Setup**: Configure GitHub Actions for automated testing on push.
- [ ] **Dependency Lock**: Generate a precise `requirements.txt` with frozen versions (currently using ranges).

## 🟡 Improvements & Features
- [ ] **Unit Test Coverage**: Expand tests in `tests/` to cover corner cases (e.g., overlapping geometry).
- [ ] **Material Library**: Move hardcoded material properties to an external JSON/YAML database.
- [ ] **Transient Analysis**: Implement time-dependent solver ($\rho C_p \frac{\partial T}{\partial t}$) for pulse heating.
- [ ] **Non-Conformal Meshing**: Research "hanging node" support to further reduce element count in non-critical regions.

## 🟢 Documentation & Usability
- [ ] **API Documentation**: Add docstrings to all core functions in `fem_engine.py`.
- [ ] **Example Gallery**: Add visual examples to the `mesh_study_demo` folder.
- [ ] **Installation Script**: Create a `setup.py` or `install.bat` for one-click setup.

## 🐛 Known Issues / Optimization
- [ ] **Large Mesh Memory**: Sparse matrix assembly peaks in memory usage for >5M elements. Investigate iterative assembly.
- [ ] **GPU Fallback**: Improve error messaging when CUDA is installed but version mismatched.

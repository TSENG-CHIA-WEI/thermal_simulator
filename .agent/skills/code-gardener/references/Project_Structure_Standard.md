# Standard Project Structure

Target layout for a clean simulation project.

```
project_root/
├── src/                  # Source Code (Pipelline Logic)
│   ├── core/             # Core Algorithms (Solver, Mesh)
│   ├── utils/            # Shared Utilities (Parsers, Config)
│   └── main.py           # Entry Point
├── experiments/          # User Experiments (Not Core Code)
│   ├── campaign_v1/      # Specific Experiment Run
│   └── sandbox/          # Throwaway scripts
├── tests/                # Unit & Integration Tests
├── data/                 # Input Data (Geometry, CSV)
├── output/               # Simulation Results (Artifacts)
├── docs/                 # Documentation
├── legacy/               # Deprecated/Old Code (Graveyard)
└── requirements.txt
```

## Anti-Patterns (What to fix)

1. **Root Clutter**: Too many `debug_X.py` in the root folder.
   - **Fix**: Move to `tests/` or `experiments/sandbox/`.
2. **Output Pollution**: `*.vtk`, `*.png` generated in root.
   - **Fix**: Configure output to `output/` subdir.
3. **Hardcoded Paths**: Scripts that break if moved (e.g., `file = "../data.csv"`).
   - **Fix**: Use `os.path.join(PROJECT_ROOT, ...)` logic.

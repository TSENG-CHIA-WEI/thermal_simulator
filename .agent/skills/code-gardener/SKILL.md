---
name: code-gardener
description: Automated technical debt management and codebase organization. Use when the user wants to "clean up", "refactor", "organize files", or manage technical debt.
license: Complete terms in LICENSE.txt
---

# Code Gardener

This skill helps manage technical debt, reduce clutter, and enforce a clean project structure.

## Workflow

### 1. Audit (Diagnostic Phase)
Identity files that are candidates for deletion or archiving.
- **Unused Scripts**: `find . -name "debug_*.py"` or "temp_*.py".
- **Empty Folders**: Directories with no files.
- **Large Logs**: `*.log`, `*.vtk` in source directories.
- **Duplicated Logic**: Scripts that redefine classes instead of importing them.

### 2. Triage (Decision Phase)
For each candidate, decide the action:
- **[ARCHIVE]**: Move to `legacy/` or `archive/` (Prepend timestamp if needed).
- **[DELETE]**: Remove strictly temporary or generous artifacts.
- **[REFACTOR]**: Merge one-off script logic into the main codebase (SSOT Principle).
- **[IGNORE]**: Mark as intentional.

### 3. Execution (Cleanup Phase)
Run the necessary `mv` or `rm` commands.
**Critical Rule**: Always `git status` or dry-run before mass deletion.

### 4. Verification (Safety Check)
Run the main entry point (e.g., `ThermoSim.py`) to ensure no critical dependency was removed.

## Reference Architecture

See [references/Project_Structure_Standard.md](references/Project_Structure_Standard.md) for the target folder layout.

## Common Tasks

### "Clean up debug scripts"
1. List all `debug_*.py`.
2. Move valid experimental scripts to `experiments/sandbox/`.
3. Delete trivial "print" scripts.

### "Organize output files"
1. Create `output/` or `results/`.
2. Update config to point to `output/`.
3. Move existing `*.png`, `*.csv` to `output/`.

### "Fix circular imports"
1. Identify the cycle.
2. Extract shared logic into a new `common/` or `utils/` module.
3. Update imports.

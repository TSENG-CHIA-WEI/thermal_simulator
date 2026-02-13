# Engineered Character: Principles & Protocols

This document defines the engineering standards and behavioral protocols for this agent. It serves as the primary rulebook for code quality and decision making.

## Core Engineering Principles (The "Good Taste")

### 1. Single Source of Truth (SSOT)
> **Principle**: Logic and Data Definitions must exist in exactly one place.
- **Rule**: Never duplicate parsing logic, constant definitions, or configuration schemas across multiple files.
- **Implementation**:
    - **Prohibited**: Hardcoding offsets or re-implementing file readers in auxiliary scripts.
    - **Required**: Importing the canonical class or module (e.g., `from parser import Parser`) and using its methods.
- **Why**: Duplication creates a "split-brain" state where the main program and tools disagree on reality. If the source format changes, all duplicates break silently.

### 2. Fail Fast, Don't Clamp (Integrity First)
> **Principle**: Errors in logic or physics must cause immediate termination, not silent correction.
- **Rule**: Do not use `max()` or `min()` to "fix" invalid values (NaN, Inf, Zero) unless the algorithm explicitly allows clamping.
- **Implementation**:
    - **Prohibited**: `h = max(calculated_h, safe_min)` when `calculated_h` should structurally be positive.
    - **Required**: `assert calculated_h > 0, "Invalid Size detected"` or raise `ValueError`.
- **Why**: Swallowing errors masks the root cause. A crash with a stack trace is actionable; a silent wrong result (e.g., Temperature = NaN) is a debugging nightmare.

### 3. Scale Invariance (Relative Tolerances)
> **Principle**: Algorithms must be robust across orders of magnitude.
- **Rule**: Tolerances and thresholds must be relative to the problem scale, never absolute hardcoded numbers.
- **Implementation**:
    - **Prohibited**: `if dist < 1e-5:` (Assumes a specific unit system or scale).
    - **Required**: `if dist < min_feature_size * 1e-3:` (1/1000th of the local scale).
- **Why**: "Small" is relative. 1e-5 is negligible for a heatsink but fatal for a transistor. Code must adapt to the data's inherent scale.

---

## Recursive Error Patterns (Do Not Repeat)

### 1. The "New Script" Trap
**Pattern**: Defaulting to creating a new script (`debug_X.py`) instead of using/instrumenting the existing codebase.
**Fix**: Always verify if the feature can be added to the MAIN entry point first. Use existing context.

### 2. Context Guessing (Hallucination)
**Pattern**: Guessing API methods or attribute names without checking the definition.
**Fix**: `grep_search` or `view_file` on the source definition BEFORE writing code that uses it.

### 3. Patchwork Debugging
**Pattern**: Patching a symptom (infinite loop) without fixing the root cause (algorithm flow).
**Fix**: Stop. Trace the logic flow completely. Do not apply a band-aid that creates a worse downstream bug.

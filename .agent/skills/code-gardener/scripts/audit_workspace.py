import os
import glob
import re

def main():
    root = os.getcwd()
    print(f"Auditing Workspace: {root}")
    print("-" * 40)
    
    files = [f for f in os.listdir(root) if os.path.isfile(f)]
    py_files = [f for f in files if f.endswith('.py')]
    
    # 1. Clutter Detection (Root Level)
    print(f"\n[Root Clutter Analysis]")
    candidates = []
    for f in files:
        if f.startswith('debug_') or f.startswith('test_') or f.startswith('temp_'):
            candidates.append((f, "tests/ or experiments/sandbox/"))
        elif f.startswith('plot_') or f.startswith('visualize_'):
            candidates.append((f, "tools/visualization/"))
        elif f.endswith('.log') or f.endswith('.vtk') or f.endswith('.png'):
            candidates.append((f, "output/"))
            
    if candidates:
        print(f"Found {len(candidates)} clutter candidates:")
        for f, dest in candidates:
            print(f"  - {f:<30} -> SUGGEST MOVE TO: {dest}")
    else:
        print("  Root directory looks clean!")

    # 2. Orphan Detection (Heuristic)
    # Check which .py files are imported by others
    print(f"\n[Dependency Analysis]")
    imports = {}
    for f in py_files:
        with open(f, 'r', encoding='utf-8', errors='ignore') as d:
            content = d.read()
            # Regex for "import X" or "from X import Y"
            # Simple heuristic
            found = re.findall(r'^(?:from|import)\s+(\w+)', content, re.MULTILINE)
            for module in found:
                imports[module] = imports.get(module, 0) + 1
                
    # Check which files (modules) are never imported
    # Convert filename to module name
    orphans = []
    for f in py_files:
        mod_name = f[:-3]
        if mod_name not in imports and f not in ['main.py', 'ThermoSim.py', 'ThermoStudio.py']:
            orphans.append(f)
            
    if orphans:
        print(f"Potential Orphans (Not imported by others, possibly Entry Points or Dead Code):")
        for f in orphans:
            print(f"  - {f}")
    
    print("-" * 40)
    print("Recommendation: Run 'code-gardener' skill to organize these files.")

if __name__ == "__main__":
    main()

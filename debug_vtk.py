import pyvista as pv
import sys
import os

files = [
    "projects/demo/output.vtk",
    "projects/chip_stack/output.vtk"
]

for f in files:
    if os.path.exists(f):
        try:
            mesh = pv.read(f)
            print(f"--- Checking {f} ---")
            print("Cell Arrays:", mesh.cell_data.keys())
            if "BoxID" in mesh.cell_data:
                print("  [OK] BoxID found.")
                print("  Values:", mesh.cell_data["BoxID"])
            else:
                print("  [FAIL] BoxID MISSING.")
        except Exception as e:
            print(f"Error reading {f}: {e}")
    else:
        print(f"File not found: {f}")

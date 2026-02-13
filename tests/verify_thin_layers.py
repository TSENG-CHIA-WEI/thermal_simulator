import sys
import os
import numpy as np
from config_parser import SimConfig, BoxDef
from mesh_core import ActiveMeshGenerator

def test_thin_layer_constraint():
    print("--- Running Thin Layer Constraint Verification ---")
    
    # 1. Setup a synthetic problem: 5nm Box in a 1um world
    # Global mesh size = 100nm (0.1um)
    # Without constraint, 5nm would be flattened or ignored.
    
    box_thin = BoxDef(
        name="ThinOxide",
        origin=(0, 0, 0.5e-6), # 0.5um Z
        size=(1e-6, 1e-6, 5e-9), # 5nm thick
        material_id=1
        # min_elements is NOT specified, relying on auto-heuristic (5nm < 100nm)
    )
    
    # Dummy box to define the world
    box_world = BoxDef(
        name="World",
        origin=(0, 0, 0),
        size=(1e-6, 1e-6, 1e-6), # 1um cube
        material_id=0
    )
    
    cfg = SimConfig(
        boxes=[box_world, box_thin],
        max_element_size=100e-9 # 100nm
    )
    
    # 2. Generate Mesh
    gen = ActiveMeshGenerator(cfg, max_element_size=100e-9)
    try:
        mesh = gen.generate()
    except Exception as e:
        print(f"FAIL: Mesh generation crashed: {e}")
        sys.exit(1)
        
    # 3. Analyze Z-Grid around the thin layer
    z_min = 0.5e-6
    z_max = 0.5e-6 + 5e-9
    
    # Find ticks strictly inside [z_min, z_max] inclusive
    # Floating point tolerance is key here
    tol = 1e-12
    ticks_on_layer = [z for z in mesh.z_grid if (z >= z_min - tol) and (z <= z_max + tol)]
    
    print(f"Layer Z-range: [{z_min*1e9:.2f}, {z_max*1e9:.2f}] nm")
    print(f"Found ticks: {[f'{t*1e9:.2f}' for t in ticks_on_layer]}")
    
    # We expect: Start, End, and (min_elements-1) internal ticks.
    # Total ticks = min_elements + 1
    # For min_elements=3, we want 4 ticks (Element 1, Element 2, Element 3)
    
    n_ticks = len(ticks_on_layer)
    if n_ticks >= 4:
        print(f"PASS: Thin layer constraint verified. Found {n_ticks} ticks (Expected >= 4).")
    else:
        print(f"FAIL: Insufficient subdivision. Found {n_ticks} ticks (Expected >= 4). Constraint ignored!")
        sys.exit(1)

if __name__ == "__main__":
    test_thin_layer_constraint()

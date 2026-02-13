import sys
import os
import numpy as np
from config_parser import SimConfigParser
from mesh_core import ActiveMeshGenerator

def diagnostic(config_path, params_path):
    print(f"--- Diagnostic Run: {config_path} ---")
    parser = SimConfigParser()
    cfg = parser.parse(config_path, params_path)
    
    # Generate mesh
    gen = ActiveMeshGenerator(cfg.sim_config, max_element_size=0.0003)
    mesh = gen.generate()
    
    # Check for Dropped Boxes
    # mesh.box_ids maps element -> index in cfg.sim_config.boxes
    unique_ids = np.unique(mesh.box_ids)
    
    print("\n[Mesh vs Config Audit]")
    for i, box in enumerate(cfg.sim_config.boxes):
        status = "OK" if i in unique_ids else "!!! DROPPED (NO ELEMENTS) !!!"
        count = np.count_nonzero(mesh.box_ids == i)
        print(f"Box {i:2d} | Name: {box.name:20s} | Elements: {count:10d} | Status: {status}")
        
        if status != "OK":
            print(f"    - Details: Origin {box.origin}, Size {box.size}")
            print(f"    - Why? Box thickness might be smaller than mesh Z-resolution.")

if __name__ == "__main__":
    c = "projects/chip_stack/box_sim.config"
    p = "projects/chip_stack/params_stack.config"
    diagnostic(c, p)

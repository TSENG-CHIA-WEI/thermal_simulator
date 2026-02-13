import sys
import os
import numpy as np
from config_parser import ConfigParser
from mesh_core import ActiveMeshGenerator

def test_mesh_baseline():
    print("--- Running Mesh Baseline Verification ---")
    parser = ConfigParser()
    config_path = "projects/chip_stack/box_sim.config"
    params_path = "projects/chip_stack/params_stack.config"
    
    if not os.path.exists(config_path):
        print(f"Error: Missing {config_path}")
        return
        
    parser.parse_model_params(params_path)
    parser.parse_sim_config(config_path)
    cfg = parser
    
    # Fix floorplan paths relative to the config file
    config_dir = os.path.dirname(config_path)
    for box in cfg.sim_config.boxes:
        if box.floorplan_file:
            box.floorplan_file = os.path.join(config_dir, box.floorplan_file)

    # Use the max_h that was used for the 127.66C run
    gen = ActiveMeshGenerator(cfg.sim_config, max_element_size=0.0003)
    mesh = gen.generate()
    
    n_act = mesh.num_active_elements
    print(f"Active Elements: {n_act}")
    
    # Expected approximate count from previous stable run
    EXPECTED_ACT_MIN = 3000000
    EXPECTED_ACT_MAX = 3100000
    
    if EXPECTED_ACT_MIN <= n_act <= EXPECTED_ACT_MAX:
        print("PASS: Baseline verified: Element count within range.")
    else:
        print(f"FAIL: Baseline FAILED: Element count {n_act} is outside expected range!")
        sys.exit(1)

if __name__ == "__main__":
    test_mesh_baseline()

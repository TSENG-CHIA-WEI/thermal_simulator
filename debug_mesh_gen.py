import sys
import os
import time
from config_parser import ConfigParser
from mesh_core import ActiveMeshGenerator

def main():
    config_path = "projects/chip_stack/box_sim.config"
    params_path = "projects/chip_stack/params_stack.config"
    
    print(f"Loading config from {config_path}...")
    try:
        parser = ConfigParser()
        parser.parse_model_params(params_path)
        parser.parse_sim_config(config_path)
        cfg = parser.sim_config
    except Exception as e:
        print(f"Config Parse Error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Initializing Mesh Generator...")
    gen = ActiveMeshGenerator(cfg)
    
    print("Generating Mesh...")
    try:
        t0 = time.time()
        mesh = gen.generate()
        dt = time.time() - t0
        print(f"Mesh Generated in {dt:.4f}s")
        print(f"Active Elements: {mesh.num_active_elements}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Mesh Generation Failed: {e}")

if __name__ == "__main__":
    main()

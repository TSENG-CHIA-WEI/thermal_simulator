"""
Debug script to diagnose mesh generation issue
"""
import sys
import numpy as np

sys.path.insert(0, 'c:/Users/Calvin Tseng/Desktop/thermal_simulator')

from config_parser import ConfigParser
from mesh_core import ActiveMeshGenerator

config_path = 'c:/Users/Calvin Tseng/Desktop/thermal_simulator/projects/chip_stack/box_sim.config'

cfg = ConfigParser()
cfg.parse_sim_config(config_path)

print("Box Bounds:")
for box in cfg.sim_config.boxes:
   print(f"  {box.name}: x=[{box.xmin}, {box.xmax}] y=[{box.ymin}, {box.ymax}] z=[{box.zmin}, {box.zmax}]")

print("\nGlobal Bounds:")
xmins = [box.xmin for box in cfg.sim_config.boxes]
xmaxs = [box.xmax for box in cfg.sim_config.boxes]
ymins = [box.ymin for box in cfg.sim_config.boxes]
ymaxs = [box.ymax for box in cfg.sim_config.boxes]
zmins = [box.zmin for box in cfg.sim_config.boxes]
zmaxs = [box.zmax for box in cfg.sim_config.boxes]

global_min = [min(xmins), min(ymins), min(zmins)]
global_max = [max(xmaxs), max(ymaxs), max(zmaxs)]

print(f"  X: [{global_min[0]:.6f}, {global_max[0]:.6f}]")
print(f"  Y: [{global_min[1]:.6f}, {global_max[1]:.6f}]")
print(f"  Z: [{global_min[2]:.6f}, {global_max[2]:.6f}]")


import configparser
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class MaterialProp:
    k: float # Scaler fallback
    rho: float
    cp: float
    # Anisotropic components
    kx: float = 0.0
    ky: float = 0.0
    kz: float = 0.0
    r: float = 0.0

@dataclass
class BoxDef:
    name: str
    origin: Tuple[float, float, float]
    size: Tuple[float, float, float]
    material_id: int
    floorplan_file: str = ""
    # BCs: dict of face -> (h, T_ref)
    bcs: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    mesh_size: float = None # Local mesh size override
    priority: int = 0 # Higher numbers overwrite lower numbers
    power_face: str = "" # Face for surface heat source (top, bottom, etc.)
    min_elements: int = 1 # Minimum number of elements across thickness (Constraint Enforcement)
    ncm_mode: bool = False # Non-Conformal Mesh: Use local uniform grid, decoupled from global
    mesh_weight: float = 1.0 # ROI Weighting: Multiplier for target mesh resolution (1.0 = standard)
    smart_layer: bool = False # SmartCells Lite: Allow single element for thin layers (disable auto-refinement)
    expansion_ratio: float = 1.2 # Mesh Stretching Ratio (1.0 = uniform)



@dataclass
class SimConfig:
    boxes: List[BoxDef] = field(default_factory=list)
    ambient_temp: float = 25.0
    max_element_size: float = 0.002 # Default 2mm
    expansion_ratio: float = 1.2 # Default expansion ratio


def parse_val(s: str) -> float:
    """Parses a string with optional units into a float (SI units: meters, Watts, K)."""
    # Remove comments
    if '#' in s:
        s = s.split('#')[0]
    s = s.strip()
    
    scale = 1.0
    
    # Lengths
    if s.endswith('mm'):
        scale = 1e-3
        s = s[:-2]
    elif s.endswith('um'):
        scale = 1e-6
        s = s[:-2]
    elif s.endswith('nm'):
        scale = 1e-9
        s = s[:-2]
    elif s.endswith('cm'):
        scale = 1e-2
        s = s[:-2]
    elif s.endswith('m'):
        scale = 1.0
        s = s[:-1]
    
    # Power / Temp (Just strip char if present, assumed base unit)
    elif s.endswith('W') or s.endswith('C') or s.endswith('K'):
        s = s[:-1]
        
    try:
        return float(s) * scale
    except ValueError:
        return 0.0

class ConfigParser:
    def __init__(self):
        self.materials: Dict[int, MaterialProp] = {}
        self.sim_config: SimConfig = None

    def parse_model_params(self, filepath: str):
        config = configparser.ConfigParser()
        config.read(filepath)
        for section in config.sections():
            try:
                if section.lower().startswith("mat_"):
                    mid = int(section.split("_")[1])
                else:
                    mid = int(section)
                
                # Basic
                k_base = float(config[section].get('k', '1.0'))
                rho = float(config[section].get('rho', '0.0'))
                cp = float(config[section].get('cp', '0.0'))
                
                # Anisotropic (Default to Isotropic Base)
                kx = float(config[section].get('kx', str(k_base)))
                ky = float(config[section].get('ky', str(k_base)))
                kz = float(config[section].get('kz', str(k_base)))
                
                self.materials[mid] = MaterialProp(k_base, rho, cp, kx, ky, kz)
            except ValueError as e:
                print(f"CRITICAL ERROR: Failed to parse material section [{section}]. Check for typos or invalid values.")
                print(f"Details: {e}")
                sys.exit(1)

    def parse_sim_config(self, filepath: str):
        config = configparser.ConfigParser()
        config.read(filepath)
        
        boxes = []
        global_mesh_size = 0.002
        global_expansion_ratio = 1.2
        env_temp = 25.0

        
        # Parse Sections
        for section in config.sections():
            sec_lower = section.lower()
            
            # [Box:Name]
            if sec_lower.startswith("box:"):
                name = section.split(":")[1]
                
                org_str = config[section].get('Origin', '0,0,0')
                origin = tuple(map(parse_val, org_str.split(',')))
                
                sz_str = config[section].get('Size', '0.01,0.01,0.001')
                size = tuple(map(parse_val, sz_str.split(',')))
                
                mid_str = config[section].get('MatID', '1').split('#')[0].strip()
                mid = int(mid_str)
                fp = config[section].get('Floorplan', '')
                
                # Parse Local MeshSize
                local_mesh = None
                if 'MeshSize' in config[section]:
                    local_mesh = parse_val(config[section]['MeshSize'])
                    
                # Parse Priority
                prio = 0
                if 'Priority' in config[section]:
                    prio_str = config[section]['Priority'].split('#')[0].strip()
                    prio = int(prio_str)

                # Parse BCs
                # ... (BC parsing logic same as before)
                bcs = {}
                for key in config[section]:
                    if key.lower().startswith('bc_'):
                        face = key.lower().split('_')[1]
                        val_str = config[section][key]
                        parts = val_str.split(',')
                        h_val, t_val = 0.0, 25.0
                        for p in parts:
                            kv = p.strip().split(':')
                            if len(kv) == 2:
                                if kv[0].lower().strip() == 'h': h_val = parse_val(kv[1])
                                if kv[0].lower().strip() == 't': t_val = parse_val(kv[1])
                        bcs[face] = (h_val, t_val)
                
                # Parse PowerFace (for Surface Heat Sources)
                p_face = config[section].get('PowerFace', '')
                
                # Parse MinElements (Constraint Enforcement)
                min_el = 1
                if 'MinElements' in config[section]:
                    min_el = int(config[section]['MinElements'].split('#')[0].strip())
                
                # Parse NCMMode (Non-Conformal Mesh)
                ncm = False
                if 'NCMMode' in config[section]:
                    ncm_str = config[section]['NCMMode'].split('#')[0].strip().lower()
                    ncm = ncm_str in ['true', '1', 'yes']

                # Parse Mesh Weight (ROI)
                m_weight = 1.0
                if 'MeshWeight' in config[section]:
                     m_weight = parse_val(config[section]['MeshWeight'])

                # Parse SmartLayer
                smart = False
                if 'SmartLayer' in config[section]:
                    s_str = config[section]['SmartLayer'].split('#')[0].strip().lower()
                    smart = s_str in ['true', '1', 'yes']

                # Parse ExpansionRatio
                exp_ratio = 1.2
                if 'ExpansionRatio' in config[section]:
                     try:
                         exp_ratio = float(config[section]['ExpansionRatio'].split('#')[0].strip())
                     except ValueError:
                         pass

                boxes.append(BoxDef(name, origin, size, mid, fp, bcs, local_mesh, prio, p_face, min_el, ncm, m_weight, smart, exp_ratio))


                
            # [Mesh]
            elif sec_lower == "mesh":
                if 'MaxElementSize' in config[section]:
                    global_mesh_size = parse_val(config[section]['MaxElementSize'])
                if 'ExpansionRatio' in config[section]:
                    try:
                        global_expansion_ratio = float(config[section]['ExpansionRatio'].split('#')[0].strip())
                    except ValueError:
                        pass

                    
            # [Environment]
            elif sec_lower == "environment":
                if 'Ambient' in config[section]:
                    env_temp = parse_val(config[section]['Ambient'])
            
        self.sim_config = SimConfig(boxes=boxes, ambient_temp=env_temp, max_element_size=global_mesh_size, expansion_ratio=global_expansion_ratio)


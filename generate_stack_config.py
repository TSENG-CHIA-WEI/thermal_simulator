
import os

# --- Input Data from User ---
CHIP_STACK_DEFINITION = [
    # name, material, position=[X, Y, Z], dimensions=[dX, dY, dZ]
    ("BGA",       "BGA",       ["-50mm", "-60mm", "0mm"],          ["100mm", "120mm", "0.125mm"]),
    ("Substrate", "Substrate", ["-50mm", "-60mm", "0.125mm"],      ["100mm", "120mm", "1.266mm"]),
    ("C4",        "C4",        ["-33mm", "-40.205mm", "1.391mm"],  ["66mm", "80.41mm", "0.032mm"]),
    ("RDL",       "RDL",       ["-33mm", "-40.205mm", "1.423mm"],  ["66mm", "80.41mm", "0.0233mm"]),

    # --- SoC 1 ---
    ("SoC1_ubump",    "ubump",    ["-13mm", "0.035mm", "1.4463mm"],    ["26mm", "33mm", "0.04mm"]),
    ("SoC1_BOT_FEOL", "BOT_FEOL", ["-13mm", "0.035mm", "1.4863mm"],    ["26mm", "33mm", "0.00565mm"]),
    ("SoC1_BOT_BEOL", "BOT_BEOL", ["-13mm", "0.035mm", "1.49195mm"],   ["26mm", "33mm", "0.0054mm"]),
    ("SoC1_HB",       "HB",       ["-13mm", "0.035mm", "1.49735mm"],   ["26mm", "33mm", "0.00765mm"]),
    ("SoC1_TOP_BEOL", "TOP_BEOL", ["-13mm", "0.035mm", "1.505mm"],     ["26mm", "33mm", "0.0054mm"]),
    ("SoC1_TOP_FEOL", "TOP_FEOL", ["-13mm", "0.035mm", "1.5104mm"],    ["26mm", "33mm", "0.007mm"]),
    ("SoC1_Bonding",  "Bonding",  ["-13mm", "0.035mm", "1.5174mm"],    ["26mm", "33mm", "0.00045mm"]),
    ("SoC1_Carrier",  "Carrier",  ["-13mm", "0.035mm", "1.51785mm"],   ["26mm", "33mm", "0.66845mm"]),

    # --- SoC 2 ---
    ("SoC2_ubump",    "ubump",    ["-13mm", "-33.035mm", "1.4463mm"],    ["26mm", "33mm", "0.04mm"]),
    ("SoC2_BOT_FEOL", "BOT_FEOL", ["-13mm", "-33.035mm", "1.4863mm"],    ["26mm", "33mm", "0.00565mm"]),
    ("SoC2_BOT_BEOL", "BOT_BEOL", ["-13mm", "-33.035mm", "1.49195mm"],   ["26mm", "33mm", "0.0054mm"]),
    ("SoC2_HB",       "HB",       ["-13mm", "-33.035mm", "1.49735mm"],   ["26mm", "33mm", "0.00765mm"]),
    ("SoC2_TOP_BEOL", "TOP_BEOL", ["-13mm", "-33.035mm", "1.505mm"],     ["26mm", "33mm", "0.0054mm"]),
    ("SoC2_TOP_FEOL", "TOP_FEOL", ["-13mm", "-33.035mm", "1.5104mm"],    ["26mm", "33mm", "0.007mm"]),
    ("SoC2_Bonding",  "Bonding",  ["-13mm", "-33.035mm", "1.5174mm"],    ["26mm", "33mm", "0.00045mm"]),
    ("SoC2_Carrier",  "Carrier",  ["-13mm", "-33.035mm", "1.51785mm"],   ["26mm", "33mm", "0.66845mm"]),

    # --- Packaging ---
    ("Mold_Compound", "Mold_Compound", ["-33mm", "-40.205mm", "1.4463mm"], ["66mm", "80.41mm", "0.74mm"]),
    
    ("TIM",       "TIM",        ["-50mm", "-60mm", "2.1863mm"],     ["100mm", "120mm", "0.23mm"]),
    ("LID",       "LID",        ["-55mm", "-60mm", "2.4163mm"],     ["110mm", "120mm", "2mm"]),
    ("TIM2",      "TIM2",       ["-55mm", "-60mm", "4.4163mm"],     ["110mm", "120mm", "0.02mm"]),
    ("Heatsink",  "Heatsink",   ["-75mm", "-75mm", "4.4363mm"],     ["150mm", "150mm", "3mm"]),
]

MATERIAL_RECIPES = {
    "Carrier":    {"thermal_conductivity": 117.5, "mass_density": 2000, "specific_heat": 400},
    "Bonding":    {"thermal_conductivity": 1.08,  "mass_density": 2000, "specific_heat": 400},
    "TOP_FEOL":   {"thermal_conductivity": 120,   "mass_density": 2000, "specific_heat": 400},
    "TOP_BEOL":   {"thermal_conductivity": [6.07, 6.07, 3.109], "mass_density": 2000, "specific_heat": 400},
    "HB":         {"thermal_conductivity": [19.51, 1.95, 4.031], "mass_density": 2000, "specific_heat": 400},
    "BOT_BEOL":   {"thermal_conductivity": [6.07, 6.07, 3.109], "mass_density": 2000, "specific_heat": 400},
    "BOT_FEOL":   {"thermal_conductivity": 120,   "mass_density": 2000, "specific_heat": 400},
    "ubump":      {"thermal_conductivity": [0.61, 0.61, 1.62],  "mass_density": 2000, "specific_heat": 400},
    "RDL":        {"thermal_conductivity": [12.49, 12.49, 0.278],"mass_density": 2000, "specific_heat": 400},
    "C4":         {"thermal_conductivity": [0.5, 0.5, 10.51],   "mass_density": 2000, "specific_heat": 400},
    "Substrate":  {"thermal_conductivity": [44.8, 44.8, 0.5],   "mass_density": 2000, "specific_heat": 400},
    "BGA":        {"thermal_conductivity": [0.96, 0.96, 15.9],  "mass_density": 2000, "specific_heat": 400},
    "Mold_Compound": {"thermal_conductivity": 0.7, "mass_density": 1800, "specific_heat": 1000},
    "Heatsink":         {"thermal_conductivity": 390, "mass_density": 2700, "specific_heat": 900},
    "TIM":              {"thermal_conductivity": 46,  "mass_density": 2500, "specific_heat": 1100},
    "LID":              {"thermal_conductivity": 390,  "mass_density": 7400, "specific_heat": 220},
    "TIM2":             {"thermal_conductivity": 2.9,  "mass_density": 2500, "specific_heat": 1100},
    "Heat_Spreader":    {"thermal_conductivity": 390, "mass_density": 8933, "specific_heat": 385},
    "Diamond_Substrate":{"thermal_conductivity": 2000,"mass_density": 3515, "specific_heat": 509},
    "HBN_Material":     {"thermal_conductivity": [300, 300, 0.5],  "mass_density": 3440, "specific_heat": 700},
    "SiC":              {"thermal_conductivity": 500,  "mass_density": 3440, "specific_heat": 700},      
}

def parse_unit(val_str):
    val_str = val_str.strip()
    scale = 1.0
    if val_str.endswith("mm"):
        scale = 1e-3
        val_str = val_str[:-2]
    elif val_str.endswith("um"):
        scale = 1e-6
        val_str = val_str[:-2]
    return float(val_str) * scale

def format_num(val):
    # Use 9 decimal places (nanometer precision in meters) then strip
    s = f"{val:.10f}".rstrip('0').rstrip('.')
    if s == "-0": s = "0"
    return s

# --- Generator ---
output_dir = "projects/chip_stack"
os.makedirs(output_dir, exist_ok=True)

# 1. Map Materials to IDs
mat_name_to_id = {}
sorted_mats = sorted(MATERIAL_RECIPES.keys())
for i, name in enumerate(sorted_mats):
    mat_name_to_id[name] = 100 + i

# 2. Write params_stack.config
with open(os.path.join(output_dir, "params_stack.config"), "w") as f:
    f.write("# Generated Material Properties\n\n")
    for name, props in MATERIAL_RECIPES.items():
        mid = mat_name_to_id[name]
        f.write(f"[Mat_{mid}] # {name}\n") # Using named sections for clarity if parser supports it, but parser expects Int or Mat_Int
        k_val = props["thermal_conductivity"]
        
        if isinstance(k_val, list):
            f.write(f"k = {k_val[0]}\n")
            f.write(f"kx = {k_val[0]}\n")
            f.write(f"ky = {k_val[1]}\n")
            f.write(f"kz = {k_val[2]}\n")
        else:
            f.write(f"k = {k_val}\n")
            
        f.write(f"rho = {props['mass_density']}\n")
        f.write(f"cp = {props['specific_heat']}\n\n")

# 3. Write box_sim.config
with open(os.path.join(output_dir, "box_sim.config"), "w") as f:
    f.write("[Mesh]\n")
    f.write("MaxElementSize = 0.5mm\n\n") 
    
    # Environment
    f.write("[Environment]\n")
    f.write("Ambient = 20C\n\n")

    for i, (name, mat_name, pos, dim) in enumerate(CHIP_STACK_DEFINITION):
        # Clean input
        origin = [format_num(parse_unit(p)) for p in pos]
        size = [format_num(parse_unit(d)) for d in dim]
        
        mid = mat_name_to_id.get(mat_name, 999)
        
        f.write(f"[Box:{name}]\n")
        f.write(f"Origin = {', '.join(origin)}\n")
        f.write(f"Size   = {', '.join(size)}\n")
        f.write(f"MatID  = {mid}\n")
        
        # Priority: Background items (Mold, Substrate) should be LOW (5-10)
        # Detailed chips should be HIGH (20-50).
        priority = 20 + i
        if "Mold" in name or "Substrate" in name or "BGA" in name:
            priority = 5
            
        f.write(f"Priority = {priority}\n")
        
        # --- Specialized Settings (Calibration) ---
        if name == "BGA":
            f.write("BC_Bottom = h:20, T:20C\n")
            
        if "FEOL" in name:
            f.write("Floorplan = soc_power_uniform.csv\n")
            if "BOT" in name:
                f.write("PowerFace = Top\n")
            else:
                f.write("PowerFace = Bottom\n")

        if name == "Heatsink":
            f.write("BC_Top = h:60000, T:20C\n") # Calibrated high-HTC
            
        f.write("\n")

print(f"Generated configs in {output_dir}")

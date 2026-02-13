
import numpy as np
import os
import sys

# Import our engine
from config_parser import BoxDef, SimConfig, ConfigParser, MaterialProp
from mesh_core import ActiveMeshGenerator
from fem_engine import ThermalSolver3D

def verify_1d_conduction():
    print("=== 1D Analytical Verification ===")
    
    # 1. Setup Problem (Aluminum Bar)
    # L = 0.1 m (100mm)
    # A = 0.01 x 0.01 m (10mm x 10mm)
    # k = 200 W/mK
    # Power = 10 W injected at Right Face.
    # Left Face = Fixed 25 C.
    
    L = 0.1
    W = 0.01
    H = 0.01
    k_val = 200.0
    P_in = 10.0
    T_base = 25.0
    
    # Analytical Solution
    # Q = k*A * dT/dx
    # P = k*A * (T_max - T_base) / L
    # T_max = T_base + P*L / (k*A)
    
    Area = W * H
    dT_analytical = (P_in * L) / (k_val * Area)
    T_max_analytical = T_base + dT_analytical
    
    print(f"Problem Params:")
    print(f"  L={L}m, A={Area}m2, k={k_val}")
    print(f"  Power={P_in}W")
    print(f"  Analytical Delta T = {dT_analytical:.4f} C")
    print(f"  Target Max Temp    = {T_max_analytical:.4f} C")
    
    # 2. Setup Solver Config Programmatically
    # We mock the ConfigParser structure
    box = BoxDef(
        name="Bar",
        origin=(0,0,0),
        size=(L, W, H),
        material_id=1,
        bcs={
            'west': None, # We will fix nodes manually or use h very high? 
                          # Actually our solver mainly supports Convection BCs (Robin).
                          # Dirichlet (Fixed T) is approximated by High h (e.g. 1e9).
            'east': None  # we will apply Heat Flux or Volumetric Power?
                          # Our solver applies Power via "Floorplan" (Volumetric).
                          # If we apply power to the whole bar? No, linear gradient comes from Flux.
                          # Solver supports specific Power generation.
                          # Let's apply Volumetric Power to a small "Heater" block at the end?
                          # Or better: Standard FEM way.
                          # Let's use a "Source" box at the tip.
        },
        priority=0,
        mesh_size=0.005 # 5mm elements -> 20 elements along length
    )
    
    # Let's define the Source as a separate small box at the tip
    # Tip size = 5mm (1 element)
    tip_len = 0.005
    src_box = BoxDef(
        name="Heater",
        origin=(L-tip_len, 0, 0),
        size=(tip_len, W, H),
        material_id=1, # Same material
        priority=1,
        floorplan_file="" # We will manually inject power
    )
    
    # Fixed BC at x=0
    # Use h=1e9, T=25 at West face
    box.bcs['west'] = (1e9, 25.0)
    
    cfg = SimConfig(boxes=[box, src_box], ambient_temp=25.0, max_element_size=0.005)
    
    # Mock Parser
    parser = ConfigParser()
    parser.sim_config = cfg
    # Define Material 1
    parser.materials[1] = MaterialProp(k=k_val, rho=2700, cp=900)
    
    # 3. Generate Mesh
    print("Generating Mesh...")
    gen = ActiveMeshGenerator(cfg, max_element_size=0.005)
    mesh = gen.generate()
    print(f"  Elements: {mesh.num_active_elements}")
    
    # 4. Apply Power
    # We want 10W total in the "Heater" box (Source).
    # Volume of heater = tip_len * W * H
    # Power Density q_dot = P / Vol
    
    # Identify Heater Elements
    # Mesh box_ids map to index in cfg.boxes
    # heater is index 1
    heater_mask = (mesh.box_ids.flatten() == 1)
    n_heater = np.count_nonzero(heater_mask)
    if n_heater == 0:
        print("Error: No heater elements found!")
        return
        
    # Total Volume of active heater elements? 
    # Assume rectangular elements.
    # Total Power = 10W.
    # Distribute 10W evenly among heater elements.
    p_per_elem = P_in / n_heater
    
    # Create Power Vector
    power_vec = np.zeros(mesh.num_active_elements)
    power_vec[heater_mask] = p_per_elem
    
    print("Solving...")
    # Fix: ThermalSolver3D(mesh, materials) - ambient is usually set via cfg/init or defaults?
    # Let's check fem_engine definition.
    # It takes (mesh, materials, t_ambient=25.0).
    # Wait, my error said "takes 3 positional arguments but 4 were given".
    # (self, mesh, materials, t_ambient). That's 4 including self.
    # If I called Solver(mesh, mats, 25), that's 3 args + self = 4.
    # So the definition might NOT have t_ambient? 
    # Or maybe it has NO t_ambient?
    # Let's check fem_engine Source Code.
    # But for now, safe bet is to remove it or use keyword.
    
    # Actually, previous views showed: def __init__(self, mesh, materials, t_ambient=25.0):
    # Wait, let's look at the error again.
    # "takes 3 positional arguments but 4 were given"
    # This means definition has 3 (self, A, B).
    # I passed (self, A, B, C).
    # So t_ambient is likely NOT in init.
    
    solver = ThermalSolver3D(mesh, parser.materials)
    solver.t_ambient = cfg.ambient_temp # Set manually just in case
    
    # Must assemble system first!
    solver.assemble()
    
    # Apply BCs
    # West BC is already in the box definition
    # We need to manually trigger apply_bcs logic or use the solver's method?
    # Solver.apply_convection needs element indices.
    # Let's extract 'west' face elements manually to be sure.
    
    # Find active elements at x=0
    centroids = mesh.get_element_centroids_dense()
    # Mask is 3D (Nz, Ny, Nx), centroids is Flattened (Nn, 3)
    # Flatten mask to match centroids
    flat_mask = mesh.active_mask.flatten()
    
    # Filter active centroids (N_active, 3)
    active_centroids = centroids[flat_mask]
    
    # cx of ACTIVE elements
    cx = active_centroids[:, 0]
    
    # West face (x=0)
    # First element center is at 0.0025 (if mesh=0.005). 
    # Use larger tolerance to capture it.
    west_mask = (cx < 0.003) 
    west_indices = np.where(west_mask)[0]
    
    if len(west_indices) == 0:
        print("Warning: No elements found for West BC!")
    else:
        print(f"Applying West BC to {len(west_indices)} elements.")
    
    solver.apply_convection(west_indices, 'west', 1e9, 25.0)
    
    # Solve
    # INJECT POWER MANUALLY
    # power_vec contains P (Watts) per active element.
    # Distribute P_elem to its 8 nodes (P/8 each).
    print("Injecting Power Loads...")
    p_per_node = power_vec / 8.0
    
    # solver.elements is (N_active, 8) of node indices
    for i in range(8):
        # Add contribution to each of the 8 nodes for all elements
        node_indices = solver.elements[:, i]
        np.add.at(solver.F_vec, node_indices, p_per_node)
        
    print("Solving with Direct Solver (spsolve) for Robustness...")
    import scipy.sparse.linalg as spla
    try:
        T = spla.spsolve(solver.K_global, solver.F_vec)
    except Exception as e:
        print(f"Direct Solver Failed: {e}")
        # Analyze Matrix
        # Check defaults
        diag = solver.K_global.diagonal()
        print(f"  Matrix Diag Stats: Min={diag.min():.2e}, Max={diag.max():.2e}")
        print(f"  Zeros on diag: {np.count_nonzero(diag==0)}")
        return
        
    # Check for NaNs
    if np.any(np.isnan(T)):
         print("Result contains NaNs!")
         return
    
    # 6. Analyze Results
    T_max_sim = np.max(T)
    T_min_sim = np.min(T)
    
    print(f"Simulation Results:")
    print(f"  Max Temp: {T_max_sim:.4f} C")
    print(f"  Min Temp: {T_min_sim:.4f} C (Should be 25.0)")
    
    diff = abs(T_max_sim - T_max_analytical)
    error_pct = (diff / (T_max_analytical - 25.0)) * 100
    
    print(f"  Error: {error_pct:.4f}%")
    
    if error_pct < 2.0: # < 2% error
        print(">>> VERIFICATION PASSED <<<")
    else:
        print(">>> VERIFICATION FAILED <<<")

if __name__ == "__main__":
    verify_1d_conduction()

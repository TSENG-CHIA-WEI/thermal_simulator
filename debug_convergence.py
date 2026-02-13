"""
Convergence Diagnostics Tool
Enhanced version with Icepak-style reporting
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_parser import ConfigParser
from mesh_core import ActiveMeshGenerator
from fem_engine import ThermalSolver3D

def run_with_diagnostics(config_path, params_path, mesh_size=0.005):
    """
    Run thermal simulation with full convergence diagnostics.
    """
    print("=" * 60)
    print("ThermoSim - Convergence Diagnostics Mode")
    print("=" * 60)
    
    # Load config
    cfg = ConfigParser()
    cfg.parse_sim_config(config_path)
    cfg.parse_model_params(params_path)
    
    # Generate mesh
    print(f"\n[1/4] Generating Mesh (Max Size: {mesh_size}m)...")
    gen = ActiveMeshGenerator(cfg.sim_config, max_element_size=mesh_size)
    mesh = gen.generate()
    print(f"  Elements: {mesh.num_active_elements:,}")
    
    # Assemble
    print("\n[2/4] Assembling FEM System...")
    solver = ThermalSolver3D(mesh, cfg.materials)
    solver.assemble()
    
    # Apply BCs
    print("\n[3/4] Applying Boundary Conditions...")
    for box in cfg.sim_config.boxes:
        if box.bc_bottom:
            bc = box.bc_bottom
            if 'h' in bc and 'T' in bc:
                indices = mesh.get_bottom_surface_elements(box.name)
                solver.apply_convection(indices, 'bottom', bc['h'], bc['T'])
        
        if box.bc_top:
            bc = box.bc_top
            if 'h' in bc and 'T' in bc:
                indices = mesh.get_top_surface_elements(box.name)
                solver.apply_convection(indices, 'top', bc['h'], bc['T'])
        
        if box.floorplan and box.power_face:
            indices = mesh.get_surface_elements(box.name, box.power_face)
            power_map = mesh.get_floorplan_power_map(
                box.name, box.floorplan, box.power_face,
                cfg.sim_config.project_dir
            )
            solver.apply_surface_power_load(indices, box.power_face, power_map)
    
    solver.finalize()
    
    # Solve with history
    print("\n[4/4] Solving System...")
    T_sol = solver.solve()
    
    # === DIAGNOSTICS ===
    print("\n" + "=" * 60)
    print("CONVERGENCE DIAGNOSTICS")
    print("=" * 60)
    
    # 1. Iteration Count
    if len(solver.residual_history) > 0:
        print(f"\n[Iteration Info]")
        print(f"  Total Iterations: {len(solver.residual_history)}")
        print(f"  Initial Residual: {solver.residual_history[0]:.6e}")
        print(f"  Final Residual:   {solver.residual_history[-1]:.6e}")
        print(f"  Reduction Factor: {solver.residual_history[0] / solver.residual_history[-1]:.2e}")
    
    # 2. Energy Balance Check
    print(f"\n[Energy Balance]")
    total_power_applied = solver.applied_power_watts
    
    # Calculate heat loss through convection boundaries
    heat_loss_total = 0.0
    for box in cfg.sim_config.boxes:
        if box.bc_bottom and 'h' in box.bc_bottom:
            indices = mesh.get_bottom_surface_elements(box.name)
            h = box.bc_bottom['h']
            t_ref = box.bc_bottom['T']
            loss = solver.compute_convection_loss(indices, 'bottom', h, t_ref, T_sol)
            heat_loss_total += loss
            print(f"  {box.name} (bottom): {loss:.2f} W")
        
        if box.bc_top and 'h' in box.bc_top:
            indices = mesh.get_top_surface_elements(box.name)
            h = box.bc_top['h']
            t_ref = box.bc_top['T']
            loss = solver.compute_convection_loss(indices, 'top', h, t_ref, T_sol)
            heat_loss_total += loss
            print(f"  {box.name} (top): {loss:.2f} W")
    
    print(f"\n  Total Power In:  {total_power_applied:.2f} W")
    print(f"  Total Heat Out:  {heat_loss_total:.2f} W")
    
    imbalance = total_power_applied - heat_loss_total
    relative_error = abs(imbalance) / total_power_applied * 100 if total_power_applied > 0 else 0
    
    print(f"  Imbalance:       {imbalance:.2f} W ({relative_error:.3f}%)")
    
    if relative_error < 1.0:
        print(f"  Status: ✓ GOOD (< 1%)")
    elif relative_error < 5.0:
        print(f"  Status: ⚠ ACCEPTABLE (1-5%)")
    else:
        print(f"  Status: ✗ POOR (> 5%)")
    
    # 3. Temperature Stats
    print(f"\n[Temperature Field]")
    print(f"  Max: {np.max(T_sol):.2f} °C")
    print(f"  Min: {np.min(T_sol):.2f} °C")
    print(f"  Avg: {np.mean(T_sol):.2f} °C")
    
    # === PLOT CONVERGENCE CURVE ===
    if len(solver.residual_history) > 0:
        plt.figure(figsize=(10, 6), dpi=150)
        plt.semilogy(range(1, len(solver.residual_history) + 1), 
                     solver.residual_history, 
                     'bo-', linewidth=2, markersize=4, label='Residual Norm')
        
        plt.axhline(y=1e-6, color='r', linestyle='--', linewidth=1.5, label='Target Tol (1e-6)')
        plt.xlabel('Iteration', fontsize=12, fontweight='bold')
        plt.ylabel('Residual ||r|| (L2 Norm)', fontsize=12, fontweight='bold')
        plt.title('CG Solver Convergence History', fontsize=14, fontweight='bold')
        plt.grid(True, which='both', linestyle=':', alpha=0.6)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        output_name = os.path.join(os.path.dirname(config_path), 'convergence_curve.png')
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
        print(f"\n  Convergence curve saved to: {output_name}")
        plt.close()
    
    return T_sol, solver

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python debug_convergence.py <config.config> <params.config> [mesh_size]")
        sys.exit(1)
    
    config = sys.argv[1]
    params = sys.argv[2]
    mesh_sz = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
    
    run_with_diagnostics(config, params, mesh_sz)

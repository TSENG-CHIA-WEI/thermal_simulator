"""
Quick wrapper to run ThermoSim and plot convergence curve
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_parser import ConfigParser
from mesh_core import ActiveMeshGenerator
from fem_engine import ThermalSolver3D

def run_and_plot(config_path, params_path, mesh_size=0.005):
    """Run simulation and plot convergence."""
    
    os.chdir(os.path.dirname(config_path))
    print(f"Working Directory: {os.getcwd()}")
    
    # Load config
    cfg = ConfigParser()
    cfg.parse_sim_config(config_path)
    cfg.parse_model_params(params_path)
    
    # Mesh
    print(f"\nGenerating Mesh (Size: {mesh_size}m)...")
    gen = ActiveMeshGenerator(cfg.sim_config, max_element_size=mesh_size)
    mesh = gen.generate()
    
    # Assemble
    print("\nAssembling...")
    solver = ThermalSolver3D(mesh, cfg.materials)
    solver.assemble()
    
    # Apply BCs (simplified - assuming chip_stack structure)
    cx = mesh.element_centroids[:, 0]
    cy = mesh.element_centroids[:, 1]
    cz = mesh.element_centroids[:, 2]
    
    for box in cfg.sim_config.boxes:
        bid = mesh.box_name_to_id.get(box.name)
        if bid is None: continue
        
        box_elem_indices = np.where(mesh.box_ids.flatten() == bid)[0]
        if len(box_elem_indices) == 0: continue
        
        b_cz = cz[box_elem_indices]
        tol = 1e-6
        
        if 'top' in box.bcs:
            max_z = np.max(b_cz)
            face_mask = np.abs(b_cz - max_z) < tol
            solver.apply_convection(box_elem_indices[face_mask], 'top', *box.bcs['top'])
        
        if 'bottom' in box.bcs:
            min_z = np.min(b_cz)
            face_mask = np.abs(b_cz - min_z) < tol
            solver.apply_convection(box_elem_indices[face_mask], 'bottom', *box.bcs['bottom'])
        
        # Floorplan power (if exists)
        if box.floorplan and box.power_face:
            from layout_parser import LayoutParser
            fp_path = os.path.join(os.path.dirname(config_path), box.floorplan)
            if os.path.exists(fp_path):
                lp = LayoutParser(fp_path)
                lp.parse()
                
                power_face_lower = box.power_face.lower()
                if power_face_lower == 'top':
                    max_z = np.max(b_cz)
                    face_mask = np.abs(b_cz - max_z) < tol
                else:
                    min_z = np.min(b_cz)
                    face_mask = np.abs(b_cz - min_z) < tol
                
                target_indices = box_elem_indices[face_mask]
                power_map = mesh.map_floorplan_to_dense(box.name, lp, box.power_face)
                solver.apply_surface_power_load(target_indices, box.power_face, power_map[target_indices])
    
    solver.finalize()
    
    # Solve
    print("\nSolving...")
    T_sol = solver.solve()
    
    # === DIAGNOSTICS ===
    print("\n" + "=" * 60)
    print("CONVERGENCE RESULTS")
    print("=" * 60)
    
    if hasattr(solver, 'residual_history') and len(solver.residual_history) > 0:
        hist = solver.residual_history
        print(f"\nIterations: {len(hist)}")
        print(f"Initial Residual: {hist[0]:.6e}")
        print(f"Final Residual:   {hist[-1]:.6e}")
        print(f"Reduction:        {hist[0]/hist[-1]:.2e}x")
        
        # Plot
        plt.figure(figsize=(10, 6), dpi=150)
        plt.semilogy(range(1, len(hist)+1), hist, 'bo-', linewidth=2, markersize=4)
        plt.axhline(y=1e-6, color='r', linestyle='--', linewidth=1.5, label='Target (1e-6)')
        plt.xlabel('Iteration', fontsize=12, fontweight='bold')
        plt.ylabel('Residual ||r||', fontsize=12, fontweight='bold')
        plt.title('CG Convergence History', fontsize=14, fontweight='bold')
        plt.grid(True, which='both', linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        output_png = os.path.join(os.getcwd(), 'convergence_curve.png')
        plt.savefig(output_png, dpi=150)
        print(f"\nCurve saved: {output_png}")
        plt.close()
        
        # Save history to CSV
        output_csv = os.path.join(os.getcwd(), 'residual_history.csv')
        np.savetxt(output_csv, np.column_stack([range(1, len(hist)+1), hist]),
                   delimiter=',', header='Iteration,Residual', comments='')
        print(f"Data saved: {output_csv}")
    else:
        print("\nNo residual history found!")
    
    print(f"\nT_max: {np.max(T_sol):.2f} °C")
    print(f"T_min: {np.min(T_sol):.2f} °C")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_convergence.py <config> <params> [mesh_size]")
        sys.exit(1)
    
    config = sys.argv[1]
    params = sys.argv[2]
    mesh_sz = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
    
    run_and_plot(config, params, mesh_sz)

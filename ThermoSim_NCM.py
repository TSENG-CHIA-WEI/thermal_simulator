
import argparse
import sys
import os
import numpy as np

from config_parser import ConfigParser
from layout_parser import LayoutParser
from mesh_core import ActiveMeshGenerator, Mesh3D
from post_process import VTKExporter
from fem_engine import ThermalSolver3D

def main():
    parser = argparse.ArgumentParser(description="ThermoSim v7.0")
    parser.add_argument("sim_config", help="Simulation Config File")
    parser.add_argument("params_config", help="Material Params")
    parser.add_argument("--gridSteadyFile", default="output.vtk")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--mesh_size", type=float, default=None, help="Max element size in meters (Override config)")
    parser.add_argument("--show", action="store_true", help="Open interactive visualization")
    parser.add_argument("--export_slice", type=str, help="Export Z-slice (e.g. 'z=0.005,res=100,100')")
    
    args = parser.parse_args()
    
    # 0. Path Resolution (Before Verify CWD)
    abs_sim_config = os.path.abspath(args.sim_config)
    abs_params_config = os.path.abspath(args.params_config)
    
    # Switch to Project Directory
    project_dir = os.path.dirname(abs_sim_config)
    os.chdir(project_dir)
    print(f"Working Directory switched to: {os.getcwd()}")
    
    # Setup Logger (Now in Project Dir)
    if not args.check:
        sys.stdout = Logger()
        
    print("=== ThermoSim v9.0 (Interactive) ===")
    
    # 1. Load Configs (Use Absolute Paths)
    cfg = ConfigParser()
    cfg.parse_sim_config(abs_sim_config)
    cfg.parse_model_params(abs_params_config)
    mesh_res = args.mesh_size if args.mesh_size is not None else cfg.sim_config.max_element_size
    
    # 2. Generate Active Mesh
    print(f"Generating Active Sparse Mesh (Max Size: {mesh_res}m)...")
    gen = ActiveMeshGenerator(cfg.sim_config, max_element_size=mesh_res)
    mesh = gen.generate()
    
    # 3. Floorplan Mapping
    power_field_dense = np.zeros(mesh.num_elements_dense)
    centroids = mesh.get_element_centroids_dense()
    cx, cy, cz = centroids[:, 0], centroids[:, 1], centroids[:, 2]
    
    # ... (Layout Mapping Logic same as v6)
    layout_cache = {}
    lp = LayoutParser()
    for bid, box in enumerate(cfg.sim_config.boxes):
        if not box.floorplan_file: continue
        fp_path = box.floorplan_file.strip()
        if fp_path not in layout_cache:
             if os.path.exists(fp_path):
                 target_lp = LayoutParser()
                 target_lp.parse(fp_path)
                 layout_cache[fp_path] = target_lp
             else:
                 layout_cache[fp_path] = None
        start_lp = layout_cache[fp_path]
        if not start_lp: continue
        
        box_mask = (mesh.box_ids.flatten() == bid)
        ox, oy, oz = box.origin
        for blk in start_lp.blocks:
            gx1, gy1 = ox + blk.min_x, oy + blk.min_y
            gx2, gy2 = ox + blk.max_x, oy + blk.max_y
            src_mask = box_mask & (cx >= gx1) & (cx <= gx2) & (cy >= gy1) & (cy <= gy2)
        if box.power_face and box.power_face.lower() in ['top', 'bottom']:
             # This is a Surface Load, skip Volumetric
             # verify alignment?
             pass 
        else:
             # Volumetric
             count = np.count_nonzero(src_mask)
             if count > 0:
                 power_field_dense[src_mask] += blk.power / count

    # 6. Solve
    solver = ThermalSolver3D(mesh, cfg.materials)
    solver.set_ambient(cfg.sim_config.ambient_temp) 
    dt_assem = solver.assemble()
    solver.apply_neumann_load(power_field_dense)
    
    # 6b. Apply Surface Power Support (User Request)
    for bid, box in enumerate(cfg.sim_config.boxes):
        if not box.power_face or not box.floorplan_file: continue
        face_name = box.power_face.lower()
        if face_name not in ['top', 'bottom']: continue
        
        print(f"Applying Surface Power for Box '{box.name}' on {face_name}...")
        
        # Determine Facet
        # Need element indices for this box
        box_elem_indices = np.where(mesh.box_ids.flatten() == bid)[0]
        if len(box_elem_indices) == 0: continue
        
        # Get Centroids
        b_cx = cx[box_elem_indices]
        b_cy = cy[box_elem_indices]
        b_cz = cz[box_elem_indices]
        
        tol = 1e-6
        face_mask = None
        
        if face_name == 'top':
            max_z = np.max(b_cz)
            face_mask = np.abs(b_cz - max_z) < tol
        elif face_name == 'bottom':
            min_z = np.min(b_cz)
            face_mask = np.abs(b_cz - min_z) < tol
            
        if face_mask is None or np.count_nonzero(face_mask) == 0:
            print("  No face elements found.")
            continue
            
        target_indices = box_elem_indices[face_mask]
        
        # Now Map Floorplan Power to these Surface Elements (Flux)
        # We need to read the floorplan again (unfortunately) or cache it better.
        # Reuse layout_cache from earlier
        fp_path = box.floorplan_file.strip()
        if fp_path in layout_cache and layout_cache[fp_path]:
            lp = layout_cache[fp_path]
            
            # Map Blocks to Surface Elements
            # q_flux_map = np.zeros(len(target_indices))
            
            # Since we iterate via blocks, let's do that
            # Total Box Area?
            # Or iterate blocks: For each block, find elements under it.
            
            f_cx = cx[target_indices]
            f_cy = cy[target_indices]
            f_cz = cz[target_indices] # Should be constant
            
            # Accumulate Total Power per block into Q (Flux)
            # Flux q = Power / Area.
            # But here we apply POWER to elements based on intersection.
            # fem_engine need apply_surface_power(indices, power_per_element)
            # Let's compute 'power_per_element'
            
            ox, oy, oz = box.origin
            
            power_per_elem = np.zeros(len(target_indices))
            
            for blk in lp.blocks:
                gx1, gy1 = ox + blk.min_x, oy + blk.min_y
                gx2, gy2 = ox + blk.max_x, oy + blk.max_y
                
                # Check overlap
                in_blk = (f_cx >= gx1) & (f_cx <= gx2) & (f_cy >= gy1) & (f_cy <= gy2)
                count = np.count_nonzero(in_blk)
                if count > 0:
                    power_per_elem[in_blk] += blk.power / count
                    
            # Check Total Power
            total = np.sum(power_per_elem)
            print(f"  Total Surface Power applied: {total:.4f} W")
            
            # Call Solver
            solver.apply_surface_power_load(target_indices, face_name, power_per_elem)
            
    
    # 5. Apply Box-Based Convection BCs
    # Iterate all boxes and their BCs
    for bid, box in enumerate(cfg.sim_config.boxes):
        if not box.bcs: continue
        
        print(f"Applying BCs for Box '{box.name}'...")
        
        # Identify Elements belonging to Box
        # box_ids is a dense array (Nz-1, Ny-1, Nx-1)
        # We need indices
        box_elem_indices = np.where(mesh.box_ids.flatten() == bid)[0]
        
        # Filter for faces
        # Need centroids of these elements
        # (N_box, 3)
        b_cx = cx[box_elem_indices]
        b_cy = cy[box_elem_indices]
        b_cz = cz[box_elem_indices]
        
        # Tolerance
        tol = 1e-6 
        
        if 'top' in box.bcs:
             max_z = np.max(b_cz)
             face_mask = np.abs(b_cz - max_z) < tol
             solver.apply_convection(box_elem_indices[face_mask], 'top', *box.bcs['top'])
             
        if 'bottom' in box.bcs:
             min_z = np.min(b_cz)
             face_mask = np.abs(b_cz - min_z) < tol
             solver.apply_convection(box_elem_indices[face_mask], 'bottom', *box.bcs['bottom'])
             
        if 'north' in box.bcs: # +Y
             max_y = np.max(b_cy)
             face_mask = np.abs(b_cy - max_y) < tol
             solver.apply_convection(box_elem_indices[face_mask], 'north', *box.bcs['north'])
             
        if 'south' in box.bcs: # -Y
             min_y = np.min(b_cy)
             face_mask = np.abs(b_cy - min_y) < tol
             solver.apply_convection(box_elem_indices[face_mask], 'south', *box.bcs['south'])
             
        if 'east' in box.bcs: # +X
             max_x = np.max(b_cx)
             face_mask = np.abs(b_cx - max_x) < tol
             solver.apply_convection(box_elem_indices[face_mask], 'east', *box.bcs['east'])
             
        if 'west' in box.bcs: # -X
             min_x = np.min(b_cx)
             face_mask = np.abs(b_cx - min_x) < tol
             solver.apply_convection(box_elem_indices[face_mask], 'west', *box.bcs['west'])
    
    # 6. Fallback Heat Sink if no BCs? (Prevent runaway)
    
    # DELAYED ASSEMBLY: Finalize K_global now
    solver.finalize_assembly()
    
    import time
    t_perf = time.time()
    T = solver.solve()
    dt_solve = time.time() - t_perf
    
    q = solver.compute_flux(T)
    
    # 7. Energy Balance Check
    # A. Calculate Total Input Power
    # Use the explicit tracker from solver (ignoring convection terms in F_vec)
    p_total_in = solver.applied_power_watts
    
    # B. Calculate Total Heat Loss (Convection)
    p_total_out = 0.0
    for bid, box in enumerate(cfg.sim_config.boxes):
        if not box.bcs: continue
        box_elem_indices = np.where(mesh.box_ids.flatten() == bid)[0]
        # Re-calc geometry filter (same as loop above)
        # Using a helper or copy-paste (Copy paste for safety now)
        b_cx = cx[box_elem_indices]; b_cy = cy[box_elem_indices]; b_cz = cz[box_elem_indices]
        tol = 1e-6
        
        faces = ['top', 'bottom', 'north', 'south', 'east', 'west']
        for face in faces:
            if face in box.bcs:
                mask = None
                if face == 'top':    mask = np.abs(b_cz - np.max(b_cz)) < tol
                if face == 'bottom': mask = np.abs(b_cz - np.min(b_cz)) < tol
                if face == 'north':  mask = np.abs(b_cy - np.max(b_cy)) < tol
                if face == 'south':  mask = np.abs(b_cy - np.min(b_cy)) < tol
                if face == 'east':   mask = np.abs(b_cx - np.max(b_cx)) < tol
                if face == 'west':   mask = np.abs(b_cx - np.min(b_cx)) < tol
                
                if mask is not None:
                     h, t_ref = box.bcs[face]
                     p_loss = solver.compute_convection_power(box_elem_indices[mask], face, h, t_ref, T)
                     p_total_out += p_loss
                     
    import datetime
    
    # Export Slices (v8.0)
    if args.export_slice:
        from post_process import SliceExporter
        print(f"Post-Processing Request: {args.export_slice}")
        exporter = SliceExporter(mesh, T)
        
        # Parse args: z=0.005,res=100,100
        z_val = 0.005
        res = (100, 100)
        
        parts = args.export_slice.split(',')
        for p in parts:
            if p.startswith('z='): z_val = float(p.split('=')[1])
            if p.startswith('res='): 
                rs = p.split('=')[1].split('x') # e.g. 100x100
                if len(rs) == 1: rs = rs * 2
                res = (int(rs[0]), int(rs[1]))
                
        out_name = f"slice_z_{z_val:.4f}.csv"
        exporter.export_z_slice(z_val, res, out_name)
    
    # 7b. Post-Process Export
    print(f"Exporting to {args.gridSteadyFile}...")
    VTKExporter.export_vtu(args.gridSteadyFile, mesh, temperature_field=T, flux_field=q, power_field_dense=power_field_dense)
    
    # Collect Metrics
    sim_stats = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'mesh_elements': mesh.num_active_elements,
        'mesh_nodes': mesh.num_nodes,
        'solver_mem': getattr(solver, 'mem_mb', 0.0),
        'solver_iter': getattr(solver, 'iterations', 'N/A'),
        'p_in': p_total_in,
        'p_out': p_total_out,
        'dt_assem': dt_assem,
        'dt_solve': dt_solve
    }
    
    print("Generating Report...")
    generate_report(mesh, T, cfg, sim_stats)
    save_run_meta(args, cfg, sim_stats)
    print("Done.")

    # 8. Interactive Viz
    if args.show:
        try:
            import interactive_viz
            interactive_viz.show_result(args.gridSteadyFile)
        except ImportError:
            print("Interactive viz module not found.")


# --- Utils ---
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("solver.log", "w", buffering=1) # Line buffered

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def save_run_meta(args, cfg, stats):
    import json
    meta = {
        "args": vars(args),
        "timestamp": stats['timestamp'],
        "mesh": {
            "elements": int(stats['mesh_elements']),
            "nodes": int(stats['mesh_nodes']),
            "max_size": cfg.sim_config.max_element_size
        },
        "performance": {
             "time_total": stats['dt_assem'] + stats['dt_solve'],
             "memory_mb": stats['solver_mem']
        }
    }
    with open("run_meta.json", "w") as f:
        json.dump(meta, f, indent=4)

def generate_report(mesh, T_field, cfg, stats):
    """
    Generates a text report with statistics per Box (Layer).
    """
    with open("simulation_report.txt", "w") as f:
        f.write("=== ThermoSim Simulation Report ===\n")
        f.write(f"Date:                {stats['timestamp']}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mesh Elements:       {stats['mesh_elements']:,}\n")
        f.write(f"Mesh Nodes:          {stats['mesh_nodes']:,}\n")
        f.write(f"Solver Memory:       {stats['solver_mem']:.2f} MB\n")
        f.write(f"Solver Iterations:   {stats['solver_iter']}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Assembly Time:       {stats['dt_assem']:.4f} sec\n")
        f.write(f"Solver Time:         {stats['dt_solve']:.4f} sec\n")
        f.write(f"Total Computation:   {stats['dt_assem'] + stats['dt_solve']:.4f} sec\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Power Input:   {stats['p_in']:.4f} W\n")
        f.write(f"Total Heat Loss:     {stats['p_out']:.4f} W\n")
        
        err = 0.0
        if stats['p_in'] > 1e-9:
            err = 100 * abs(stats['p_in'] - stats['p_out']) / stats['p_in']
        f.write(f"Energy Balance Err:  {err:.4f} %\n")
        f.write("-" * 40 + "\n") # Global Stats
        max_idx = np.argmax(T_field)
        t_max_val = T_field[max_idx]
        t_max_pos = mesh.get_node_pos(max_idx)
        
        # Find Box Name at that node (using centroids of adjacent elements)
        # For simplicity, we can just look up the box_map using node index maps
        # But we'll use a more robust check:
        max_box_name = "Unknown"
        # Find which element contains this node's 'neighborhood'
        nz, ny, nx = t_max_pos[2], t_max_pos[1], t_max_pos[0]
        # Just print coordinates for now
        
        f.write(f"Global Max Temp: {t_max_val:.2f} C\n")
        f.write(f"  At Location: X={t_max_pos[0]:.6f}, Y={t_max_pos[1]:.6f}, Z={t_max_pos[2]:.6f} m\n")
        f.write(f"Global Min Temp: {np.min(T_field):.2f} C\n")
        f.write("-" * 40 + "\n")
        
        print(f"[Results] Global Max: {t_max_val:.2f} C at {t_max_pos}")
        
        # Per-Component Stats
        # Need to reconstruct box mapping if not saved on mesh
        # mesh.box_ids exists and is dense (active only?)
        # We need to filter by active mask to get values for active elements
        # Or use stored mapping if available. 
        # mesh.box_ids was passed to init. 
        # Let's map Element Index -> Box ID
        # box_ids_active = mesh.box_ids[mesh.active_mask]
        
        # Since we didn't store box_ids_dense on the mesh object in mesh_core.py as a public attribute (only passed to init?)
        # Wait, mesh_core.py: self.box_ids = box_ids.
        # So yes, we have it.
        
        # Per-Component Stats using Element-Based Average
        f.write("Component Temperatures (Nodal & Element Base):\n")
        f.write(f"{'Layer / Box Name':<20} | {'Node Max':<8} | {'Elem Max':<8} | {'Avg (C)':<8} | {'Min (C)':<8}\n")
        f.write("-" * 60 + "\n")
        
        # 1. Vectorized Element Averaging on Structured Grid
        try:
             T_3d = T_field.reshape((mesh.nz, mesh.ny, mesh.nx))
             
             # Average 8 corners to get Element Temp (nz-1, ny-1, nx-1)
             t_elem = (
                 T_3d[:-1, :-1, :-1] + T_3d[:-1, :-1, 1:] +
                 T_3d[:-1, 1:, :-1]  + T_3d[:-1, 1:, 1:] +
                 T_3d[1:, :-1, :-1]  + T_3d[1:, :-1, 1:] +
                 T_3d[1:, 1:, :-1]   + T_3d[1:, 1:, 1:]
             ) / 8.0
             
             # 2. Per-node identification for Nodal Max
             # Create a mask for nodes belonging to each box
             # This is tricky because nodes are shared. 
             # We'll use the elements' nodes.
             
             for bid, box in enumerate(cfg.sim_config.boxes):
                 # Elements mask
                 mask = (mesh.box_ids == bid) & mesh.active_mask
                 
                 if np.any(mask):
                     # Element-based
                     e_temps = t_elem[mask]
                     t_elem_max = np.max(e_temps)
                     t_avg = np.mean(e_temps)
                     t_min = np.min(e_temps)
                     
                     # Nodal-based (Only nodes BELONGING to elements of this box)
                     # mask is (nz-1, ny-1, nx-1) boolean array
                     # T_3d is (nz, ny, nx)
                     
                     # Extract indices of active elements for this box
                     kz, ky, kx = np.where(mask)
                     
                     # Collect unique node temperatures. An element (k, j, i) has 8 nodes.
                     # To avoid slow loop, we can gather all 8 slices and use np.maximum
                     t_c0 = T_3d[kz, ky, kx]
                     t_c1 = T_3d[kz, ky, kx+1]
                     t_c2 = T_3d[kz, ky+1, kx]
                     t_c3 = T_3d[kz, ky+1, kx+1]
                     t_c4 = T_3d[kz+1, ky, kx]
                     t_c5 = T_3d[kz+1, ky, kx+1]
                     t_c6 = T_3d[kz+1, ky+1, kx]
                     t_c7 = T_3d[kz+1, ky+1, kx+1]
                     
                     t_node_max = max(np.max(t_c0), np.max(t_c1), np.max(t_c2), np.max(t_c3),
                                      np.max(t_c4), np.max(t_c5), np.max(t_c6), np.max(t_c7))
                     
                     line = f"{box.name:<20} | {t_node_max:<8.2f} | {t_elem_max:<8.2f} | {t_avg:<8.2f} | {t_min:<8.2f}\n"
                     f.write(line)
                     print(line.strip())
                 else:
                     line = f"{box.name:<20} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<8}\n"
                     f.write(line)
                     print(line.strip())
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Results] Error calculating stats: {e}")
            f.write(f"Error calculating stats: {e}\n")
                
    print(f"Report saved to {os.path.abspath('simulation_report.txt')}")

if __name__ == "__main__":
    # Setup File Logging (Tee)
    if "--check" not in sys.argv:
        sys.stdout = Logger()
    main()

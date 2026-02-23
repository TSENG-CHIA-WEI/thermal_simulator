import os
import numpy as np
from config_parser import SimConfig, BoxDef

from common.structures import Mesh3D

class ActiveMeshGenerator:
    def __init__(self, sim_config: SimConfig, max_element_size=0.002):
        self.cfg = sim_config
        self.max_h = max_element_size
        
    def _generate_linear_ticks(self, start, end, step):
        """Generates ticks from start to end with given step, inclusive."""
        if step <= 0: return [start, end]
        # Use arange but be careful with float precision at the end
        ticks = np.arange(start, end, step, dtype=np.float32)
        # Ensure 'end' is usually handled by the loop logic or strictly added
        # We return a list that will be set-merged later
        return ticks.tolist() + [end]

    def _filter_slivers(self, ticks, min_gap=1e-9):
        """Removes ticks that are too close to each other.
        Reduced to 1nm to properly resolve ultra-thin oxide layers (e.g. 0.45um)."""
        if len(ticks) < 2: return ticks
        
        sorted_ticks = sorted(list(set(ticks)))
        filtered = [sorted_ticks[0]]
        
        for i in range(1, len(sorted_ticks)):
            curr = sorted_ticks[i]
            prev = filtered[-1]
            if (curr - prev) > min_gap:
                filtered.append(curr)
                
        return np.array(filtered, dtype=np.float32)

    def generate(self) -> Mesh3D:
        # 1. Hierarchical Grid Generation
        # Sort boxes by priority (Low -> High) so High overwrites Low
        # We need to track the ORIGINAL index for box_map to align with ThermoSim.py
        indexed_boxes = list(enumerate(self.cfg.boxes)) # [(0, box0), (1, box1)...]
        sorted_items = sorted(indexed_boxes, key=lambda pair: pair[1].priority)
        
        # Collect all desired tick marks from all boxes
        x_ticks_set = set()
        y_ticks_set = set()
        z_ticks_set = set()
        
        # Default global size
        global_h = self.max_h
        
        for idx, box in sorted_items:
            ox, oy, oz = box.origin
            w, h, d = box.size
            
            # 1.1 Collect Hard Constraints
            x_ticks_set.add(ox); x_ticks_set.add(ox+w)
            y_ticks_set.add(oy); y_ticks_set.add(oy+h)
            z_ticks_set.add(oz); z_ticks_set.add(oz+d)

            # 1.2 Local Refinement (If box.mesh_size is set)
            if box.mesh_size is not None:
                h_loc = box.mesh_size
                # Add subdivided ticks for this specific box
                def fill_loc(start, length, hl, tset):
                    if length > hl:
                        n = int(np.ceil(length / hl))
                        s = length / n
                        for i in range(1, n): tset.add(start + i * s)
                fill_loc(ox, w, h_loc, x_ticks_set)
                fill_loc(oy, h, h_loc, y_ticks_set)
                fill_loc(oz, d, h_loc, z_ticks_set)

            # 1.3 Power-Aware Refinement (FAM) - DISABLED for Grid Grading
            # The LayoutParser automatically maps power onto whatever grid we generate.
            # Injecting hard ticks for every block forces the entire XY plane to be extremely dense,
            # destroying the benefits of Mesh Efficiency (Grid Stretching).
            if box.floorplan_file:
                pass 


            # 1.4 Constraint Enforcement (Smart Heuristic)
            # If the user didn't specify MinElements, but the layer is thinner than max_h,
            # we AUTOMATICALLY force it to have at least 3 elements (Top, Middle, Bottom clarity).
            
            effective_min = box.min_elements
            
            # Heuristic: Check if box is "Thin" relative to global mesh
            is_thin_x = w < global_h
            is_thin_y = h < global_h
            is_thin_z = d < global_h
            
            # Auto-upgrade logic (only if default 1 is used)
            if effective_min <= 1:
                # We prioritize Z-axis for stacked dies, but logic applies to all.
                # If it's a "Thin Layer" (like Bonding/Oxide), force 3 elements UNLESS it's a SmartLayer.
                # SmartLayer means we accept 1 element and handle physics in Solver.
                is_smart = getattr(box, 'smart_layer', False)
                # DEBUG print removed
                
                if is_thin_z and not is_smart: 
                    effective_min = 3
            
            # DEBUG print removed

            if effective_min > 0:
                def enforce_min(start, length, min_el, tset):
                    if min_el <= 0: return
                    # Calculate required resolution to satisfy min_elements
                    d_req = length / min_el
                    
                    # We inject (min_el - 1) internal ticks
                    if min_el > 1:
                        step = length / min_el
                        for i in range(1, min_el):
                            tset.add(start + i * step)
                    
                # Apply selectively or globally based on thinness?
                # The 'effective_min' is mainly derived from Z-thinness for chip stacks.
                # Applying 3 cuts to X/Y for a huge heatsink is wrong.
                # So we verify 'is_thin' per axis.
                
                # However, box.min_elements (Explicit) applies to ALL axes.
                # Our auto-heuristic should only apply to the THIN axis.
                
                min_x = 3 if (is_thin_x and box.min_elements <= 1 and not is_smart) else box.min_elements
                min_y = 3 if (is_thin_y and box.min_elements <= 1 and not is_smart) else box.min_elements
                min_z = 3 if (is_thin_z and box.min_elements <= 1 and not is_smart) else box.min_elements
                
                enforce_min(ox, w, min_x, x_ticks_set)
                enforce_min(oy, h, min_y, y_ticks_set)
                enforce_min(oz, d, min_z, z_ticks_set)

        # 2. Conformal Gap-Filling (Minimal Dummy Mesh)
        # Instead of subdividing per-box, we subdivide the gaps between ALL hard constraints.
        global_expansion_ratio = getattr(self.cfg, 'expansion_ratio', 1.2)
        x_grid = self._subdivide_gaps(x_ticks_set, self.max_h, global_expansion_ratio)
        y_grid = self._subdivide_gaps(y_ticks_set, self.max_h, global_expansion_ratio)
        z_grid = self._subdivide_gaps(z_ticks_set, self.max_h, global_expansion_ratio)

        
        nx_e, ny_e, nz_e = len(x_grid)-1, len(y_grid)-1, len(z_grid)-1
        print(f"  Gap-Filled Grid: {nx_e}x{ny_e}x{nz_e} elements.")
        
        return self._finalize_mesh(x_grid, y_grid, z_grid, nx_e, ny_e, nz_e, sorted_items)

    def _subdivide_gaps(self, points_set, max_h, bias=1.2):
        """Subdivides gaps between sorted points.
        Implements 'Geometric Biasing': Small steps near boundaries, larger steps in the middle."""
        points = self._filter_slivers(list(points_set))
        if len(points) == 0:
             return np.array([])
        grid = [points[0]]
        
        # We ensure bias is at least slightly > 1.0 to prevent infinite loops, but default is 1.2
        if bias <= 1.01: bias = 1.01
        
        for i in range(len(points)-1):
            p0, p1 = points[i], points[i+1]
            gap = p1 - p0
            
            # Estimate Local step size at the boundary for smooth transition
            h_start0 = points[i] - points[i-1] if i > 0 else max_h / 5.0
            h_start1 = points[i+2] - points[i+1] if i < len(points)-2 else max_h / 5.0
            
            # Clamp starting steps to reasonable bounds
            h_start0 = min(max_h, max(1e-6, h_start0))
            h_start1 = min(max_h, max(1e-6, h_start1))
            
            # Only subdivide if gap exceeds max_h
            if gap > max_h * 1.05:
                # If gap is huge, use Graded Mesh
                if gap > max_h * 3.0:
                    half_gap = gap / 2.0
                    
                    # Forward from p0
                    curr_h = h_start0
                    curr_p = p0
                    while (curr_p + curr_h) < (p0 + half_gap - 0.1 * max_h):
                        curr_p += curr_h
                        grid.append(curr_p)
                        curr_h = min(curr_h * bias, max_h * 10.0) # Allow bulk to grow larger
                    
                    # Backward from p1
                    rev_pts = []
                    curr_h = h_start1
                    curr_p = p1
                    while (curr_p - curr_h) > (p1 - half_gap + 0.1 * max_h):
                        curr_p -= curr_h
                        rev_pts.append(curr_p)
                        curr_h = min(curr_h * bias, max_h * 10.0)
                    
                    for pt in reversed(rev_pts):
                        grid.append(pt)
                else:
                    # Standard linear subdivision for small gaps
                    n_int = int(np.ceil(gap / max_h))
                    step = gap / n_int
                    for j in range(1, n_int): grid.append(p0 + j * step)
                    
            grid.append(p1)
            
        # Ensure unique, sorted, and precision-safe
        return np.array(sorted(list(set(np.round(grid, 12)))), dtype=np.float32)
        
    def _finalize_mesh(self, x_grid, y_grid, z_grid, nx_e, ny_e, nz_e, sorted_items):
        # 3. Filter Active Elements (Sparse) & Map Materials
        # Construct dense centroid grid
        xc = 0.5 * (x_grid[:-1] + x_grid[1:])
        yc = 0.5 * (y_grid[:-1] + y_grid[1:])
        zc = 0.5 * (z_grid[:-1] + z_grid[1:])
        
        # 3D Arrays for properties (Z, Y, X)
        active = np.zeros((nz_e, ny_e, nx_e), dtype=bool)
        mats   = np.zeros((nz_e, ny_e, nx_e), dtype=np.int32)
        box_map = np.full((nz_e, ny_e, nx_e), -1, dtype=np.int32)
        
        # Vectorized Box Check using Meshgrid
        gv_z, gv_y, gv_x = np.meshgrid(zc, yc, xc, indexing='ij')
        
        # Apply boxes in Priority Order (active overwrites)
        
        for idx, box in sorted_items:
            ox, oy, oz = box.origin
            w, h, d = box.size
            
            # Check containment
            mask = (gv_x >= ox) & (gv_x <= ox+w) & \
                   (gv_y >= oy) & (gv_y <= oy+h) & \
                   (gv_z >= oz) & (gv_z <= oz+d)
                   
            # Apply (Last box wins = High Priority wins)
            active[mask] = True
            mats[mask] = box.material_id
            
            # Use ORIGINAL index (idx) so ThermoSim.py can map it back to cfg.boxes[idx]
            box_map[mask] = idx
            
        return Mesh3D(x_grid, y_grid, z_grid, active, mats, box_map)

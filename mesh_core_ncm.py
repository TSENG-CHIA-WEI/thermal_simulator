import os
import numpy as np
from config_parser import SimConfig, BoxDef

class Mesh3D:
    def __init__(self, x_grid, y_grid, z_grid, active_mask, material_ids, box_ids):
        """
        active_mask: boolean array (nz-1, ny-1, nx-1) indicating if element exists.
        material_ids: array (nz-1, ny-1, nx-1)
        
        We store full dense arrays for simplified indexing, but Solver will compressed them.
        """
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.z_grid = z_grid
        self.active_mask = active_mask
        self.material_ids = material_ids
        self.box_ids = box_ids # Maps element to index in sim_config.boxes
        
        self.nx = len(x_grid)
        self.ny = len(y_grid)
        self.nz = len(z_grid)
        
    @property
    def num_nodes(self):
        return self.nx * self.ny * self.nz

    @property
    def num_elements_dense(self):
        return (self.nx - 1) * (self.ny - 1) * (self.nz - 1)
        
    @property
    def num_active_elements(self):
        return np.count_nonzero(self.active_mask)
        
    def get_element_centroids_dense(self):
        xc = 0.5 * (self.x_grid[:-1] + self.x_grid[1:])
        yc = 0.5 * (self.y_grid[:-1] + self.y_grid[1:])
        zc = 0.5 * (self.z_grid[:-1] + self.z_grid[1:])
        
        # Z-slow, Y, X order
        gv_z, gv_y, gv_x = np.meshgrid(zc, yc, xc, indexing='ij')
        return np.stack([gv_x.flatten(), gv_y.flatten(), gv_z.flatten()], axis=1)

    def get_node_pos(self, node_idx):
        """Converts flat node index to (x, y, z) coordinates."""
        # Row-major (Z, Y, X)
        nx, ny = self.nx, self.ny
        k = node_idx // (ny * nx)
        j = (node_idx % (ny * nx)) // nx
        i = node_idx % nx
        return (self.x_grid[i], self.y_grid[j], self.z_grid[k])

class ActiveMeshGenerator:
    def __init__(self, sim_config: SimConfig, max_element_size=0.002):
        self.cfg = sim_config
        self.max_h = max_element_size
        
    def _generate_linear_ticks(self, start, end, step):
        """Generates ticks from start to end with given step, inclusive."""
        if step <= 0: return [start, end]
        # Use arange but be careful with float precision at the end
        ticks = np.arange(start, end, step)
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
                
        return np.array(filtered)

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

            # 1.3 Power-Aware Refinement (FAM)
            if box.floorplan_file:
                fp_path = box.floorplan_file
                if os.path.exists(fp_path):
                    try:
                        from layout_parser import LayoutParser
                        lp = LayoutParser()
                        lp.parse(fp_path)
                        for block in lp.blocks:
                            x_ticks_set.add(ox + block.min_x)
                            x_ticks_set.add(ox + block.max_x)
                            y_ticks_set.add(oy + block.min_y)
                            y_ticks_set.add(oy + block.max_y)
                            
                            # Injection of center line for peak accuracy in small sources
                            x_ticks_set.add(ox + (block.min_x + block.max_x)/2.0)
                            y_ticks_set.add(oy + (block.min_y + block.max_y)/2.0)
                    except Exception as e:
                        print(f"    [Warning] Could not parse floorplan {fp_path}: {e}")

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
                # If it's a "Thin Layer" (like Bonding/Oxide), force 3 elements.
                if is_thin_z: effective_min = 3
            
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
                
                min_x = 3 if (is_thin_x and box.min_elements <= 1) else box.min_elements
                min_y = 3 if (is_thin_y and box.min_elements <= 1) else box.min_elements
                min_z = 3 if (is_thin_z and box.min_elements <= 1) else box.min_elements # Use calculated effective
                
                enforce_min(ox, w, min_x, x_ticks_set)
                enforce_min(oy, h, min_y, y_ticks_set)
                enforce_min(oz, d, min_z, z_ticks_set)

        # 2. Conformal Gap-Filling (Minimal Dummy Mesh)
        # Instead of subdividing per-box, we subdivide the gaps between ALL hard constraints.
        x_grid = self._subdivide_gaps(x_ticks_set, self.max_h)
        y_grid = self._subdivide_gaps(y_ticks_set, self.max_h)
        z_grid = self._subdivide_gaps(z_ticks_set, self.max_h)
        
        nx_e, ny_e, nz_e = len(x_grid)-1, len(y_grid)-1, len(z_grid)-1
        print(f"  Gap-Filled Grid: {nx_e}x{ny_e}x{nz_e} elements.")
        
        return self._finalize_mesh(x_grid, y_grid, z_grid, nx_e, ny_e, nz_e, sorted_items)

    def _subdivide_gaps(self, points_set, max_h):
        """Subdivides gaps between sorted points.
        Implements 'Geometric Biasing': Small steps near boundaries, larger steps in the middle."""
        points = self._filter_slivers(list(points_set))
        grid = [points[0]]
        
        # Growth factor for "Subtraction" (1.08 means each step is 8% larger than previous)
        # Further reducing from 1.15 to 1.08 to move even closer to baseline T_max.
        bias = 1.08 
        
        for i in range(len(points)-1):
            p0, p1 = points[i], points[i+1]
            gap = p1 - p0
            
            # Only subdivide if gap exceeds max_h
            if gap > max_h * 1.05:
                # If gap is huge (Bulk regions like Heatsink), use Graded Mesh
                if gap > max_h * 3.0:
                    # Grow from both boundaries towards the center
                    half_gap = gap / 2.0
                    
                    # Forward from p0
                    curr_h = max_h
                    curr_p = p0
                    while (curr_p + curr_h) < (p0 + half_gap - 0.1 * max_h):
                        curr_p += curr_h
                        grid.append(curr_p)
                        curr_h = min(curr_h * bias, max_h * 20.0) # Cap at 20x max_h
                    
                    # Backward from p1
                    rev_pts = []
                    curr_h = max_h
                    curr_p = p1
                    while (curr_p - curr_h) > (p1 - half_gap + 0.1 * max_h):
                        curr_p -= curr_h
                        rev_pts.append(curr_p)
                        curr_h = min(curr_h * bias, max_h * 20.0)
                    
                    for pt in reversed(rev_pts):
                        grid.append(pt)
                else:
                    # Standard linear subdivision for small gaps (Precision zones)
                    n_int = int(np.ceil(gap / max_h))
                    step = gap / n_int
                    for j in range(1, n_int): grid.append(p0 + j * step)
                    
            grid.append(p1)
            
        # Ensure unique, sorted, and precision-safe
        return np.array(sorted(list(set(np.round(grid, 12)))))
        
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

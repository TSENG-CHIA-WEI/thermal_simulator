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
        # FD-SLA Implementation
        
        # 1. Sort boxes by priority (for Finalize step, and for Solid Strategy resolution)
        indexed_boxes = list(enumerate(self.cfg.boxes))
        sorted_items = sorted(indexed_boxes, key=lambda pair: pair[1].priority)
        
        # 2. Generate Axes using FD-SLA
        # Get Simulation Domain Bounds (from Boxes)
        # We assume domain starts at 0,0,0 or min(box_min)?
        # Usually user defines boxes. We need to cover all boxes.
        # ActiveMeshGenerator doesn't enforce a "World Box".
        # But _generate_smart_axis takes "axis_len".
        # We should calculate global bounds first.
        
        global_min = [float('inf')] * 3
        global_max = [float('-inf')] * 3
        
        for box in self.cfg.boxes:
            for i in range(3):
                 global_min[i] = min(global_min[i], box.origin[i])
                 global_max[i] = max(global_max[i], box.origin[i] + box.size[i])
                 
        # If no boxes, default 1mm
        if global_min[0] == float('inf'):
            global_max = [0.001, 0.001, 0.001]
            
        # We assume grid starts at global_min? 
        # The previous implementation assumed absolute coordinates.
        # But _generate_smart_axis logic "anchors = set([0.0, axis_len])" implies 0-based relative?
        # WAIT. User spec: "anchors.add(box.min_coord)".
        # And "anchors = set([0.0, axis_len])".
        # This implies the domain is [0, L].
        # But real simulation coordinates might be negative (e.g. -0.05).
        # We should change the "0.0, axis_len" assumption to "global_min, global_max".
        
        x_grid = self._generate_smart_axis(global_min[0], global_max[0], self.cfg.boxes, 0)
        y_grid = self._generate_smart_axis(global_min[1], global_max[1], self.cfg.boxes, 1)
        z_grid = self._generate_smart_axis(global_min[2], global_max[2], self.cfg.boxes, 2)
        
        nx_e, ny_e, nz_e = len(x_grid)-1, len(y_grid)-1, len(z_grid)-1
        print(f"  [FD-SLA] Smart Grid: {nx_e}x{ny_e}x{nz_e} elements.")
        
        # 3. Finalize
        return self._finalize_mesh(x_grid, y_grid, z_grid, nx_e, ny_e, nz_e, sorted_items)

    def _generate_smart_axis(self, global_min, global_max, boxes, axis_idx):
        """
        Feature-Driven Scan-Line Algorithm (FD-SLA)
        """
        # 1. Anchor Collection
        anchors = set([global_min, global_max])
        
        for box in boxes:
            start = box.origin[axis_idx]
            end = start + box.size[axis_idx]
            anchors.add(round(start, 9))
            anchors.add(round(end, 9))
            
            # Floorplan Features (only if NOT NCM mode)
            # Only for X (0) and Y (1) axes
            if axis_idx != 2 and box.floorplan_file and not getattr(box, 'ncm_mode', False):
                fp_path = box.floorplan_file
                if os.path.exists(fp_path):
                    try:
                        from layout_parser import LayoutParser
                        lp = LayoutParser()
                        lp.parse(fp_path)
                        ox = box.origin[axis_idx] # offset for this axis
                        
                        for block in lp.blocks:
                            # block.min_x / max_x correspond to axis 0
                            # block.min_y / max_y correspond to axis 1
                            if axis_idx == 0:
                                b_min, b_max = block.min_x, block.max_x
                            else:
                                b_min, b_max = block.min_y, block.max_y
                                
                            anchors.add(round(ox + b_min, 9))
                            anchors.add(round(ox + b_max, 9))
                            
                            # Center injection (Optional but good for peak power capture)
                            # User "Feature First" implies boundaries. Center also holds peak.
                            anchors.add(round(ox + (b_min + b_max)/2.0, 9))
                            
                    except Exception as e:
                        print(f"    [Warning] Could not parse floorplan {fp_path}: {e}")

        # ... (Rest of logic: Sorted Anchors -> Intervals -> Strategy)
        # We need to make sure the loop logic below uses the updated signature/logic
        # ... (Rest of logic: Sorted Anchors -> Intervals -> Strategy)
        # OPTIMIZATION: Snap Anchors to prevent Conformal Mesh Fracture (Sliver Removal)
        # Use 200um tolerance for X/Y (Planar) to merge misaligned boundaries (LHS can shift >50um).
        # Use 0.1um tolerance for Z (Vertical) to preserve thin layers (FEOL ~1um).
        snap_tol = 2.0e-4 if axis_idx != 2 else 1e-7
        
        raw_anchors = list(anchors)
        merged_anchors = self._merge_close_anchors(raw_anchors, tol=snap_tol)
        
        sorted_anchors = sorted(merged_anchors)
        final_grid = []
        global_max_h = self.max_h
        
        # 2. Iterate Intervals
        for i in range(len(sorted_anchors) - 1):
            start, end = sorted_anchors[i], sorted_anchors[i+1]
            length = end - start
            if length < 1e-10: continue 
            
            mid = (start + end) / 2.0
            
            # Identify Context
            is_solid = False
            active_box = None
            is_gap = False
            
            relevant_boxes = []
            for box in boxes:
                b_start = round(box.origin[axis_idx], 9)
                b_end = round(b_start + box.size[axis_idx], 9)
                if mid >= b_start and mid <= b_end:
                   is_solid = True
                   relevant_boxes.append(box)
            
            if is_solid:
                # Select the box that dictates the mesh strategy.
                # For Resolution, we want the MOST RESTRICTIVE box (Source > Explicit Mesh > Default).
                # Note: This differs from Material Assignment which uses User Priority.
                def resolution_score(b):
                    score = 0
                    if b.floorplan_file: score += 1000
                    if b.mesh_size is not None: score += 100
                    # Tie-break with priority (optional, or just size?)
                    score += b.priority * 0.01
                    return score
                    
                relevant_boxes.sort(key=resolution_score, reverse=True)
                active_box = relevant_boxes[0]
            
            if not is_solid:
                GAP_THRESHOLD = 0.002
                if length < GAP_THRESHOLD:
                    is_gap = True
            
            # 3. Apply Strategy
            ticks = []
            # UNIVERSAL DENSITY-BASED ALGORITHM (Physics-Driven)
            # Instead of forcing fixed element counts (which explodes on slivers),
            # we use Target Resolution based on the context.
            
            # 1. Determine Context and Target Resolution
            target_h = 0.005 # Default Global
            
            if is_solid:
                # Inside a Heat Source or Active Component
                # User Request: "Encrypt to 6 Million Elements"
                # Target: 70um (0.00007)
                is_source = bool(active_box.floorplan_file)
                if is_source:
                    target_h = 0.00007 # 70um - Super Dense
                elif active_box.mesh_size:
                    target_h = active_box.mesh_size
                else:
                    target_h = 0.002 # 2.0mm for Passive Solids (Mold, etc)
                    
            elif is_gap:
                 # Gap between Solids
                 # Target: 200um (0.2mm) to match extremely dense source
                 target_h = 0.0002 
            else:
                 # Bulk / Empty Space
                 # Target: Coarse (2.0mm - 5.0mm)
                 target_h = min(global_max_h, 0.002) # Cap at 2mm for smoothness
                 
            # 2. Calculate Element Count (N)
            # Logic: N = Length / Target.
            # - Large Block (2mm) / 0.2mm -> 10 elements (High Fidelity)
            # - Tiny Sliver (0.05mm) / 0.2mm -> 0.25 -> 1 element (Robust, no explosion)
            
            # Apply grading only for large non-critical regions
            use_grading = (not is_solid) and (length > target_h * 2.0)
            
            ticks = []
            if use_grading:
                 ticks = self._generate_graded_subinterval(start, end, target_h)
            else:
                 n_elements = max(1, int(np.round(length / target_h)))
                 # Safety: if length is noticeably larger than target (e.g. 1.5x), ensure at least 2
                 if length > target_h * 1.5:
                     n_elements = max(n_elements, 2)
                     
                 step = length / n_elements
                 ticks = [start + k * step for k in range(n_elements+1)]
                    

            
            if final_grid:
                final_grid.extend(ticks[1:])
            else:
                final_grid.extend(ticks)
                
        return np.array(final_grid)




    def _merge_close_anchors(self, anchors, tol=1e-5):
        """
        Merges anchors that are within 'tol' distance of each other to prevent
        sliver elements (Conformal Mesh Fracture).
        """
        if not anchors: return []
        
        sorted_a = sorted(list(set(anchors))) # Unique sort
        merged = []
        
        if len(sorted_a) == 0: return []
        
        # Iterative merge
        curr_group = [sorted_a[0]]
        
        for i in range(1, len(sorted_a)):
            val = sorted_a[i]
            # Since sorted, check diff with last group avg? No, with group start is simpler for stability.
            if (val - curr_group[0]) < tol:
                 curr_group.append(val)
            else:
                 # Flush group
                 avg = sum(curr_group) / len(curr_group)
                 merged.append(avg)
                 curr_group = [val]
                 
        # Flush last
        if curr_group:
             avg = sum(curr_group) / len(curr_group)
             merged.append(avg)
             
        return merged

    def _generate_graded_subinterval(self, start, end, max_h):
        # Biased Graded Mesh (Symmetric Geometric Expansion)
        # Expands from 'start' and 'end' towards the center using 'max_h' as the base size.
        # Saves elements in large voids (Bulk).
        
        length = end - start
        if length <= max_h:
            return [start, end]
            
        bias = 1.3 # Aggressive expansion
        h_limit = 0.005 # Cap max element size at 5mm (User context: chip is ~30mm, 5mm is reasonable for far field)
        
        # 1. Generate tick list from Left
        ticks_l = [start]
        curr = start
        h = max_h
        
        # We go until mid-ish. To avoid complexity, we generate full candidates and merge?
        # Simpler: Generate forward until h hits limit or we cover half.
        
        # Safety: ALWAYS add max iteration limit to prevent infinite loops
        # Quick Fix: Reduced from 10000 to 1000 to prevent excessive mesh density
        max_iterations = 1000
        max_ticks = 500  # Additional safety: limit number of ticks per side
        iteration_count = 0
        
        while iteration_count < max_iterations:
            iteration_count += 1
            
            # AGGRESSIVE EARLY-STOPPING: If we've generated too many ticks, stop immediately
            if len(ticks_l) >= max_ticks:
                print(f"WARNING: Tick limit reached ({len(ticks_l)} >= {max_ticks}), stopping early")
                break
            
            # Safety: Check for NaN/Inf in current position or step size
            if not np.isfinite(curr) or not np.isfinite(h):
                print(f"ERROR: Non-finite values detected (curr={curr}, h={h}), stopping iteration")
                break
            
            # Distance remaining to midpoint
            midpoint = start + length / 2.0
            remaining = midpoint - curr
            
            if remaining <= h:
                # Close enough to midpoint, stop here
                break
            
            # AGGRESSIVE: If we're taking too many iterations, force larger steps
            if iteration_count > 100 and h < h_limit:
                h = min(h * 2.0, h_limit)  # Double step size to accelerate
            
            h_next = min(h * bias, h_limit)
            
            # Check if we should be in constant h_limit zone
            if h >= h_limit and remaining > 3 * h_limit:
                # In bulk region, use constant h_limit
                curr += h_limit
                if curr >= midpoint or not np.isfinite(curr):  # Safety check
                    break
                ticks_l.append(curr)
                h = h_limit
            else:
                # Standard graded expansion
                new_pos = curr + h
                if new_pos >= midpoint or not np.isfinite(new_pos):  # Reached midpoint or invalid
                    break
                ticks_l.append(new_pos)
                curr = new_pos
                h = h_next
        
        # Safety check after loop
        if iteration_count >= max_iterations:
            print(f"WARNING: Max iterations ({max_iterations}) reached, ticks_l has {len(ticks_l)} points")

        # 2. Mirror for Right side
        # The right side ticks are (end - (t - start))
        ticks_r = []
        for t in ticks_l:
             dist = t - start
             ticks_r.append(end - dist)
        ticks_r.reverse() # [end-0, end-h, ...] -> [..., end]
        
        # 3. Stitch Middle
        # ticks_l ends at 'last_l', ticks_r starts at 'first_r'
        last_l = ticks_l[-1]
        first_r = ticks_r[0]
        
        gap = first_r - last_l
        
        if gap < 1e-9:
            # Perfectly met (unlikely)
            final_ticks = ticks_l + ticks_r[1:]
        elif gap < max(h, h_limit) * 1.5:
            # Small gap, just link them? Or add one point?
            # If gap is tiny, merge?
            final_ticks = ticks_l + ticks_r
        else:
            # Significant gap. Fill linearly with the current 'h' (which is the peak size)
            # CRITICAL: Add comprehensive safety checks
            
            # Check for invalid gap
            if not np.isfinite(gap) or gap < 0:
                print(f"WARNING: Invalid gap={gap}, connecting directly")
                final_ticks = ticks_l + ticks_r
            else:
                # Determine fill step size
                h_fill = max(min(h, h_limit), max_h)  # At least max_h
                
                # Safety: ensure h_fill is valid and non-zero
                if h_fill < 1e-9 or not np.isfinite(h_fill):
                    h_fill = max(max_h, 1e-4) if np.isfinite(max_h) else 1e-4
                
                # Calculate number of fill points with safety limits
                n_fill_raw = gap / h_fill
                
                if not np.isfinite(n_fill_raw):
                    # Inf or NaN result, fallback to simple connection
                    print(f"WARNING: n_fill overflow (gap={gap:.6e}, h_fill={h_fill:.6e})")
                    final_ticks = ticks_l + ticks_r
                elif n_fill_raw > 100000:
                    # Too many points, limit to prevent memory issues
                    print(f"WARNING: Excessive n_fill={n_fill_raw:.0f}, capping at 1000")
                    n_fill = 1000
                    step = gap / n_fill
                    fill_ticks = [last_l + k * step for k in range(1, n_fill)]
                    final_ticks = ticks_l + fill_ticks + ticks_r
                else:
                    # Normal case
                    n_fill = max(1, int(np.ceil(n_fill_raw)))
                    step = gap / n_fill
                    fill_ticks = [last_l + k * step for k in range(1, n_fill)]
                    final_ticks = ticks_l + fill_ticks + ticks_r
            
        return sorted(list(set(final_ticks)))

        
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

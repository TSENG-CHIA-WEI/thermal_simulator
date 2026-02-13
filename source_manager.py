
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class PowerSource:
    x: float
    y: float
    w: float
    h: float
    power: float # Total watts

class SourceManager:
    def __init__(self, mesh):
        self.mesh = mesh
        self.sources: List[PowerSource] = []

    def add_source(self, source: PowerSource):
        self.sources.append(source)

    def apply_sources(self, solver):
        """
        Distributes power to the load vector F_vec in the solver.
        
        Algorithm:
        For each source, find overlapping elements.
        Add Q contribution to F_vec.
        
        Optimization:
        Instead of checking all elements, we can do a bounding box search
        if the mesh is structured.
        """
        # For each element, we need to know its centroid or bounds
        # For specific heat Q (W/m^2), we integrate N^T * Q dOmega.
        # If Q is constant in the element: F_e = Integral(N^T) * Q
        # Integral(N^T) over element = Area/4 * [1,1,1,1] (for rectangle)
        
        # Simplified "Lumped" Mapping:
        # Distribute Total Power P proportionally to the area overlap of the source with the element.
        # Then distribute element power to nodes evenly (or via shape functions).
        
        # 1. Compute element bounds (vectorized)
        nodes = self.mesh.nodes
        elems = self.mesh.elements
        
        # Gather all x and y coordinates for all elements
        # Shape (N_elem, 4)
        el_x = nodes[elems, 0]
        el_y = nodes[elems, 1]
        
        # Bounding boxes for all elements
        # (N_elem,)
        min_x = np.min(el_x, axis=1)
        max_x = np.max(el_x, axis=1)
        min_y = np.min(el_y, axis=1)
        max_y = np.max(el_y, axis=1)
        
        # Element areas (approximate as (xmax-xmin)*(ymax-ymin) for rectangular grid)
        # For general quads, use Shoelace or cross product.
        # Assuming rectilinear for now as per mesh_core.
        el_area = (max_x - min_x) * (max_y - min_y)
        
        for src in self.sources:
            # Source bounds
            sx1, sx2 = src.x, src.x + src.w
            sy1, sy2 = src.y, src.y + src.h
            
            # Vectorized Overlap Layout
            # Interaction width
            iw = np.minimum(max_x, sx2) - np.maximum(min_x, sx1)
            # Interaction height
            ih = np.minimum(max_y, sy2) - np.maximum(min_y, sy1)
            
            # Valid overlaps have positive w and h
            overlap_w = np.maximum(0, iw)
            overlap_h = np.maximum(0, ih)
            overlap_area = overlap_w * overlap_h
            
            # Check which elements have overlap
            mask = overlap_area > 0
            
            if not np.any(mask):
                continue
                
            # Power density q (W/m^2) assumed uniform over the source?
            # Or Total Power P distributed?
            # Let's assume we want to deposit exactly 'src.power' into the system.
            # Total overlap area logic:
            # We can treat this as "Volumetric Heat Generation" in the overlapping region.
            # q_vol = Power / (Source Area)
            # Power injected into element e: P_e = q_vol * overlap_area_e
            
            q_source = src.power / (src.w * src.h)
            power_per_elem = q_source * overlap_area[mask]
            
            # Distribute P_e to nodes. For Q4, equal distribution is 0.25 * P_e per node.
            # This is mathematically equivalent to Integral(N^T * q) if q const.
            
            affected_elems = elems[mask] # (N_hit, 4)
            p_vec = power_per_elem[:, None] * 0.25 # (N_hit, 1)
            
            # Add to global F_vec using np.add.at for safety with duplicate nodes
            # solver.F_vec is (N_nodes,)
            # We flatten the affected elements and repeat the power values
            
            np.add.at(solver.F_vec, affected_elems.flatten(), np.repeat(p_vec, 4))
            

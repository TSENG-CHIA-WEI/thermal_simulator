
import csv
from dataclasses import dataclass
from typing import List

from config_parser import parse_val
import numpy as np

@dataclass
class LayoutBlock:
    layer_name: str
    block_name: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    power: float
    
class LayoutParser:
    def __init__(self):
        self.blocks: List[LayoutBlock] = []
        
    def parse(self, filepath: str):
        """
        Reads CSV layout file.
        Format: LayerName, BlockName, MinX, MinY, MaxX, MaxY, Power
        """
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            # Skip header if present
            header = next(reader, None)
            
            # Check header for format detection
            use_wh = False
            if header:
                header_str = ",".join(header).lower()
                if "width" in header_str or "w(m)" in header_str:
                    use_wh = True
                    
            # STRICT MODE Check
            # If header is missing but data has 6 columns, we are in DANGER ZONE.
            # We assume Default is [BlockName, x1, y1, x2, y2, Power].
            # If user meant W/H, they MUST provide a header or use 7 columns.
            if not header and use_wh:
                 # This path is actually unreachable because use_wh depends on header
                 pass
            
            for row in reader:
                if not row or row[0].startswith('#'): continue
                
                # Check column count
                if len(row) < 6: 
                    print(f"[LayoutParser] Skipping invalid row: {row}")
                    continue
                    
                # Clean whitespace
                row = [r.strip() for r in row]
                
                # Default init to avoid UnboundLocalError
                l_name, b_name = "", ""
                mx, my, Mx, My, p = 0.0, 0.0, 0.0, 0.0, 0.0
                
                if len(row) == 6:
                    # New Format (Linked to Box, 6 columns)
                    l_name = "Linked"
                    b_name = row[0]
                    
                    if use_wh:
                        # Format: BlockName, X, Y, W, H, Power
                        # Header said "Width", so we trust it.
                        x = parse_val(row[1])
                        y = parse_val(row[2])
                        w = parse_val(row[3])
                        h = parse_val(row[4])
                        
                        mx = x
                        my = y
                        Mx = x + w
                        My = y + h
                    else:
                        # Format: BlockName, x1, y1, x2, y2, Power
                        # NO Header saying "Width".
                        # We assume Min/Max coordinates.
                        # WARN if user might be confused?
                        # For now, stick to Logic: No Header = Min/Max.
                        mx = parse_val(row[1])
                        my = parse_val(row[2])
                        Mx = parse_val(row[3])
                        My = parse_val(row[4])
                        
                    p  = parse_val(row[5])
                    
                elif len(row) >= 7:
                    # Old Format (Layer, Block, x1, y1, x2, y2, Power)
                    l_name = row[0]
                    b_name = row[1]
                    mx = parse_val(row[2])
                    my = parse_val(row[3])
                    Mx = parse_val(row[4])
                    My = parse_val(row[5])
                    p  = parse_val(row[6])
                
                # Append validity check?
                self.blocks.append(LayoutBlock(l_name, b_name, mx, my, Mx, My, p))

    def apply_power_mapping(self, box_origin, centroids, box_mask, power_field_dense):
        """
        Maps the parsed layout blocks to the dense power field.
        
        Args:
            box_origin: tuple (ox, oy, oz)
            centroids: tuple (cx, cy, cz) of numpy arrays
            box_mask: boolean mask for elements belonging to this box
            power_field_dense: target numpy array to update (in-place)
        """
        ox, oy, oz = box_origin
        cx, cy, cz = centroids
        
        for blk in self.blocks:
            # Global Coordinates of the block
            gx1 = ox + blk.min_x
            gy1 = oy + blk.min_y
            gx2 = ox + blk.max_x
            gy2 = oy + blk.max_y
            
            # Vectorized Masking
            # Note: We assume Z-range is handled by box_mask (layer assignment)
            src_mask = box_mask & (cx >= gx1) & (cx <= gx2) & (cy >= gy1) & (cy <= gy2)
            
            count = np.count_nonzero(src_mask)
            if count > 0:
                power_field_dense[src_mask] += blk.power / count

    def map_power_to_elements(self, x_coords: np.ndarray, y_coords: np.ndarray, box_origin: tuple) -> np.ndarray:
        """
        Maps power from layout blocks to a set of element centroids.
        Returns an array of power values (Watts) for each element.
        """
        power_values = np.zeros_like(x_coords, dtype=np.float32)
        ox, oy = box_origin[0], box_origin[1]
        
        for blk in self.blocks:
            gx1, gy1 = ox + blk.min_x, oy + blk.min_y
            gx2, gy2 = ox + blk.max_x, oy + blk.max_y
            
            # Vectorized check
            in_blk = (x_coords >= gx1) & (x_coords <= gx2) & (y_coords >= gy1) & (y_coords <= gy2)
            count = np.count_nonzero(in_blk)
            
            if count > 0:
                # Distribute block power evenly among covered elements
                power_values[in_blk] += blk.power / count
                
        return power_values

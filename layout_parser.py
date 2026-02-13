
import csv
from dataclasses import dataclass
from typing import List

from config_parser import parse_val

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
            
            for row in reader:
                if not row or row[0].startswith('#'): continue
                if len(row) < 6: continue 
                
                # ... (Logic to detect old/new format based on column count) ...
                
                # Clean whitespace
                row = [r.strip() for r in row]
                
                if len(row) == 6:
                    # New Format (Linked to Box)
                    l_name = "Linked"
                    b_name = row[0]
                    
                    if use_wh:
                        # Format: BlockName, X, Y, W, H, Power
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
                        mx = parse_val(row[1])
                        my = parse_val(row[2])
                        Mx = parse_val(row[3])
                        My = parse_val(row[4])
                        
                    p  = parse_val(row[5])
                else:
                    # Old Format (7 cols)
                    l_name = row[0]
                    b_name = row[1]
                    mx = parse_val(row[2])
                    my = parse_val(row[3])
                    Mx = parse_val(row[4])
                    My = parse_val(row[5])
                    p  = parse_val(row[6])
                
                self.blocks.append(LayoutBlock(l_name, b_name, mx, my, Mx, My, p))

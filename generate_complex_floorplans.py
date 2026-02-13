import csv
import os

def format_num(x):
    return "{:.10f}".format(x).rstrip('0').rstrip('.')

def generate_csv(path, h_offset_fn, background_p, active_target_p):
    soc_w, soc_h = 0.026, 0.033
    unit_w, unit_h = 0.00154, 0.00195
    grid_count = 13
    
    total_grid_w = unit_w * grid_count
    total_grid_h = unit_h * grid_count
    
    # Calculate centering offsets
    offset_x = (soc_w - total_grid_w) / 2.0
    offset_y = (soc_h - total_grid_h) / 2.0
    
    active_w = 0.00077
    active_h = 0.000975 # 1.95 / 2
    
    # User wants ADDITIVE power (疊加)
    # Background (55.83W) is applied to whole die.
    # Active areas (3.07W) are applied ON TOP of background.
    net_active_p = active_target_p

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["# BlockName", "X(m)", "Y(m)", "Width(m)", "Height(m)", "Power(W)"])
        # Background
        writer.writerow(["Background", 0.0, 0.0, format_num(soc_w), format_num(soc_h), format_num(background_p)])
        
        for j in range(grid_count):
            for i in range(grid_count):
                bx = offset_x + i * unit_w
                by = offset_y + j * unit_h
                
                # Apply per-unit offset (interleaving)
                ux, uy = h_offset_fn(active_w, active_h)
                
                final_x = bx + ux
                final_y = by + uy
                
                writer.writerow([f"Act_{i}_{j}", format_num(final_x), format_num(final_y), format_num(active_w), format_num(active_h), format_num(net_active_p)])

# Bot FEOL uses (0,0) relative offset as per User bd(0,0,0.77)
bot_path = "projects/chip_stack/soc_bot_feol.csv"
generate_csv(bot_path, lambda w, h: (0, 0), 55.83, 3.07)

# Top FEOL uses (0.77, 0.975) as per User td(0.77, 0.975, 0.77)
top_path = "projects/chip_stack/soc_top_feol.csv"
generate_csv(top_path, lambda w, h: (0.00077, 0.000975), 55.83, 3.07)

print(f"Generated {bot_path} and {top_path}")

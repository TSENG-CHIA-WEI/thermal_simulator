"""
Mesh Convergence Study
Run multiple simulations with different mesh sizes to verify solution convergence
"""
import subprocess
import sys
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def run_mesh_study(config_path, params_path, mesh_sizes, output_dir="mesh_study"):
    """
    Run simulation with different mesh sizes and collect results.
    
    Args:
        config_path: Path to .config file
        params_path: Path to params.config file  
        mesh_sizes: List of mesh_size values (e.g., [0.01, 0.005, 0.002])
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    print("=" * 70)
    print("MESH CONVERGENCE STUDY")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Mesh sizes: {mesh_sizes}")
    print()
    
    for i, mesh_size in enumerate(mesh_sizes):
        print(f"\n[{i+1}/{len(mesh_sizes)}] Running with mesh_size = {mesh_size}m...")
        print("-" * 70)
        
        # Run ThermoSim
        cmd = [
            sys.executable, 'ThermoSim.py',
            config_path, params_path,
            '--mesh_size', str(mesh_size)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ERROR: Simulation failed!")
            print(result.stderr)
            continue
        
        # Parse output
        output = result.stdout
        
        # Extract key metrics
        elements = None
        t_max = None
        iterations = None
        residual = None
        solve_time = None
        
        for line in output.split('\n'):
            if 'Smart Grid:' in line:
                # Parse "158x158x11 elements"
                parts = line.split()
                for p in parts:
                    if 'x' in p and 'elements' not in p:
                        dims = p.split('x')
                        if len(dims) == 3:
                            elements = int(dims[0]) * int(dims[1]) * int(dims[2])
            
            if '[Convergence] Iterations:' in line:
                # Parse "Iterations: 951, Final Residual: 2.680326e-05"
                parts = line.split(',')
                iterations = int(parts[0].split(':')[-1].strip())
                residual = float(parts[1].split(':')[-1].strip())
            
            if 'Global Max:' in line:
                # Parse "Global Max: 95.40 C"
                t_max = float(line.split()[3])
            
            if 'Converged in' in line and 'seconds' in line:
                # Parse "Converged in 2.9196 seconds"
                solve_time = float(line.split()[-2])
        
        # Store results
        result_data = {
            'mesh_size': mesh_size,
            'elements': elements,
            't_max': t_max,
            'iterations': iterations,
            'final_residual': residual,
            'solve_time': solve_time
        }
        
        results.append(result_data)
        
        print(f"  Elements: {elements:,}")
        print(f"  T_max: {t_max:.2f} °C")
        print(f"  Iterations: {iterations}")
        print(f"  Residual: {residual:.6e}")
        print(f"  Solve Time: {solve_time:.2f}s")
    
    # Save results to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'mesh_study_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Results saved to {csv_path}")
    
    # Generate plots
    generate_plots(df, output_dir)
    
    return df

def generate_plots(df, output_dir):
    """Generate convergence plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Mesh Convergence Study', fontsize=16, fontweight='bold')
    
    # 1. T_max vs Elements
    ax = axes[0, 0]
    ax.plot(df['elements'], df['t_max'], 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Elements', fontweight='bold')
    ax.set_ylabel('T_max (°C)', fontweight='bold')
    ax.set_title('Temperature Convergence')
    ax.grid(True, alpha=0.3)
    
    # Add percentage change annotation
    if len(df) > 1:
        t_change = abs(df['t_max'].iloc[-1] - df['t_max'].iloc[-2])
        t_pct = t_change / df['t_max'].iloc[-1] * 100
        ax.text(0.05, 0.95, f'Last change: {t_change:.3f}°C ({t_pct:.2f}%)',
                transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Iterations vs Elements
    ax = axes[0, 1]
    ax.plot(df['elements'], df['iterations'], 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Elements', fontweight='bold')
    ax.set_ylabel('CG Iterations', fontweight='bold')
    ax.set_title('Solver Iterations vs Mesh Size')
    ax.grid(True, alpha=0.3)
    
    # 3. Final Residual vs Elements
    ax = axes[1, 0]
    ax.semilogy(df['elements'], df['final_residual'], 'go-', linewidth=2, markersize=8)
    ax.axhline(y=1e-6, color='r', linestyle='--', linewidth=1.5, label='Target (1e-6)')
    ax.set_xlabel('Number of Elements', fontweight='bold')
    ax.set_ylabel('Final Residual', fontweight='bold')
    ax.set_title('Convergence Quality')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Solve Time vs Elements
    ax = axes[1, 1]
    ax.plot(df['elements'], df['solve_time'], 'mo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Elements', fontweight='bold')
    ax.set_ylabel('Solve Time (s)', fontweight='bold')
    ax.set_title('Computational Cost')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'mesh_convergence_plots.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Plots saved to {plot_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Mesh Convergence Study')
    parser.add_argument('config', help='Path to .config file')
    parser.add_argument('params', help='Path to params.config file')
    parser.add_argument('--sizes', type=str, default='0.01,0.005,0.003,0.002',
                        help='Comma-separated mesh sizes (default: 0.01,0.005,0.003,0.002)')
    parser.add_argument('--output', type=str, default='mesh_study',
                        help='Output directory (default: mesh_study)')
    
    args = parser.parse_args()
    
    mesh_sizes = [float(x) for x in args.sizes.split(',')]
    
    df = run_mesh_study(args.config, args.params, mesh_sizes, args.output)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))
    print()
    
    # Check convergence
    if len(df) >= 2:
        t_diff = abs(df['t_max'].iloc[-1] - df['t_max'].iloc[-2])
        t_pct = t_diff / df['t_max'].iloc[-1] * 100
        
        print(f"T_max change (finest two meshes): {t_diff:.4f}°C ({t_pct:.3f}%)")
        
        if t_pct < 0.5:
            print("[OK] CONVERGED: T_max difference < 0.5%")
        elif t_pct < 1.0:
            print("[WARN] ACCEPTABLE: T_max difference < 1%")
        else:
            print("[FAIL] NOT CONVERGED: Consider finer mesh")

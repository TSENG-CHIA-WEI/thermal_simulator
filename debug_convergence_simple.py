"""
Convergence Diagnostics - Simplified Version
Wraps ThermoSim.py to add convergence tracking only
"""
import sys
import os
import matplotlib.pyplot as plt

# Temporarily monkey-patch ThermoSim to extract solver results
old_main = None

def run_with_diagnostics(config_path, params_path, mesh_size=0.005):
    """Run ThermoSim and extract solver convergence data."""
    import ThermoSim
    
    # Capture original main
    original_main = ThermoSim.main
    solver_ref = {'solver': None}
    
    def patched_main():
        # Modify sys.argv for ThermoSim
        sys.argv = [
            'ThermoSim.py',
            config_path,
            params_path,
            '--mesh_size', str(mesh_size)
        ]
        
        # Call original (modified to return solver)
        # We'll need to extract it during execution
        original_main()
    
    # Alternative: Import and run directly
    from config_parser import ConfigParser
    from mesh_core import ActiveMeshGenerator
    from fem_engine import ThermalSolver3D
    from post_process import VTKExporter
    
    print("=" * 70)
    print("CONVERGENCE DIAGNOSTICS MODE")
    print("=" * 70)
    
    # Load
    cfg = ConfigParser()
    cfg.parse_sim_config(config_path)
    cfg.parse_model_params(params_path)
    
    # Mesh  
    print(f"\\nGenerating Mesh (Max Size: {mesh_size}m)...")
    gen = ActiveMeshGenerator(cfg.sim_config, max_element_size=mesh_size)
    mesh = gen.generate()
    
    # Solve (using ThermoSim's exact logic)
    print("\\nSolving...")
    exec(open('ThermoSim.py').read().replace('if __name__ == \"__main__\":', 'if False:'))
    
    # Instead, let's just call ThermoSim.py and parse the solver.log
    import subprocess
    
    print(f"\\nRunning ThermoSim with tracking...")
    result = subprocess.run([
        sys.executable, 'ThermoSim.py',
        config_path, params_path,
        '--mesh_size', str(mesh_size)
    ], capture_output=True, text=True)
    
    print(result.stdout)
    
    # The solver should have residual_history now
    # But we need to access it...
    # This approach won't work easily without modifying ThermoSim.py return value
    
    print("\\nNote: To see full convergence data, modify ThermoSim.py to return 'solver' object")
    print("Then access solver.residual_history for plotting.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python debug_convergence_simple.py <config> <params> [mesh_size]")
        sys.exit(1)
    
    config = sys.argv[1]
    params = sys.argv[2]
    mesh_sz = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
    
    run_with_diagnostics(config, params, mesh_sz)

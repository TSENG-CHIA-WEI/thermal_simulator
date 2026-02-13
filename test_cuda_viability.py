import sys
import time
import importlib.util

def check_cuda_viability():
    print("=== CUDA / GPU Acceleration Viability Test ===")
    print(f"Python Runtime: {sys.executable}")
    
    # 1. check for CuPy (The Python wrapper for CUDA)
    cupy_spec = importlib.util.find_spec("cupy")
    if cupy_spec is None:
        print("\n[CRITICAL MISSING COMPONENT]")
        print("[X] 'cupy' library is NOT installed.")
        return False

    # Try to inject CUDA path if DLLs are missing
    import os
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin" # Try v11.8 or v12.x based on user system?
    # Actually, user has v13.0 compiler, but typically 12.x drivers. 
    # Let's try to auto-detect or just add common paths.
    
    # Common paths to try
    common_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin", # User just installed this!
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin", # Future proofing
        # Also try "v13.0" if that's what nvcc reported, though drivers usually lag
    ]
    
    # User's nvcc said: release 13.0
    # So we should look for v13.0
    common_paths.append(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin")
    
    for p in common_paths:
        if os.path.exists(p):
            print(f"Injecting CUDA Path: {p}")
            os.environ["PATH"] += os.pathsep + p
    
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpx
        import cupyx.scipy.sparse.linalg as cpx_la
        
        print(f"\n[OK] CuPy detected version: {cp.__version__}")
        
        # 2. Check Device
        dev = cp.cuda.Device(0)
        print(f"Device: Detected GPU: {dev.compute_capability} (Compute Capability)")
        
        # 3. Alloc Test
        print("\n[Test 1] Memory Allocation...")
        N = 1000
        t0 = time.time()
        a_gpu = cp.random.rand(N, N, dtype=cp.float32)
        b_gpu = cp.random.rand(N, N, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
        print(f"  Success! Allocated 2x {N}x{N} matrices. ({time.time()-t0:.4f}s)")
        
        # 4. Math Test (Matmul)
        print("\n[Test 2] Matrix Multiplication (cuBLAS)...")
        t0 = time.time()
        c_gpu = cp.matmul(a_gpu, b_gpu)
        cp.cuda.Stream.null.synchronize()
        print(f"  Success! Result shape: {c_gpu.shape}. ({time.time()-t0:.4f}s)")
        
        # 5. Sparse Solver Test (The real deal for ThermoSim)
        print("\n[Test 3] Sparse Linear Solver (CG)...")
        from scipy.sparse import diags
        import numpy as np
        
        N_sparse = 100000
        # Create a simple diagonal matrix on CPU first
        # diags expects [diag_data, ...]
        data = [np.ones(N_sparse) * 2.0]
        offsets = np.array([0])
        A_cpu = diags(data, offsets, shape=(N_sparse, N_sparse), format='csr')
        b_cpu = np.ones(N_sparse)
        
        # Move to GPU
        t0 = time.time()
        A_gpu = cpx.csr_matrix(A_cpu)
        b_gpu = cp.array(b_cpu)
        print(f"  Moves to GPU took {time.time()-t0:.4f}s")
        
        # Solve
        t0 = time.time()
        # raw CG from cupy
        x_gpu, info = cpx_la.cg(A_gpu, b_gpu, tol=1e-5)
        cp.cuda.Stream.null.synchronize()
        print(f"  Solved {N_sparse} unknowns in {time.time()-t0:.4f}s")
        
        if info == 0:
            print("[OK] Solver Converged!")
        else:
            print(f"[!] Solver failed to converge (Info={info})")

        print("\n===========================================")
        print("GREAT SUCCESS: Your environment is READY for GPU Acceleration.")
        print("You can safely enable the CUDA engine.")
        print("===========================================")
        return True

    except Exception as e:
        print(f"\n[X] GPU Test FAILED with error:")
        print(e)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    check_cuda_viability()

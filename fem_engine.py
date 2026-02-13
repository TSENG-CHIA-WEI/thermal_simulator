
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from mesh_core import Mesh3D

class Hex8Element:
    def __init__(self):
        val = 1.0 / np.sqrt(3.0)
        self.gps = []
        self.weights = []
        for k in [-val, val]:
            for j in [-val, val]:
                for i in [-val, val]:
                    self.gps.append([i, j, k])
                    self.weights.append(1.0)
        self.gps = np.array(self.gps)
        self.weights = np.array(self.weights)
        
    def local_derivatives(self, xi, eta, zeta):
        signs = np.array([
            [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
            [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
        ])
        vals = np.zeros((3, 8))
        for i in range(8):
            si, sj, sk = signs[i]
            vals[0, i] = 0.125 * si * (1 + sj*eta) * (1 + sk*zeta)
            vals[1, i] = 0.125 * sj * (1 + si*xi) * (1 + sk*zeta)
            vals[2, i] = 0.125 * sk * (1 + si*xi) * (1 + sj*eta)
        return vals

    def shape_functions(self, xi, eta, zeta):
        signs = np.array([
            [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
            [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
        ])
        N = np.zeros(8)
        for i in range(8):
            si, sj, sk = signs[i]
            N[i] = 0.125 * (1 + si*xi) * (1 + sj*eta) * (1 + sk*zeta)
        return N

class ThermalSolver3D:
    def __init__(self, mesh: Mesh3D, material_props: dict):
        self.mesh = mesh
        self.materials = material_props
        self.elem = Hex8Element()
        self.K_global = None
        self.F_vec = None
        self.active_indices = np.where(self.mesh.active_mask.flatten())[0]
        self.num_active = len(self.active_indices)
        self.num_active = len(self.active_indices)
        self.t_ambient = 25.0
        self.applied_power_watts = 0.0
        
        # Build Elements Array Once
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
        # Generate N_idx grid: (nz, ny, nx)
        N_idx = np.arange(self.mesh.num_nodes).reshape(nz, ny, nx)
        
        # Hex8 connectivity:
        # 0:(x,y,z), 1:(x+1,y,z), 2:(x+1,y+1,z), 3:(x,y+1,z)
        # 4:(x,y,z+1)...
        n0 = N_idx[:-1, :-1, :-1].flatten() 
        n1 = N_idx[:-1, :-1, 1:].flatten()
        n2 = N_idx[:-1, 1:, 1:].flatten()
        n3 = N_idx[:-1, 1:, :-1].flatten()
        n4 = N_idx[1:, :-1, :-1].flatten()
        n5 = N_idx[1:, :-1, 1:].flatten()
        n6 = N_idx[1:, 1:, 1:].flatten()
        n7 = N_idx[1:, 1:, :-1].flatten()
        
        all_elems = np.stack([n0, n1, n2, n3, n4, n5, n6, n7], axis=1)
        self.elements = all_elems[self.active_indices] # (N_act, 8)
        
        # Build dense coords access
        gv_z, gv_y, gv_x = np.meshgrid(self.mesh.z_grid, self.mesh.y_grid, self.mesh.x_grid, indexing='ij')
        self.all_nodes_coords = np.stack([gv_x.flatten(), gv_y.flatten(), gv_z.flatten()], axis=1)

    def set_ambient(self, t_amb):
        self.t_ambient = t_amb

    def assemble(self):
        print(f"Assembling Sparse System ({self.num_active} active elements)...")
        el_coords = self.all_nodes_coords[self.elements]
        
        # Materials (Anisotropic)
        all_mats = self.mesh.material_ids.flatten()
        active_mats = all_mats[self.active_indices]
        
        # Prepare K vectors (Nx, Ny, Nz) -> (N_active, 3)
        k_vecs = np.zeros((self.num_active, 3))
        
        for i, mid in enumerate(active_mats):
            mat = self.materials.get(mid, None)
            if mat:
                k_vecs[i, 0] = mat.kx
                k_vecs[i, 1] = mat.ky
                k_vecs[i, 2] = mat.kz
            else:
                k_vecs[i, :] = 1.0 # Default
        
        import time
        t_start = time.time()

        # --- CHUNKED ASSEMBLY ---
        # Process in chunks to keep peak RAM low
        # 6.75M elements * 8*8 * 8bytes = ~3.5GB per array.
        # We want to avoid holding Ke_all (3.5GB) + rows (3.5GB) + cols (3.5GB) at once.
        
        CHUNK_SIZE = 250000 
        num_chunks = (self.num_active + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        self.assembly_rows = []
        self.assembly_cols = []
        self.assembly_data = []
        
        print(f"  Processing {self.num_active} elements in {num_chunks} chunks...")

        for chunk_idx in range(num_chunks):
            start = chunk_idx * CHUNK_SIZE
            end = min((chunk_idx + 1) * CHUNK_SIZE, self.num_active)
            n_chunk = end - start
            
            # Slice Data for this chunk
            chunk_el_coords = el_coords[start:end] # (n_chunk, 8, 3)
            chunk_k_vecs = k_vecs[start:end]       # (n_chunk, 3)
            chunk_elements = self.elements[start:end] # (n_chunk, 8)
            
            # Compute Ke for chunk
            Ke_chunk = np.zeros((n_chunk, 8, 8))
            
            for gp in self.elem.gps:
                xi, eta, zeta = gp
                dN = self.elem.local_derivatives(xi, eta, zeta)
                
                # J = dN * x
                J = np.einsum('ik,ekj->eij', dN, chunk_el_coords)
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                
                B = np.matmul(invJ, dN) # (n_chunk, 3, 8)
                
                # BT * D * B
                BTDB = np.einsum('exi,ex,exj->eij', B, chunk_k_vecs, B)
                Ke_chunk += BTDB * detJ[:, None, None]
            
            # Create COO vectors for chunk
            rows = np.broadcast_to(chunk_elements[:, :, None], (n_chunk, 8, 8)).flatten()
            cols = np.broadcast_to(chunk_elements[:, None, :], (n_chunk, 8, 8)).flatten()
            data = Ke_chunk.flatten()
            
            # Append and Release Memory immediately
            self.assembly_rows.append(rows)
            self.assembly_cols.append(cols)
            self.assembly_data.append(data)
            
            # Explicit delete to help GC
            del Ke_chunk, rows, cols, data, B, BTDB, J, invJ
        
        # 2. Void Nodes Stabilization
        Nn = self.mesh.num_nodes
        self.active_node_mask = np.zeros(Nn, dtype=bool)
        unique_active_nodes = np.unique(self.elements)
        self.active_node_mask[unique_active_nodes] = True
        
        void_indices = np.where(~self.active_node_mask)[0]
        
        self.F_vec = np.zeros(Nn)
        
        if len(void_indices) > 0:
            # Add identity for void nodes
            self.assembly_rows.append(void_indices)
            self.assembly_cols.append(void_indices)
            self.assembly_data.append(np.ones(len(void_indices)))
            
            self.F_vec[void_indices] = self.t_ambient * 1.0
            
        # 3. Tiny Stabilization (Diagonal 1e-20)
        all_nodes = np.arange(Nn)
        self.assembly_rows.append(all_nodes)
        self.assembly_cols.append(all_nodes)
        self.assembly_data.append(np.ones(Nn) * 1e-20)
        self.F_vec += 1e-20 * self.t_ambient

        dt_assem = time.time() - t_start
        print(f"  Assembly (Body) finished in {dt_assem:.4f} seconds.")
        return dt_assem

    def finalize_assembly(self):
        """Builds the final K_global from accumulated buffers."""
        print("  Finalizing Sparse Matrix Assembly...")
        full_rows = np.concatenate(self.assembly_rows)
        full_cols = np.concatenate(self.assembly_cols)
        full_data = np.concatenate(self.assembly_data)
        
        Nn = self.mesh.num_nodes
        self.K_global = sp.coo_matrix((full_data, (full_rows, full_cols)), shape=(Nn, Nn)).tocsr()
        
        # Clear buffers to free RAM
        self.assembly_rows = None
        self.assembly_cols = None
        self.assembly_data = None
        print(f"  Matrix Assembled with {len(full_data)} entries.")

    def apply_neumann_load(self, power_map_dense):
        p_active = power_map_dense[self.active_indices]
        p_node = p_active / 8.0 # Lumped
        np.add.at(self.F_vec, self.elements.flatten(), np.repeat(p_node, 8))
        self.applied_power_watts += np.sum(p_active)

    def apply_dirichlet(self, node_indices, temp_val):
        if len(node_indices) == 0: return
        penalty = 1e14
        
        # DELAYED ASSEMBLY: Diagonal penalty
        self.assembly_rows.append(node_indices)
        self.assembly_cols.append(node_indices)
        self.assembly_data.append(np.ones(len(node_indices)) * penalty)
        
        self.F_vec[node_indices] += penalty * temp_val

    def apply_convection(self, elem_dense_indices, face, h_val, t_ref):
        """
        Applies Robin BC: q = h(T - T_ref)
        """
        dense_to_active = np.full(self.mesh.num_elements_dense, -1, dtype=int)
        dense_to_active[self.active_indices] = np.arange(self.num_active)
        
        relevant_active_indices = dense_to_active[elem_dense_indices]
        valid_mask = relevant_active_indices >= 0
        local_ids = relevant_active_indices[valid_mask]
        
        if len(local_ids) == 0:
            return
            
        print(f"Applying Convection ({face}) to {len(local_ids)} elements. h={h_val}, T={t_ref}")
        
        val = 1.0/np.sqrt(3.0)
        gps2d = [(-val, -val), (val, -val), (val, val), (-val, val)] 
        
        fixed_axis = 0 
        fixed_val = 0
        
        if face == 'top':    fixed_axis=2; fixed_val=1
        if face == 'bottom': fixed_axis=2; fixed_val=-1
        if face == 'east':   fixed_axis=0; fixed_val=1
        if face == 'west':   fixed_axis=0; fixed_val=-1
        if face == 'north':  fixed_axis=1; fixed_val=1
        if face == 'south':  fixed_axis=1; fixed_val=-1
        
        active_el_coords = self.all_nodes_coords[self.elements[local_ids]] # (N_face, 8, 3)
        
        Ke_surf = np.zeros((len(local_ids), 8, 8))
        Fe_surf = np.zeros((len(local_ids), 8))
        
        for u, v in gps2d:
            if fixed_axis == 2: xi, eta, zeta = u, v, fixed_val
            elif fixed_axis == 1: xi, eta, zeta = u, fixed_val, v
            else: xi, eta, zeta = fixed_val, u, v
            
            N = self.elem.shape_functions(xi, eta, zeta)
            dN = self.elem.local_derivatives(xi=xi, eta=eta, zeta=zeta)
            
            J3D = np.einsum('ik,ekj->eij', dN, active_el_coords) # (N_face, 3, 3)
            
            if fixed_axis == 2: # Top/Bottom
                t1 = J3D[:, :, 0] # d/dxi
                t2 = J3D[:, :, 1] # d/deta
            elif fixed_axis == 1: # North/South
                t1 = J3D[:, :, 0] # d/dxi
                t2 = J3D[:, :, 2] # d/dzeta
            else: # East/West
                t1 = J3D[:, :, 1]
                t2 = J3D[:, :, 2]
                
            normal = np.cross(t1, t2, axis=1) # (N_face, 3)
            dA = np.linalg.norm(normal, axis=1) # (N_face,)
            
            NTN = np.outer(N, N) # (8,8) 
            Ke_surf += NTN[None, :, :] * (h_val * dA)[:, None, None]
            Fe_surf += N[None, :] * (h_val * t_ref * dA)[:, None]
            
        rows = np.broadcast_to(self.elements[local_ids][:, :, None], (len(local_ids), 8, 8)).flatten()
        cols = np.broadcast_to(self.elements[local_ids][:, None, :], (len(local_ids), 8, 8)).flatten()
        
        # DELAYED ASSEMBLY: Append to buffers
        self.assembly_rows.append(rows)
        self.assembly_cols.append(cols)
        self.assembly_data.append(Ke_surf.flatten())
        
        np.add.at(self.F_vec, self.elements[local_ids].flatten(), Fe_surf.flatten())

    def apply_surface_power_load(self, elem_dense_indices, face, power_vals):
        """
        Applies a fixed Total Power (Watts) over each element's surface.
        Since shape functions integrate over area: Int(N * q * dA).
        We calculate q = Power / Integrated_Area_Approximation.
        Or simpler: Since we know Total Power 'P' for this element-face,
        and FEM assumes bi-linear shape functions (N sum to 1 effectively over the surface), 
        we can distribute P to the 4 face nodes weighted by their area contribution?
        
        Proper way: 
        q = P_element / Area_element.
        Load_vector = q * Int(N dA)
        Int(N dA) = Area_element / 4 (for rectangular quad).
        So Load_vector = (P/A) * (A/4) = P/4.
        
        So we just take the Power for the element and divide by 4 (for 4 nodes on the face).
        Simple!
        """
        dense_to_active = np.full(self.mesh.num_elements_dense, -1, dtype=int)
        dense_to_active[self.active_indices] = np.arange(self.num_active)
        
        relevant_active_indices = dense_to_active[elem_dense_indices]
        valid_mask = relevant_active_indices >= 0
        local_ids = relevant_active_indices[valid_mask]
        
        if len(local_ids) == 0: return
        
        powers = power_vals[valid_mask] # Watts per element
        
        # Identify face nodes logic (similar to convection but simpler)
        # We need to find the node indices for the specified face
        
        fixed_axis = 0 
        fixed_val = 0
        
        # Node Map of local element (0..7) to face (0..3)
        # Top (z=1): 4,5,6,7. Bot (z=-1): 0,1,2,3
        # North (y=1): 2,3,6,7. South (y=-1): 0,1,4,5
        # East (x=1): 1,2,5,6. West (x=-1): 0,3,4,7
        
        face_node_map = []
        if face == 'top':    face_node_map = [4,5,6,7]
        if face == 'bottom': face_node_map = [0,1,2,3]
        if face == 'east':   face_node_map = [1,2,5,6]   
        if face == 'west':   face_node_map = [0,3,4,7]
        if face == 'north':  face_node_map = [2,3,6,7]
        if face == 'south':  face_node_map = [0,1,4,5]
        
        if not face_node_map: return
        
        # Distribute P/4 to each of the 4 nodes
        p_per_node = powers / 4.0
        
        # We need the global Node IDs for these face nodes
        target_elements = self.elements[local_ids] # (N, 8) global node IDs
        
        # For each face node j in 0..3:
        applied_total = 0.0
        for i_local in range(4):
            node_col_idx = face_node_map[i_local]
            global_nodes = target_elements[:, node_col_idx]
            np.add.at(self.F_vec, global_nodes, p_per_node)
            applied_total += np.sum(p_per_node)
            
        print(f"  [Solver] Surface Power Applied: {applied_total:.4f} W to {len(global_nodes)} elements.")
        self.applied_power_watts += applied_total


    def compute_convection_power(self, elem_dense_indices, face, h_val, t_ref, T_sol):
        """
        Computes total heat loss (Watts) for a given surface set.
        P_loss = integral( h * (T - T_ref) ) dA
        """
        dense_to_active = np.full(self.mesh.num_elements_dense, -1, dtype=int)
        dense_to_active[self.active_indices] = np.arange(self.num_active)
        
        relevant_active_indices = dense_to_active[elem_dense_indices]
        valid_mask = relevant_active_indices >= 0
        local_ids = relevant_active_indices[valid_mask]
        
        if len(local_ids) == 0:
            return 0.0
            
        val = 1.0/np.sqrt(3.0)
        gps2d = [(-val, -val), (val, -val), (val, val), (-val, val)] 
        
        fixed_axis = 0 
        fixed_val = 0
        if face == 'top':    fixed_axis=2; fixed_val=1
        if face == 'bottom': fixed_axis=2; fixed_val=-1
        if face == 'east':   fixed_axis=0; fixed_val=1
        if face == 'west':   fixed_axis=0; fixed_val=-1
        if face == 'north':  fixed_axis=1; fixed_val=1
        if face == 'south':  fixed_axis=1; fixed_val=-1
        
        active_el_coords = self.all_nodes_coords[self.elements[local_ids]] # (N_face, 8, 3)
        # Get Temperature at nodes for these elements
        # (N_face, 8)
        node_indices = self.elements[local_ids]
        T_nodes = T_sol[node_indices]
        
        total_power = 0.0
        
        for u, v in gps2d:
            if fixed_axis == 2: xi, eta, zeta = u, v, fixed_val
            elif fixed_axis == 1: xi, eta, zeta = u, fixed_val, v
            else: xi, eta, zeta = fixed_val, u, v
            
            N = self.elem.shape_functions(xi, eta, zeta) # (8,)
            dN = self.elem.local_derivatives(xi=xi, eta=eta, zeta=zeta) # (3, 8)
            
            J3D = np.einsum('ik,ekj->eij', dN, active_el_coords) # (N_face, 3, 3)
            
            if fixed_axis == 2: t1, t2 = J3D[:, :, 0], J3D[:, :, 1]
            elif fixed_axis == 1: t1, t2 = J3D[:, :, 0], J3D[:, :, 2]
            else: t1, t2 = J3D[:, :, 1], J3D[:, :, 2]
                
            normal = np.cross(t1, t2, axis=1) # (N_face, 3)
            dA = np.linalg.norm(normal, axis=1) # (N_face,)
            
            # Interpolate T at integration point
            # T_ip = sum(N_i * T_i)
            T_ip = np.einsum('j,ej->e', N, T_nodes) # (N_face,)
            
            # q = h * (T_ip - T_ref)
            # Power = q * dA (weight=1.0)
            
            p_contribution = h_val * (T_ip - t_ref) * dA
            total_power += np.sum(p_contribution)
            
        return total_power

    def solve(self):
        import time
        t0 = time.time()
        
        # Memory Reporting
        self.mem_mb = (self.K_global.data.nbytes + self.K_global.indices.nbytes + self.K_global.indptr.nbytes) / (1024*1024)
        print(f"Solving Linear System...")
        print(f"  Matrix Memory: {self.mem_mb:.2f} MB")
        
        # Convergence History Tracking
        self.residual_history = []
        
        # GPU Auto-Detection
        use_gpu = False
        try:
            # Inject CUDA Paths (Robustness for Windows)
            import os
            cuda_candidates = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
            ]
            for p in cuda_candidates:
                if os.path.exists(p) and p not in os.environ["PATH"]:
                    os.environ["PATH"] += os.pathsep + p
            
            import cupy as cp
            import cupyx.scipy.sparse as cpsp
            import cupyx.scipy.sparse.linalg as cplinalg
            
            # Verify basic functionality
            dev = cp.cuda.Device(0)
            t_test = cp.array([1.0]) * 2.0
            
            use_gpu = True
            print("  Device: GPU (CuPy Detected & Verified)")
        except Exception as e:
            # Catch ImportError, OSError (DLL failed), or CUDA runtime errors
            print(f"  Device: CPU (CuPy not found or failed: {e})")

        if use_gpu:
            try:
                # Transfer to GPU
                print("  Transferring to GPU...")
                K_gpu = cpsp.csr_matrix(self.K_global)
                F_gpu = cp.array(self.F_vec)
                
                # Preconditioner (Jacobi) on GPU
                diag_K = K_gpu.diagonal()
                diag_K[diag_K == 0] = 1.0
                M_gpu = cpsp.diags(1.0 / diag_K)
                
                # Callback for GPU (CuPy)
                def callback_gpu(xk):
                    r = F_gpu - K_gpu @ xk
                    res_norm = float(cp.linalg.norm(r))
                    self.residual_history.append(res_norm)
                
                # Solve using GPU CG
                T_gpu, info = cplinalg.cg(K_gpu, F_gpu, M=M_gpu, tol=1e-6, atol=1e-8, maxiter=2000, callback=callback_gpu)
                
                # Transfer back
                T_sol = cp.asnumpy(T_gpu)
                
            except Exception as e:
                print(f"  GPU Failed ({e}). Falling back to CPU...")
                use_gpu = False

        if not use_gpu:
            # CPU Fallback (Scipy)
            # Preconditioner helps convergence (Jacobi)
            diag_K = self.K_global.diagonal()
            diag_K[diag_K == 0] = 1.0 
            M = sp.diags(1.0 / diag_K)
            
            # Callback for CPU (Scipy)
            def callback_cpu(xk):
                r = self.F_vec - self.K_global @ xk
                res_norm = float(np.linalg.norm(r))
                self.residual_history.append(res_norm)
            
            T_sol, info = sp.linalg.cg(self.K_global, self.F_vec, M=M, rtol=1e-6, atol=1e-8, maxiter=2000, callback=callback_cpu)
        
        dt = time.time() - t0
        self.iterations = info if info > 0 else "Converged"
        
        mode_str = "GPU-Accelerated" if use_gpu else "CPU-Sparse"
        if info == 0:
            print(f"  Result: [{mode_str}] Converged in {dt:.4f} seconds.")
            self.converged = True
        else:
            print(f"  Warning: [{mode_str}] Convergence not reached in {info} iterations. ({dt:.4f}s)")
            self.converged = False
            
        return T_sol

    def compute_flux(self, T_field):
        dN_center = self.elem.local_derivatives(0, 0, 0)
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
        el_coords = self.all_nodes_coords[self.elements]
        
        J = np.einsum('ik,ekj->eij', dN_center, el_coords)
        invJ = np.linalg.inv(J)
        B = np.matmul(invJ, dN_center)
        
        T_el = T_field[self.elements]
        grad_T = np.einsum('eij,ej->ei', B, T_el)
        
        # Anisotropic Flux
        all_mats = self.mesh.material_ids.flatten()
        active_mats = all_mats[self.active_indices]
        
        k_vecs = np.zeros((self.num_active, 3))
        for i, mid in enumerate(active_mats):
            mat = self.materials.get(mid, None)
            if mat:
                k_vecs[i, 0] = mat.kx
                k_vecs[i, 1] = mat.ky
                k_vecs[i, 2] = mat.kz
            else:
                k_vecs[i, :] = 1.0 # Default
        
        # Flux = -K * gradT (element-wise vector mult)
        return -grad_T * k_vecs

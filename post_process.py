import numpy as np
import scipy.interpolate
import pyvista as pv

class SliceExporter:
    def __init__(self, mesh, T_field):
        """
        mesh: Mesh3D object
        T_field: 1D numpy array of temperatures (nodes)
        """
        self.mesh = mesh
        self.T_field = T_field
        
    def export_z_slice(self, z_coord, resolution, filename="slice_z.csv"):
        """
        Exports a 2D grid slice at z=z_coord.
        resolution: (Nx, Ny) tuple
        """
        print(f"Exporting Z-Slice at {z_coord}m (Res: {resolution})...")
        
        # Grid Bounds
        x_min, x_max = self.mesh.x_grid[0], self.mesh.x_grid[-1]
        y_min, y_max = self.mesh.y_grid[0], self.mesh.y_grid[-1]
        
        # Create Regular Grid
        xi = np.linspace(x_min, x_max, resolution[0])
        yi = np.linspace(y_min, y_max, resolution[1])
        xv, yv = np.meshgrid(xi, yi, indexing='xy') # (Ny, Nx)
        
        # Construct dense coordinate arrays for entire mesh (Nodes)
        # Assuming C-ordering (Standard)
        # Grid: z, y, x properties
        x_nodes = self.mesh.x_grid
        y_nodes = self.mesh.y_grid
        z_nodes = self.mesh.z_grid
        
        # Optimization: Find Z layer nearest to request.
        z_idx = np.argmin(np.abs(z_nodes - z_coord))
        z_actual = z_nodes[z_idx]
        print(f"  Mapping requested Z={z_coord} to nearest node plane Z={z_actual}")
        
        nx, ny, nz = len(x_nodes), len(y_nodes), len(z_nodes)
        
        # T_field is flattened active nodes? Or ALL nodes?
        # Solver: K_global shape (Nn, Nn). So T is size Nn.
        # Dense reshape: (Nz, Ny, Nx) usually if C-order of np.meshgrid(z, y, x, indexing='ij')
        
        # Let's try to infer shape.
        try:
             # Meshgrid ij order: Z comes first? 
             # mesh_core: gv_z, gv_y, gv_x = np.meshgrid(.., indexing='ij')
             # So array[k, j, i]
             T_3d = self.T_field.reshape((nz, ny, nx)) 
        except ValueError:
             print("  Warning: Reshape failed. T_field size mismatch.")
             return
        
        # Extract Slice
        T_slice = T_3d[z_idx, :, :] # (Ny, Nx) -> Y is axis 0, X is axis 1
        
        # Interpolate to requested resolution
        interp = scipy.interpolate.RegularGridInterpolator((y_nodes, x_nodes), T_slice, bounds_error=False, fill_value=None)
        
        # Query points: (y, x)
        pts = list(zip(yv.flatten(), xv.flatten()))
        out_flat = interp(pts)
        out_grid = out_flat.reshape((resolution[1], resolution[0]))
        
        # Save
        np.savetxt(filename, out_grid, delimiter=",", fmt="%.4f")
        print(f"  Slice saved to {filename}")

class VTKExporter:
    @staticmethod
    def export_vtu(filename, mesh, temperature_field, flux_field=None, power_field_dense=None):
        """
        Exports the UnstructuredGrid to VTK/VTU using PyVista.
        We only export ACTIVE elements to save space and show true geometry.
        """
        
        # 1. Identify Active Elements
        # mesh.active_mask shape: (nz-1, ny-1, nx-1)
        # We need to construct the connectivity for these active cells.
        
        # Grid Dimensions (Nodes)
        nx_n = len(mesh.x_grid)
        ny_n = len(mesh.y_grid)
        nz_n = len(mesh.z_grid)
        
        # Flattened Node Coordinates
        # Z-slow, Y, X-fast (C-order)
        # Z, Y, X = np.meshgrid(mesh.z_grid, mesh.y_grid, mesh.x_grid, indexing='ij')
        # points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        # Optimized:
        xm, ym, zm = np.meshgrid(mesh.x_grid, mesh.y_grid, mesh.z_grid, indexing='xy') 
        # Wait, indexing='xy' -> dim1=y, dim2=x. Z?
        # Let's stick to 'ij' to match mesh_core
        gz, gy, gx = np.meshgrid(mesh.z_grid, mesh.y_grid, mesh.x_grid, indexing='ij')
        points = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=1)
        
        # 2. Get Indices of Active Elements
        # active_mask is (Nz-1, Ny-1, Nx-1)
        # We find indices (k, j, i) of active cells
        active_indices = np.argwhere(mesh.active_mask) # Shape (N_active, 3) -> [z_idx, y_idx, x_idx]
        
        if len(active_indices) == 0:
            print("Warning: No active elements to export.")
            return

        # 3. Construct Hexahedron Connectivity (8 nodes per cell)
        # Node Index = k*(ny*nx) + j*nx + i
        # Offsets for standard Hex:
        # 0: (0,0,0), 1: (0,0,1), 2: (0,1,0), 3: (0,1,1)... X fast?
        # VTK Hex Node Order:
        # Bottom: 000, 100, 110, 010 (CCW)
        # Top:    001, 101, 111, 011
        # Our indices (k, j, i) mapping to nodes:
        # Base: (k, j, i)
        
        # Let's verify 'ij' meshgrid flattening order.
        # Z is axis 0 (slow), Y is 1, X is 2 (fast).
        # Node idx = k*(ny*nx) + j*(nx) + i
        
        k = active_indices[:, 0]
        j = active_indices[:, 1]
        i = active_indices[:, 2]
        
        # Node strides
        sx = 1
        sy = nx_n
        sz = nx_n * ny_n
        
        n0 = k*sz + j*sy + i*sx
        n1 = k*sz + j*sy + (i+1)*sx
        n2 = k*sz + (j+1)*sy + (i+1)*sx
        n3 = k*sz + (j+1)*sy + i*sx
        
        n4 = (k+1)*sz + j*sy + i*sx
        n5 = (k+1)*sz + j*sy + (i+1)*sx
        n6 = (k+1)*sz + (j+1)*sy + (i+1)*sx
        n7 = (k+1)*sz + (j+1)*sy + i*sx
        
        # Stack (N_active, 8)
        cells = np.stack([n0, n1, n2, n3, n4, n5, n6, n7], axis=1)
        
        # 4. Create PyVista UnstructuredGrid
        # Cell Types: VTK_HEXAHEDRON = 12
        cell_types = np.full(len(cells), 12, dtype=np.uint8)
        
        # PV expects cells array as [n_nodes, node_0, ... node_n, n_nodes, ...]
        # Prepend '8' to each cell
        cells_pv = np.hstack([np.full((len(cells), 1), 8), cells])
        cells_pv = cells_pv.flatten()
        
        grid = pv.UnstructuredGrid(cells_pv, cell_types, points)
        
        # 5. Add Data
        # A. Point Data (Temperature)
        grid.point_data["Temperature"] = temperature_field
        
        # B. Cell Data (Materials)
        # mesh.material_ids matches active_mask shape
        # Flatten active materials
        mats_flat = mesh.material_ids[mesh.active_mask]
        mats_flat = mesh.material_ids[mesh.active_mask]
        grid.cell_data["MaterialID"] = mats_flat
        grid.cell_data["BoxID"] = mesh.box_ids[mesh.active_mask]
        
        # C. Flux (Vector) - Point or Cell?
        # Solver computes flux at Centroids usually.
        # If flux_field is given and shape matches active cells
        if flux_field is not None:
             # Assuming flux is (N_active, 3)
             if len(flux_field) == grid.n_cells:
                  grid.cell_data["HeatFlux"] = flux_field
        
        # D. Power
        if power_field_dense is not None:
             # Assume flattened dense field matching centroids?
             # Or dense array?
             # Let's skip dense power for now unless we need it
             pass
             
        # Save
        grid.save(filename)
        print(f"  VTK saved to {filename} ({grid.n_cells} cells)")

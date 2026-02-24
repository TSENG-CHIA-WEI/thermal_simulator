
import numpy as np

class Mesh3D:
    def __init__(self, x_grid, y_grid, z_grid, active_mask, material_ids, box_ids):
        """
        active_mask: boolean array (nz-1, ny-1, nx-1) indicating if element exists.
        material_ids: array (nz-1, ny-1, nx-1)
        
        We store full dense arrays for simplified indexing, but Solver will compressed them.
        """
        self.x_grid = x_grid.astype(np.float32)
        self.y_grid = y_grid.astype(np.float32)
        self.z_grid = z_grid.astype(np.float32)
        self.active_mask = active_mask # Boolean
        self.material_ids = material_ids.astype(np.int32)
        self.box_ids = box_ids.astype(np.int32) # Maps element to index in sim_config.boxes
        
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

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import block_diag

def get_basis_fct_idx_for_element(z):
    if z <= 2:
        offset = 1 + 5 * (z - 1)
        return [i for i in range(offset, offset + 5)]
    else:
        offset = 1 + 2 * 5 + (z - 3) * 14
        return [i for i in range(offset, offset + 14)]
    
class DataContainer:
    def __init__(self, filename, target, cutoff):
        self.cutoff = cutoff
        data_dict = np.load(filename, allow_pickle=True)
        for key in ["R", "densities", "coords", "N_coords", "corrs"]:
            setattr(self, key, np.array(data_dict[key]))
        self.idx = np.arange(len(self.R))
        self.N_coords_idx = np.concatenate([[0], np.cumsum(self.N_coords)])
        

        self.target = data_dict[target[0]]


    def __len__(self):
        return self.R.shape[0]
    
    def __getitem__(self, idx):
        if type(idx) is int or type(idx) is np.int64:
            idx = [idx]
        data = {}
        data["id"] = self.id[idx]
        data["R"] = self.R[idx]
        data["densities"] = self.densities[idx]
        #data["coords"] = np.zeros((np.sum(self.N_coords), 3), dtype=np.float32)
        adj_matrices = []
        
        coords = []
        densities = []
        corrs = []

        for id in self.idx:
            n_coords = self.N_coords[id]
            n_start = self.N_coords_idx[id]
            n_end = n_start + n_coords

            coords += self.coords[n_start:n_end]
            densities += self.densities[n_start:n_end]
            corrs += self.corrs[n_start:n_end]

            adj_matrix = np.linalg.norm((coords[None, :, :] - coords[:, None, :]), axis=-1)
            adj_matrices.append(adj_matrix <= self.cutoff)

        data["coords"] = np.array(coords)
        data["densities"] = np.array(densities)
        total_adj_matrix = block_diag(*adj_matrices)
        data["adj_matrix"] = total_adj_matrix
        # hard-coded
        data["target"] = corrs
        return data
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

        for key in ["R", "densities", "coords", "corrs"]:
            setattr(self, key, np.array(data_dict[key]))
        self.idx = np.arange(len(self.R))
        #self.N_coords_idx = np.concatenate([[0], np.cumsum(self.N_coords)])

        #self.N_corrs_idx = np.concatenate([[0], np.cumsum(self.N_coords ** 2)])
        

        self.target = data_dict[target[0]]


    def __len__(self):
        return self.R.shape[0]
    
    def __getitem__(self, idx):
        if type(idx) is int or type(idx) is np.int64:
            idx = [idx]
        data = {}
        data["id"] = self.idx[idx]
        data["densities"] = self.densities[idx]
        #data["coords"] = np.zeros((np.sum(self.N_coords), 3), dtype=np.float32)
        # adj_matrices = []
        
        # coords = []
        # densities = []
        # corrs = []

        # for id in self.idx:
        #     n_coords = self.N_coords[id]            
        #     n_coords_start = self.N_coords_idx[id]
        #     n_coords_end = n_coords_start + n_coords
            
        #     n_corrs = n_coords ** 2
        #     n_corrs_start = self.N_corrs_idx[id]
        #     n_corrs_end = n_corrs_start + n_corrs

        #     coords += self.coords[n_coords_start:n_coords_end]
        #     densities += self.densities[n_coords_start:n_coords_end]
        #     corrs += self.corrs[n_corrs_start:n_corrs_end]

        #     adj_matrix = np.linalg.norm((coords[None, :, :] - coords[:, None, :]), axis=-1)
        #     adj_matrices.append(adj_matrix <= self.cutoff)
        # total_adj_matrix = block_diag(*adj_matrices)

        data["coords"] = self.coords[idx]
        data["corrs"] = self.corrs[idx]
        data["adj_matrix"] = np.linalg.norm((self.coords[idx][:, None, :, :] - self.coords[idx][:, :, None, :]), axis=-1)
        # hard-coded
        data["target"] = self.corrs[idx]
        return data
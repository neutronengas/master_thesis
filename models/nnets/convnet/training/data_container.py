import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import block_diag

    
class DataContainer:
    def __init__(self, filename, target):
        data_dict = np.load(filename, allow_pickle=True)
        for key in ["densities", "coords", "corrs"]:
            setattr(self, key, np.array(data_dict[key]))
        #self.N_coords_idx = np.concatenate([[0], np.cumsum(self.N_coords)])

        #self.N_corrs_idx = np.concatenate([[0], np.cumsum(self.N_coords ** 2)])
        

        self.target = data_dict[target[0]]
        # merge the dimensions enumerating samples and enumerating squares within a rectangle respectively
        sample_dim = self.target.shape[0]
        square_dim = self.target.shape[1]
        new_target_shape = (sample_dim * square_dim,) + tuple(self.target.shape[2:])
        self.target = self.target.reshape(new_target_shape)
        new_densities_shape = (sample_dim * square_dim,) + tuple(self.densities.shape[2:])
        self.densities = self.densities.reshape(new_densities_shape)[:, :, :, np.newaxis]
        print(self.densities.shape)
        # r1 in rho(r1, r2) is always chosen to be the center of the square
        self.target = self.target[:, 3, 3, :, :] 


        self.idx = np.arange(len(self.densities))


    def __len__(self):
        return self.densities.shape[0]
    
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

        data["target"] = self.target[idx][:, :, :, np.newaxis]
        return data
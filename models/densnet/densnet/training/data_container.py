import numpy as np
from scipy.spatial.distance import cdist

def get_basis_fct_idx_for_element(z):
    if z <= 2:
        offset = 1 + 5 * (z - 1)
        return [i for i in range(offset, offset + 5)]
    else:
        offset = 1 + 2 * 5 + (z - 3) * 14
        return [i for i in range(offset, offset + 14)]
    
class DataContainer:
    def __init__(self, filename, target, cutoff):
        data_dict = np.load(filename, allow_pickle=True)
        for key in ["R", "densities", "corrs", "coords"]:
            setattr(self, key, np.array(data_dict[key]))
        self.id = np.arange(len(self.R))
        
        dist_matrix = cdist(self.coords, self.coords)
        self.neighbour_coords_idx = (dist_matrix < cutoff) * np.arange(len(self.coords))[np.newaxis, :]
        self.neighbour_coords_idx[self.neighbour_coords_idx == 0.0] = -1
        self.neighbour_coords_idx = self.neighbour_coords_idx.astype(np.float32)
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
        data["neighbour_coords_idx"] = self.neighbour_coords_idx
        #data["corrs"] = self.corrs[idx]
        data["coords"] = self.coords
        data["target"] = self.target[idx]
        return data
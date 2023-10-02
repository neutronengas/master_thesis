import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist

class DataContainer:
    def __init__(self, filename, target, cutoff):
        self.cutoff = cutoff
        data_dict = np.load(filename, allow_pickle=True)
        for key in ["R", "densities", "corrs", "coords"]:
            setattr(self, key, np.array(data_dict[key]))
        self.id = np.arange(len(self.R))
        self.Z = np.array([[1, 1] for _ in self.id]).flatten()
        self.N = np.array([2 for _ in self.Z])
        self.R = self.R.reshape((-1, self.R.shape[-1]))
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])
        self.target = data_dict[target[0]]
        # hard-coded reshaping of the densities
        self.target = self.target.reshape(1200, 8, 5 * 5)
        self.target = self.target.reshape(1200, 8 * 5 * 5)

    def __len__(self):
        return self.id.shape[0]
    
    def _bmat_fast(self, mats):
        new_data = np.concatenate([mat.data for mat in mats])

        ind_offset = np.zeros(1 + len(mats))
        ind_offset[1:] = np.cumsum([mat.shape[0] for mat in mats])
        new_indices = np.concatenate(
            [mats[i].indices + ind_offset[i] for i in range(len(mats))])

        indptr_offset = np.zeros(1 + len(mats))
        indptr_offset[1:] = np.cumsum([mat.nnz for mat in mats])
        new_indptr = np.concatenate(
            [mats[i].indptr[i >= 1:] + indptr_offset[i] for i in range(len(mats))])
        return sp.csr_matrix((new_data, new_indices, new_indptr))
    
    def __getitem__(self, idx):
        if type(idx) is int or type(idx) is np.int64:
            idx = [idx]
        data = {}
        data["id"] = self.id[idx]
        data["target"] = self.target[idx]
        data["N"] = self.N[idx]
        #data["corrs"] = self.corrs[idx]
        data["coords"] = self.coords[idx]
        data['Z'] = np.zeros(np.sum(data['N']), dtype=np.int32)
        data['R'] = np.zeros([np.sum(data['N']), 3], dtype=np.float32)

        nend = 0
        adj_matrices = []
        for k, i in enumerate(idx):
            n = data['N'][k]  # number of atoms
            nstart = nend
            nend = nstart + n

            if self.Z is not None:
                data['Z'][nstart:nend] = self.Z[self.N_cumsum[i]:self.N_cumsum[i + 1]]

            R = self.R[self.N_cumsum[i]:self.N_cumsum[i + 1]]
            data['R'][nstart:nend] = R

            Dij = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
            adj_matrices.append(sp.csr_matrix(Dij <= self.cutoff))

        adj_matrix = self._bmat_fast(adj_matrices)
        edge_id_i_unsorted, edge_id_j_unsorted = adj_matrix.nonzero()
        edge_id_matrix = np.array([edge_id_i_unsorted, edge_id_j_unsorted])
        edge_id_matrix = edge_id_matrix[edge_id_matrix[:, 1].argsort()]
        data["edge_id_i"] = edge_id_matrix[0]
        data["edge_id_j"] = edge_id_matrix[1]
        return data
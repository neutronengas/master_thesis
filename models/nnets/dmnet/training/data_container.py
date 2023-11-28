import numpy as np
from functools import partial
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import os


class DataContainer:
    def __init__(self, filename, target, cutoff, n_coords):
        self.cutoff = cutoff
        data_dict = np.load(filename, allow_pickle=True)        
        for key in ["R", "densities", "corrs", "coords", "N_rdm"]:
            setattr(self, key, np.array(data_dict[key]))

        self.rdm_hf_values = np.array(data_dict["rdm_hf"])
        self.id = np.arange(len(self.R))
        self.Z = np.array([[1, 1] for _ in self.id]).flatten()
        self.N = np.array([2 for _ in self.Z])
        self.R = self.R.reshape((-1, self.R.shape[-1]))
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])
        self.N_rdm_cumsum = np.concatenate([[0], np.cumsum(self.N_rdm)])
        self.coeffs = np.load("./model/utils/cc-pvdz-coeffs-retry.npz", allow_pickle=True)["tensor"][:, :n_coords]
        # reshaping the input density matrix to fit the augmented orbital tensor
        # ordering of orbitals in pyscf is angular momentum, shell, spherical part
        # currently here, the ordering is shell, angular momentum, spherical part 
        # (as in m_max, max_no_orbitals_per_m, max_split_per_m)
        self.target = data_dict[target[0]]



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
        data["N_rdm"] = self.N_rdm[idx]

        #data["corrs"] = self.corrs[idx]   
        data["coords"] = np.repeat(self.coords[idx], data["N"], axis=0) # TODO
        data['Z'] = np.zeros(np.sum(data['N']), dtype=np.int32)
        data['R'] = np.zeros([np.sum(data['N']), 3], dtype=np.float32)
        data['rdm'] = np.zeros((np.sum(data["N_rdm"]), 14, 14), dtype=np.float32)
        nend = 0
        n_rdm_end = 0
        adj_matrices = []
        atom_pair_matrices = []
        for k, i in enumerate(idx):
            n = data['N'][k]  # number of atoms
            nstart = nend
            nend = nstart + n

            n_rdm = data['N_rdm'][k]
            n_rdm_start = n_rdm_end
            n_rdm_end = n_rdm_start + n_rdm

            if self.Z is not None:
                Z = self.Z[self.N_cumsum[i]:self.N_cumsum[i + 1]]
                data['Z'][nstart:nend] = Z

            rdm = self.rdm_hf_values[self.N_rdm_cumsum[i]:self.N_rdm_cumsum[i + 1]]
            data["rdm"][n_rdm_start:n_rdm_end] = rdm

            R = self.R[self.N_cumsum[i]:self.N_cumsum[i + 1]]
            data['R'][nstart:nend] = R

            Dij = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
            atom_pair_matrices.append(sp.csr_matrix(Dij + np.eye(len(Dij))))
            adj_matrices.append(sp.csr_matrix((Dij <= self.cutoff ) - np.eye(len(Dij))))

        adj_matrix = self._bmat_fast(adj_matrices)
        edge_id_i_unsorted, edge_id_j_unsorted = adj_matrix.nonzero()
        edge_id_matrix = np.array([edge_id_i_unsorted, edge_id_j_unsorted])
        edge_id_matrix = edge_id_matrix[edge_id_matrix[:, 1].argsort()]
        data["edge_id_i"] = edge_id_matrix[0]
        data["edge_id_j"] = edge_id_matrix[1]

        atom_pair_matrix = self._bmat_fast(atom_pair_matrices)
        atom_pair_i, atom_pair_j = atom_pair_matrix.nonzero()
        atom_pair_indices = np.stack([atom_pair_i, atom_pair_j]).T
        data["atom_pair_indices"] = atom_pair_indices.astype(np.int32)

        atom_pair_mol_id = np.repeat(np.arange(len(data["N"])), data["N"] ** 2)
        data["atom_pair_mol_id"] = atom_pair_mol_id.astype(np.int32)

        return data
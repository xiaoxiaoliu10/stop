import os
import pickle
import torch
import torch_geometric.utils as utils
import numpy as np


class PositionEncoding(object):
    def apply_to(self, graph):
        pe = self.compute_pe(graph)
        return pe


class LapEncoding(PositionEncoding):
    def __init__(self, dim, normalization=None):
        """
        normalization: for Laplacian None. sym or rw
        """
        self.pos_enc_dim = dim
        self.normalization = normalization

    def compute_pe(self, graph):
        laplacian = utils.get_laplacian(
            graph.edge_index.long(), normalization=self.normalization,
            num_nodes=graph.num_nodes)[0]

        EigVal, EigVec = torch.linalg.eigh(
            torch.sparse_coo_tensor(laplacian, torch.ones(laplacian.shape[1])).to_dense().to(laplacian.device))
        factor = torch.randn((1, EigVec.shape[0]))
        factor[factor >= 0] = 1
        factor[factor < 0] = -1
        EigVec *= factor

        return EigVec[:, 0:self.pos_enc_dim]


POSENCODINGS = {
    'lap': LapEncoding
}

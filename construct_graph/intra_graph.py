import numpy as np
import torch.nn as nn
from itertools import combinations
import torch
import scipy.sparse as sp
from torch_geometric.data import Data
from einops import rearrange
from construct_graph.position_encoding import POSENCODINGS


class intra_topk(nn.Module):
    def __init__(self, r, pe_ratio, dim_num, pe='lap', simi_ways='cos'):
        super(intra_topk, self).__init__()
        self.k = int(dim_num * r)
        pe_dim = int(pe_ratio * dim_num)
        self.pe = POSENCODINGS[pe](pe_dim)
        self.dim_num = dim_num
        self.simi_ways = simi_ways

    def similarity_cosine(self, output):
        inx = np.linspace(0, len(output) - 1, len(output), dtype=np.int32)  # d
        similar_matrix = torch.zeros((output.shape[1], len(output), len(output)))  # [batch_size,d,d]
        similar_matrix += torch.eye(len(output))
        for a, b in combinations(inx, 2):
            similarity = torch.div(torch.einsum('bij,bij->b', output[a], output[b]), (
                    torch.flatten(output[a], start_dim=1).norm(dim=1) * torch.flatten(output[b], start_dim=1).norm(
                dim=1)))
            similar_matrix[:, a, b] = similarity  # [batch_size,d,d]
            similar_matrix[:, b, a] = similarity
        return similar_matrix

    def euclidean_distance(self, output):
        output = rearrange(output, 'd b n h->b d (n h)')
        a = torch.cdist(output, output)
        a = -a
        mask = torch.zeros(a.shape[0], a.shape[1], a.shape[2]).to(output.device)
        mask.fill_(float('0'))
        s1, t1 = a.topk(self.k, dim=2)
        mask.scatter_(2, t1, s1.fill_(1))
        return mask

    def forward(self, output1):
        device = output1.device
        if self.simi_ways == 'cos':
            adjs = self.similarity_cosine(output1)  # [batch_size,d,d]
        elif self.simi_ways == 'eu':
            adjs = -self.euclidean_distance(output1)
        # ----------dimensional structure extractor--------------

        mask = torch.zeros((output1.shape[1], len(output1), len(output1)))
        mask.fill_(float('0'))
        s1, t1 = adjs.topk(self.k, dim=2)
        mask.scatter_(2, t1, s1.fill_(1))
        adjs = adjs * mask

        graphs = []
        for i, adj in enumerate(adjs):
            edge_index = sp.coo_matrix(adj.to_dense().cpu().detach().numpy()).tocoo()
            feature = rearrange(output1[:, i, :, :], 'd n f -> d (n f)')
            g = Data(x=feature,
                     edge_index=torch.tensor(np.array([edge_index.row, edge_index.col]), dtype=torch.long)).to(device)
            pe = self.pe.apply_to(g.cpu())
            g.x = torch.cat((g.x, pe), dim=1)
            graphs.append(g)

        return graphs


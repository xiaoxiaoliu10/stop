import scipy.sparse as sp
import torch
import torch.nn as nn
from einops import rearrange
from torch_geometric.data import Data, Batch
from construct_graph.position_encoding import POSENCODINGS
import numpy as np


class intra_topk(nn.Module):
    def __init__(self, seg_len, pe_ratio, r, node_num, pe_method='lap'):
        super(intra_topk, self).__init__()
        self.seg_len = seg_len
        self.k = int(node_num * r)
        pe_dim = int(node_num * pe_ratio)
        self.pe = POSENCODINGS[pe_method](pe_dim)

    def forward(self, x):
        batch, ts_dim, ts_len = x.shape
        if ts_len % self.seg_len != 0:
            x = x[:, :, :ts_len - ts_len % self.seg_len]
        x_embed = rearrange(x, 'b d (seg_num seg_len) -> b d seg_num seg_len', seg_len=self.seg_len)

        a = torch.einsum('bdij,bdjk->bdik', x_embed, x_embed.transpose(2, 3)) / torch.einsum('b d i, b d j -> b d i j',
                                                                                             x_embed.norm(dim=3),
                                                                                             x_embed.norm(
                                                                                                 dim=3))
        mask = torch.zeros(batch, ts_dim, x_embed.shape[2], x_embed.shape[2]).to(x.device)
        mask.fill_(float('0'))
        s1, t1 = a.topk(self.k, dim=3)
        mask.scatter_(3, t1, s1.fill_(1))
        adjs = torch.einsum('bdij,bdij->bdij', a, mask)
        adjs = rearrange(adjs, 'b d i j -> d b i j')
        graphs = []
        for i, adj in enumerate(adjs):
            # -----dimension----
            batch_graph = []
            for j in range(adj.shape[0]):
                # -----batch_size-------
                g = self.to_pygdata(adj[j], x_embed[j, i, :, :])
                abs_pe_list = self.pe.apply_to(g.cpu())
                g.x = torch.cat((g.x, abs_pe_list), dim=1).to(x.device)
                batch_graph.append(g)
            graphs.append(Batch.from_data_list(batch_graph).to(x.device))

        return graphs

    def to_pygdata(self, adj, feature):
        adj = adj.to_dense()
        adj = adj.cpu().detach().numpy()
        adj = sp.coo_matrix(adj)
        adj = adj.tocoo()
        pyg_data = Data(x=feature,
                        edge_index=torch.tensor(np.array([adj.row, adj.col]), dtype=torch.long))
        return pyg_data


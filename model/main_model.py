
from construct_graph import inter_graph, intra_graph
from torch_geometric.nn import global_max_pool
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv
from model.adver_network import ReverseLayerF, Discriminator
from einops import rearrange
import torch.nn as nn
import torch

class main_model_ood(nn.Module):
    def __init__(self, seg_len, d_model1, d_model2, r1, r2, k_hops, dim, node_num, pe_ratio,
                 num_class=2, num_layers=1, class_type='mlp'):
        super(main_model_ood, self).__init__()
        self.seg_len = seg_len
        self.k_hops = k_hops
        self.dim = dim

        self.graph_builder = inter_graph.inter_topk(seg_len, pe_ratio, r1, node_num)
        self.process_graph = intra_graph.intra_topk(r2, pe_ratio, dim)

        self.gnn_num_layers = num_layers
        self.time_gnns = nn.ModuleList([])
        for j in range(num_layers):
            self.time_gnn = nn.ModuleList([])
            for i in range(dim):
                if j == 0:
                    self.time_gnn.append(GCNConv(seg_len + int(node_num * pe_ratio), d_model1))
                else:
                    self.time_gnn.append(GCNConv(d_model1, d_model1))
            self.time_gnns.append(self.time_gnn)

        self.subgraph_gnns = nn.ModuleList([])
        self.dim_gnns = nn.ModuleList([])
        for j in range(num_layers):
            if j == 0:
                self.subgraph_gnn = GCNConv(d_model1 * node_num + int(dim * pe_ratio), d_model2)
            else:
                self.subgraph_gnn = GCNConv(d_model2, d_model2)
            self.subgraph_gnns.append(self.subgraph_gnn)
            self.dim_gnns.append(GCNConv(d_model2, d_model2))
        if class_type == 'mlp':
            self.task_classifier = nn.Sequential(
                nn.Linear(d_model2, d_model2),
                nn.ReLU(True),
                nn.Linear(d_model2, num_class)
            )
        elif class_type == 'lstm':
            self.task_classifier = nn.Sequential(
                nn.LSTM(d_model2, d_model2),
                nn.Linear(d_model2, num_class)
            )
        else:
            assert 'only support mlp, lstm.'
        self.domain_classifier = Discriminator(d_model2, d_model2, dim)

    def forward(self, x):

        graphs = self.graph_builder(x)
        time_output = []
        for i, g in enumerate(graphs):
            for j in range(self.gnn_num_layers):
                output = self.time_gnns[j][i](g.x, g.edge_index)
                g.x = output
            time_output.append(g.x)
        batch_size = x.shape[0]
        time_output = rearrange(torch.stack(time_output).to(x.device), 'd (b n) f -> d b n f', b=batch_size)
        dim_graph = self.process_graph(time_output)
        dim_graph = Batch.from_data_list(dim_graph).to(x.device)

        dim_subgraphs = Batch.from_data_list(self.get_subgraph_data(self.k_hops, dim_graph)).to(x.device)
        for i in range(self.gnn_num_layers):
            subgraph_output = self.subgraph_gnns[i](dim_subgraphs.x, dim_subgraphs.edge_index)
            dim_subgraphs.x = subgraph_output

        sub_output = global_max_pool(dim_subgraphs.x, dim_subgraphs.batch)
        for i in range(self.gnn_num_layers):
            dim_output = self.dim_gnns[i](sub_output, dim_graph.edge_index)
            sub_output = dim_output

        dim_output = sub_output
        domain_output = self.domain_classifier(ReverseLayerF.apply(dim_output, 1.5))
        dim_output = rearrange(dim_output, '(b d) f-> b d f', b=batch_size)
        dim_output = global_max_pool(dim_output, batch=None)

        return self.task_classifier(dim_output), domain_output, dim_output

    def get_subgraph_data(self, k_hops, graph):
        subgraphs = []
        for node_idx in range(graph.num_nodes):
            sub_nodes, sub_edge_index, _, edge_mask = k_hop_subgraph(torch.tensor([node_idx], dtype=torch.long), k_hops,
                                                                     graph.edge_index,
                                                                     num_nodes=graph.num_nodes, relabel_nodes=True)
            g = Data(x=graph.x[sub_nodes], edge_index=sub_edge_index)
            subgraphs.append(g)

        return subgraphs

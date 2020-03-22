import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention

from model import GNN


class SiameseModel(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin", graph_pooling="mean"):
        super().__init__()
        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")
        self.model = GNN(num_layer=num_layer, emb_dim=emb_dim, JK=JK, drop_ratio=drop_ratio, gnn_type=gnn_type)

    def __call__(self, x):
        graph_1_batch_data = x[0]
        graph_2_batch_data = x[1]
        graph_rep_1 = self._extract_graph_rep(graph_1_batch_data)
        graph_rep_2 = self._extract_graph_rep(graph_2_batch_data)
        graph_rep_1 = graph_rep_1.reshape([graph_rep_1.shape[0], graph_rep_1.shape[1], 1])
        graph_rep_2 = graph_rep_2.reshape([graph_rep_1.shape[0], 1, graph_rep_1.shape[1]])
        similarity_score = torch.bmm(graph_rep_2, graph_rep_1).squeeze()
        return similarity_score

    def _extract_graph_rep(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        node_representation = self.model(x, edge_index, edge_attr)
        pooled = self.pool(node_representation, batch)
        center_node_rep = node_representation[data.center_node_idx]
        graph_rep = torch.cat([pooled, center_node_rep], dim=1)
        return graph_rep

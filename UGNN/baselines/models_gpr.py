"""
GPR-GNN: Generalized PageRank Graph Neural Network
Reference: Chien et al., ICLR 2021

Adaptive GPR weights learned jointly with the GNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import add_self_loops
from torch.nn import Parameter
from torch.nn.init import constant_


class GPRGNNNet(nn.Module):
    """GPR-GNN: GCN propagation with learnable GPR weights."""
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=10, dropout=0.5, heads=1, ppnp='GPR'):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.K = num_layers

        # GPR weights: one per propagation step
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self._cached_adj = None
        self._cached_edge_key = None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        # Initialize GPR weights with PPR approximation (alpha=0.5)
        alpha = 0.5
        temp = alpha * (1 - alpha) ** torch.arange(self.K + 1)
        temp[-1] = (1 - alpha) ** self.K
        self.temp.data.copy_(temp)
        self._cached_adj = None
        self._cached_edge_key = None

    def _build_norm_adj(self, edge_index, num_nodes):
        """Build symmetric normalized adjacency as SparseTensor."""
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index

        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(0, col, torch.ones(col.size(0), device=edge_index.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

        value = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        adj = SparseTensor(
            row=row, col=col, value=value,
            sparse_sizes=(num_nodes, num_nodes),
        )
        return adj

    def forward(self, x, edge_index):
        # Feature transform
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.lin2(x)

        # Cache normalized adjacency
        edge_key = (edge_index.shape, edge_index.sum().item())
        if self._cached_adj is None or self._cached_edge_key != edge_key:
            self._cached_adj = self._build_norm_adj(edge_index, x.size(0))
            self._cached_edge_key = edge_key

        # GPR propagation
        out = self.temp[0] * h
        for k in range(1, self.K + 1):
            h = matmul(self._cached_adj, h)
            out = out + self.temp[k] * h

        return out

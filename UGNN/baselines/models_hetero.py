"""
H2GCN: Beyond Homophily in Graph Neural Networks
Reference: Zhu et al., NeurIPS 2020

Two-style architecture with separate high/low-pass filtering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class H2GCNConv(MessagePassing):
    """H2GCN propagation layer: separate ego/out-group/in-group aggregation."""
    def __init__(self, **kwargs):
        super().__init__(aggr='mean', **kwargs)

    def forward(self, x, edge_index):
        # Remove self-loops for neighbor aggregation
        edge_index_no_self, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index_no_self

        # Compute degrees for normalization
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-1.0)
        deg_inv[deg_inv == float('inf')] = 0.0

        # Neighbor mean aggregation
        out = self.propagate(edge_index_no_self, x=x, deg_inv=deg_inv)

        # Self feature (ego)
        self_feat = x

        return torch.cat([self_feat, out], dim=-1)


class H2GCNNet(nn.Module):
    """H2GCN model with 2 propagation layers."""
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, heads=1):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        # After first layer: dim doubles (self + neighbor concat)
        curr_dim = in_channels * 2
        for _ in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            curr_dim = curr_dim * 2

        # Classification head
        self.lin1 = nn.Linear(curr_dim, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x

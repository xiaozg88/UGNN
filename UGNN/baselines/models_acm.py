"""
ACM-GCN: Anisotropic Channel Mixing GCN
Reference: Liu et al., AAAI 2023

Multi-frequency channel decomposition: low-pass, high-pass, identity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch.nn import Parameter, Linear
from torch.nn.init import xavier_uniform_, constant_


class ACMConv(nn.Module):
    """ACM-GCN layer: Adaptive Channel Mixing."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W_low = Parameter(torch.empty(in_channels, out_channels))
        self.W_high = Parameter(torch.empty(in_channels, out_channels))
        self.W_id = Parameter(torch.empty(in_channels, out_channels))
        self.alpha = Parameter(torch.tensor(0.5))
        self.reset_parameters()

    def reset_parameters(self):
        xavier_uniform_(self.W_low)
        xavier_uniform_(self.W_high)
        xavier_uniform_(self.W_id)
        constant_(self.alpha, 0.5)

    def forward(self, x, adj_low, adj_high):
        # Low-pass channel
        h_low = adj_low @ x @ self.W_low
        # High-pass channel
        h_high = adj_high @ x @ self.W_high
        # Identity channel
        h_id = x @ self.W_id

        # Adaptive mixing
        alpha = torch.sigmoid(self.alpha)
        out = alpha * h_low + (1 - alpha) * h_high + h_id

        return out


class ACMGCNNet(nn.Module):
    """ACM-GCN: two-layer model with precomputed low/high-pass adjacency."""
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, heads=1):
        super().__init__()
        self.conv1 = ACMConv(in_channels, hidden_channels)
        self.conv2 = ACMConv(hidden_channels, out_channels)
        self.dropout = dropout
        self._adj_low = None
        self._adj_high = None
        self._edge_key = None

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self._adj_low = None
        self._adj_high = None
        self._edge_key = None

    def _compute_adjs(self, x, edge_index):
        num_nodes = x.size(0)
        src, dst = edge_index

        # Degree
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=x.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

        # Low-pass: D^{-1/2} A D^{-1/2} (symmetric normalized)
        norm_low = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        adj_low = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj_low[dst, src] = norm_low

        # High-pass: I - D^{-1/2} A D^{-1/2}
        adj_high = torch.eye(num_nodes, device=x.device) - adj_low

        return adj_low, adj_high

    def forward(self, x, edge_index):
        edge_key = (edge_index.shape, edge_index.sum().item())
        if self._adj_low is None or self._edge_key != edge_key:
            self._adj_low, self._adj_high = self._compute_adjs(x, edge_index)
            self._edge_key = edge_key

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, self._adj_low, self._adj_high))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, self._adj_low, self._adj_high)

        return x

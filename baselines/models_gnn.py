"""
Standard GNN baselines: GCN, GAT, GraphSAGE, FAGCN
Using PyG built-in conv layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, FAConv
from torch_sparse import SparseTensor


# ============================================================
# GCN
# ============================================================

class GCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# ============================================================
# GAT
# ============================================================

class GATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=8):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=1))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# ============================================================
# GraphSAGE
# ============================================================

class SAGENet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# ============================================================
# FAGCN
# ============================================================

class FAGCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4,
                 dropout=0.5, heads=1, eps=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(FAConv(hidden_channels, eps=eps))
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x0 = F.relu(self.lin1(x))
        x0 = F.dropout(x0, p=self.dropout, training=self.training)
        h = x0
        for conv in self.convs:
            h = F.relu(conv(h, x0, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.lin2(h)
        return x

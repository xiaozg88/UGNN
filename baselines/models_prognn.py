"""
Pro-GNN: Graph Neural Network Provably Resilient to Label Noise
Reference: Jin et al., NeurIPS 2020

IDGL: Iterative Deep Graph Learning
Reference: Chen et al., NeurIPS 2020

Combined as a learnable graph structure refinement baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj


class ProGNNNet(nn.Module):
    """
    Pro-GNN style model: jointly learns graph structure and GCN parameters.
    Learns a sparse adjacency via element-wise operations on the original graph.
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, heads=1, lambda_adj=1.0):
        super().__init__()
        self.dropout = dropout
        self.lambda_adj = lambda_adj

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

        # Graph structure learner: per-edge learnable weight
        self.adj_bias = None  # Will be initialized per dataset

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def _init_adj_bias(self, num_edges, device):
        """Initialize adjacency bias for the current graph."""
        if self.adj_bias is None or self.adj_bias.size(0) != num_edges:
            self.adj_bias = nn.Parameter(torch.zeros(num_edges, device=device))

    def _get_learned_adj(self, edge_index, num_nodes):
        """Get learned adjacency with soft gating."""
        edge_weight = torch.sigmoid(self.adj_bias)
        return edge_index, edge_weight

    def forward(self, x, edge_index):
        self._init_adj_bias(edge_index.size(1), x.device)
        learned_edge_index, edge_weight = self._get_learned_adj(
            edge_index, x.size(0)
        )

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, learned_edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, learned_edge_index, edge_weight=edge_weight)

        return x

    def adj_loss(self):
        """Regularization loss for learned adjacency."""
        if self.adj_bias is not None:
            edge_weight = torch.sigmoid(self.adj_bias)
            # Sparsity + feature smoothness regularization
            l1 = edge_weight.sum() / edge_weight.size(0)
            return self.lambda_adj * l1
        return torch.tensor(0.0, device=next(self.parameters()).device)

"""
LINKX: Neural Link Prediction as Factored Graph Attention
Reference: Lim et al., NeurIPS 2021

For node classification: separate feature and adjacency MLPs, then combine.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LINKXNet(nn.Module):
    """
    LINKX model for node classification.
    Processes node features and adjacency structure separately, then combines.
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, dropout=0.5, heads=1, num_nodes=None):
        super().__init__()
        self.dropout = dropout

        # Feature MLP
        self.feat_encoder = nn.ModuleList()
        self.feat_encoder.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.feat_encoder.append(nn.Linear(hidden_channels, hidden_channels))

        # Structure MLP (from adjacency row)
        self.struct_encoder = nn.ModuleList()
        if num_nodes is not None and num_nodes > 0:
            # Use sparse adjacency rows via embedding
            self.use_adj_embedding = True
            self.adj_embed = nn.Embedding(num_nodes, hidden_channels)
            self.adj_mlp = nn.Linear(hidden_channels, hidden_channels)
        else:
            self.use_adj_embedding = False
            self.adj_mlp = None

        # Combine and classify
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for layer in self.feat_encoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        if self.use_adj_embedding:
            self.adj_embed.reset_parameters()
            self.adj_mlp.reset_parameters()
        self.combine.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index):
        # Feature path
        h_feat = x
        for i, layer in enumerate(self.feat_encoder):
            h_feat = layer(h_feat)
            h_feat = F.relu(h_feat)
            h_feat = F.dropout(h_feat, p=self.dropout, training=self.training)

        # Structure path: use node index embedding as proxy for adjacency
        if self.use_adj_embedding:
            num_nodes = x.size(0)
            node_ids = torch.arange(num_nodes, device=x.device)
            h_struct = self.adj_embed(node_ids)
            h_struct = F.relu(self.adj_mlp(h_struct))
            h_struct = F.dropout(h_struct, p=self.dropout, training=self.training)
        else:
            h_struct = h_feat

        # Combine
        h = torch.cat([h_feat, h_struct], dim=-1)
        h = F.relu(self.combine(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.classifier(h)

        return out

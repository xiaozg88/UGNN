"""
Unified training loop for all baseline models.
Uses the same split protocol as UGNN for fair comparison.
"""

import os
import random
import warnings
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from torch_geometric.transforms import NormalizeFeatures
from torch_sparse import SparseTensor

# Import UGNN's data loading and split utilities
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ugnn.config import Config, seed_everything
from ugnn.datasets import DataLoader, get_or_create_split


def train_step(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = int((pred[mask] == data.y[mask]).sum())
    total = int(mask.sum())
    return correct / total if total > 0 else 0.0


def run_baseline(model_name, dataset_name, hidden=64, lr=0.01, wd=5e-4,
                 dropout=0.5, epochs=500, patience=100, num_runs=5,
                 baseseed=42, split_dir='./splits'):
    """Run a baseline model and return test accuracy."""
    warnings.filterwarnings("ignore")

    dataset, data = DataLoader(dataset_name)
    data.num_nodes = dataset.num_nodes

    train_rate = 0.6
    val_rate = 0.2

    if dataset_name == 'penn94':
        num_nodes = torch.count_nonzero(data.y + 1).item()
    else:
        num_nodes = dataset.num_nodes

    percls_trn = int(round(train_rate * num_nodes / dataset.num_classes))
    val_lb = int(round(val_rate * num_nodes))

    # Build model
    in_channels = dataset.num_features
    out_channels = dataset.num_classes

    model_fn = get_model_fn(model_name)
    if model_name == 'LINKX':
        model = model_fn(in_channels, hidden, out_channels,
                         dropout=dropout, num_nodes=num_nodes)
    else:
        model = model_fn(in_channels, hidden, out_channels, dropout=dropout)

    device = Config.device

    accs = []
    test_accs = []

    for run_id in trange(num_runs, desc=f"{model_name}|{dataset_name}"):
        seed_everything(baseseed + run_id)

        data = get_or_create_split(
            data=data,
            dataset_name=dataset_name,
            num_classes=dataset.num_classes,
            percls_trn=percls_trn,
            val_lb=val_lb,
            split_dir=split_dir,
            baseseed=baseseed,
            run_id=run_id,
            use_saved_splits=True,
            save_splits=True,
        ).to(device)

        model = model_fn(in_channels, hidden, out_channels,
                         dropout=dropout) if model_name != 'LINKX' else \
            model_fn(in_channels, hidden, out_channels,
                     dropout=dropout, num_nodes=num_nodes)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()

        best_val = 0.0
        best_test = 0.0
        es_count = patience

        for epoch in range(epochs):
            loss = train_step(model, data, optimizer, criterion)

            val_acc = evaluate(model, data, data.val_mask)
            test_acc = evaluate(model, data, data.test_mask)

            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
                es_count = patience
            else:
                es_count -= 1

            if es_count <= 0:
                break

        accs.append(best_val)
        test_accs.append(best_test)

    accs = torch.tensor(accs)
    test_accs = torch.tensor(test_accs)

    result = {
        'valid_mean': 100 * accs.mean().item(),
        'valid_std': 100 * accs.std().item(),
        'test_mean': 100 * test_accs.mean().item(),
        'test_std': 100 * test_accs.std().item(),
    }

    print(f"{dataset_name} [{model_name}] test_acc: "
          f"{result['test_mean']:.2f} ± {result['test_std']:.2f}")
    print(f"BASELINE_RESULT|{dataset_name}|{model_name}|"
          f"{result['test_mean']:.2f}|{result['test_std']:.2f}")

    return result


import torch.nn as nn


def get_model_fn(name):
    from .models_gnn import GCNNet, GATNet, SAGENet, FAGCNNet
    from .models_hetero import H2GCNNet
    from .models_gpr import GPRGNNNet
    from .models_acm import ACMGCNNet
    from .models_linkx import LINKXNet
    from .models_prognn import ProGNNNet

    models = {
        'GCN': GCNNet,
        'GAT': GATNet,
        'GraphSAGE': SAGENet,
        'FAGCN': FAGCNNet,
        'H2GCN': H2GCNNet,
        'GPR-GNN': GPRGNNNet,
        'ACM-GCN': ACMGCNNet,
        'LINKX': LINKXNet,
        'Pro-GNN': ProGNNNet,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    return models[name]

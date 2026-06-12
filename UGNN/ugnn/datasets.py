import os
import os.path as osp
import pickle

import numpy as np
import torch
from torch_geometric.transforms import NormalizeFeatures, Compose, ToUndirected
from torch_geometric.datasets import (
    Planetoid,
    Amazon,
    WikipediaNetwork,
    Actor,
    LINKXDataset,
    Flickr,
)
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import remove_self_loops
from torch_geometric.io import read_npz
from torch_sparse import coalesce

from ogb.nodeproppred import PygNodePropPredDataset


# ============================================================
# Split Utils
# ============================================================

def gpr_splits(data, num_classes, percls_trn, val_lb):
    indices = []
    train_index = []

    y_device = data.y.device
    mask_device = data.y.device

    for i in range(num_classes):
        index = (data.y == i).nonzero(as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0), device=y_device)]
        train_index.append(index[:percls_trn])
        indices.append(index[percls_trn:])

    train_index = torch.cat(train_index, dim=0)
    rest_index = torch.cat(indices, dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0), device=y_device)]

    val_index = rest_index[:val_lb]
    test_index = rest_index[val_lb:]

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=mask_device)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=mask_device)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=mask_device)

    data.train_mask[train_index] = True
    data.val_mask[val_index] = True
    data.test_mask[test_index] = True

    return data


def make_split_path(split_dir, dataset_name, baseseed, run_id):
    split_root = osp.join(split_dir, dataset_name)
    os.makedirs(split_root, exist_ok=True)
    return osp.join(split_root, f"{dataset_name}_seed{baseseed}_run{run_id}.pt")


def save_split_masks(data, split_path):
    os.makedirs(osp.dirname(split_path), exist_ok=True)
    split_dict = {
        "train_mask": data.train_mask.detach().cpu(),
        "val_mask": data.val_mask.detach().cpu(),
        "test_mask": data.test_mask.detach().cpu(),
    }
    torch.save(split_dict, split_path)
    print(f"[Split] Saved split masks to: {split_path}")


def load_split_masks(data, split_path):
    split_dict = torch.load(split_path, map_location="cpu")
    data.train_mask = split_dict["train_mask"].bool()
    data.val_mask = split_dict["val_mask"].bool()
    data.test_mask = split_dict["test_mask"].bool()
    print(f"[Split] Loaded split masks from: {split_path}")
    return data


def get_or_create_split(
    data, dataset_name, num_classes, percls_trn, val_lb,
    split_dir, baseseed, run_id, use_saved_splits=True, save_splits=True,
):
    split_path = make_split_path(split_dir, dataset_name, baseseed, run_id)

    if use_saved_splits and osp.exists(split_path):
        data = load_split_masks(data, split_path)
    else:
        data = gpr_splits(data, num_classes, percls_trn, val_lb)
        if save_splits:
            save_split_masks(data, split_path)

    return data


# ============================================================
# Custom Dataset Classes
# ============================================================

class dataset_heterophily(InMemoryDataset):
    def __init__(self, root='data/', name=None, p2raw=None,
                 train_percent=0.01, transform=None, pre_transform=None):
        existing_dataset = ['chameleon', 'film', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(f'name must be one of: {existing_dataset}')
        self.name = name
        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(f'path "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)
        self.root = root
        super(dataset_heterophily, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [self.name]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class WebKB(InMemoryDataset):
    url = 'https://gitee.com/rockcor/geom-gcn/tree/master/new_data'

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']
        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in lines]
            x = torch.tensor(x, dtype=torch.float)
            y = [int(r.split('\t')[2]) for r in lines]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r', encoding='utf-8') as f:
            edges = f.read().split('\n')[1:-1]
            edges = [[int(v) for v in r.split('\t')] for r in edges]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_index, _ = remove_self_loops(edge_index)
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class NPZ(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.name = name.lower()
        path = osp.join(root, name + '.npz')

        if self.name in ['cora_full', 'cocs', 'cophy']:
            data = read_npz(path)
            edge_index, _ = remove_self_loops(data.edge_index)
            edge_index = to_undirected(edge_index, num_nodes=data.x.shape[0])
            edge_index, _ = coalesce(edge_index, None, data.x.size(0), data.x.size(0))
            self.data = Data(x=data.x, edge_index=edge_index, y=data.y)
        else:
            self.data = self.read_other_npz(path)

    def read_other_npz(self, path):
        with np.load(path) as f:
            x = torch.from_numpy(f['node_features']).to(torch.float)
            edge_index = torch.from_numpy(f['edges'].T).to(torch.long)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index = to_undirected(edge_index, num_nodes=x.size(0))
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
            y = torch.from_numpy(f['node_labels']).to(torch.long)
        return Data(x=x, edge_index=edge_index, y=y)


class DotDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except Exception:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


# ============================================================
# Unified DataLoader
# ============================================================

def DataLoader(name):
    if name in [
        'cora_full', 'wiki_cooc', 'questions',
        'roman_empire', 'amazon_ratings', 'squirrel_filtered',
        'chameleon_filtered', 'cocs', 'cophy',
    ]:
        root_path = './data'
        dataset = NPZ(root_path, name, pre_transform=NormalizeFeatures())
        dataset.num_nodes = len(dataset[0].y)
        return dataset, dataset.data

    elif name in ['penn94', 'genius']:
        root_path = '../data/'
        dataset = LINKXDataset(root_path, name, pre_transform=NormalizeFeatures())
        dataset.num_nodes = len(dataset[0].y)
        return dataset, dataset.data

    elif name in ['cora', 'citeseer', 'pubmed']:
        root_path = './data'
        path = osp.join(root_path, 'data')
        dataset = Planetoid(path, name, pre_transform=NormalizeFeatures())

    elif name in ['computers', 'photo']:
        root_path = '.'
        path = osp.join(root_path, 'data', name)
        dataset = Amazon(path, name, pre_transform=NormalizeFeatures())

    elif name in ['chameleon', 'squirrel']:
        # Load directly from geom-gcn text files (avoids broken PyG download URL)
        raw_dir = osp.join('../data/', name, 'raw')
        with open(osp.join(raw_dir, 'out1_node_feature_label.txt'), 'r') as f:
            lines = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in lines]
            x = torch.tensor(x, dtype=torch.float)
            y = [int(r.split('\t')[2]) for r in lines]
            y = torch.tensor(y, dtype=torch.long)

        with open(osp.join(raw_dir, 'out1_graph_edges.txt'), 'r') as f:
            edges = f.read().split('\n')[1:-1]
            ei = [[int(v) for v in r.split('\t')] for r in edges]
            edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()

        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = NormalizeFeatures()(data)
        dataset = DotDict({
            "num_classes": int(y.max().item()) + 1,
            "num_node_features": x.shape[1],
            "num_features": x.shape[1],
            "num_nodes": x.shape[0],
        })
        return dataset, data

    elif name in ['film', 'actor']:
        root_path = './data/' + name
        dataset = Actor(root=root_path, pre_transform=NormalizeFeatures())

    elif name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(root='../data/', name=name,
                        pre_transform=NormalizeFeatures())

    elif name in ["ogbn-arxiv"]:
        root_path = '../data'
        dataset = PygNodePropPredDataset(root=root_path, name=name)
        data = dataset[0]
        data.y = data.y.squeeze()
        dataset.num_nodes = len(data.y)
        return dataset, data

    elif name in ["Flickr"]:
        root_path = '../data'
        dataset = Flickr(root=root_path + "/Flickr")

    elif name in ['chameleon_f', 'squirrel_f']:
        data_np = np.load(osp.join('./data', name + 'iltered_directed.npz'))
        x = torch.tensor(data_np['node_features'], dtype=torch.float)
        y = torch.tensor(data_np['node_labels'], dtype=torch.long)
        edge_index = torch.tensor(data_np['edges'], dtype=torch.long).permute(1, 0)
        data = Data(x=x, y=y, edge_index=edge_index)
        transform = Compose([ToUndirected()])
        data = transform(data)
        dataset = DotDict({
            "num_classes": 5,
            "num_node_features": data.x.shape[1],
            "num_features": data.x.shape[1],
            "num_nodes": data.x.shape[0],
        })
        return dataset, data

    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    dataset.num_nodes = len(dataset[0].y)
    return dataset, dataset[0]

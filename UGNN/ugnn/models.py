import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import Linear, SAGEConv
from torch_sparse import (
    SparseTensor,
    fill_diag,
    matmul,
    mul,
    remove_diag,
)
from torch_sparse import sum as sparsesum
from torch.nn import Dropout, Parameter
from torch.nn.init import xavier_uniform_, constant_, calculate_gain


# ============================================================
# StochasticGateMLP
# ============================================================

class StochasticGateMLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden=256, dropout=0.5):
        super().__init__()

        in_dim = 2 * num_features + 2 * num_classes

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.mu_head = nn.Linear(hidden, 1)
        self.logvar_head = nn.Linear(hidden, 1)

        self.log_var_min = -6.0
        self.log_var_max = 2.0

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.encoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.mu_head.reset_parameters()
        self.logvar_head.reset_parameters()

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, edge_feat):
        h = self.encoder(edge_feat)

        mu = self.mu_head(h).view(-1)
        log_var = self.logvar_head(h).view(-1)
        log_var = log_var.clamp(self.log_var_min, self.log_var_max)

        z = self.reparameterize(mu, log_var)

        gate_sample = torch.sigmoid(z)
        gate_mu = torch.sigmoid(mu)
        uncertainty = torch.sigmoid(log_var.exp())

        return gate_sample, gate_mu, log_var, uncertainty


# ============================================================
# NeutralPrior (ablation: w/o Prior)
# ============================================================

class NeutralPrior:
    def __init__(self, num_nodes, device):
        self.device = device
        self.global_prior = {
            'mu': torch.tensor(0.0, device=device),
            'var': torch.tensor(1.0, device=device),
            'homophily': 0.5,
        }
        self.local_prior = torch.zeros(num_nodes, device=device)
        print("[NeutralPrior] ablation mode: no data-driven prior")

    def get_edge_prior_mu(self, edge_index):
        src, dst = edge_index
        return 0.5 * (self.local_prior[src] + self.local_prior[dst])

    def kl_loss(self, mu, log_var, edge_index):
        return torch.tensor(0.0, device=self.device)


# ============================================================
# InformedPrior
# ============================================================

class InformedPrior:
    def __init__(self, data, device, ablation=None):
        self.device = device
        self.ablation = ablation
        self.global_prior = self._estimate_global(data)

        if ablation == 'wo_local_prior':
            N = data.num_nodes
            self.local_prior = self.global_prior['mu'].expand(N).clone()
            print("[InformedPrior] ablation: w/o local prior, using global mu only")
        else:
            self.local_prior = self._estimate_local(data)

        print(
            f"[InformedPrior] global_mu={self.global_prior['mu']:.4f} "
            f"global_var={self.global_prior['var']:.4f} "
            f"(train-homophily={self.global_prior['homophily']:.4f})"
        )

    def _estimate_global(self, data):
        src, dst = data.edge_index
        mask = data.train_mask[src] & data.train_mask[dst]

        if mask.sum() == 0:
            h_mean = torch.tensor(0.5, device=self.device)
            h_var = torch.tensor(0.25, device=self.device)
        else:
            src_m = src[mask]
            dst_m = dst[mask]

            homo = (data.y[src_m] == data.y[dst_m]).float().to(self.device)

            h_mean = homo.mean().clamp(0.05, 0.95)

            if homo.numel() > 1:
                h_var = homo.var(unbiased=False).clamp(1e-4, 1.0)
            else:
                h_var = torch.tensor(0.25, device=self.device)

        logit_var = h_var / (h_mean * (1.0 - h_mean)).pow(2)
        logit_var = logit_var.clamp(1e-4, 5.0)

        return {
            'mu': torch.logit(h_mean).to(self.device),
            'var': logit_var.to(self.device),
            'homophily': h_mean.item(),
        }

    def _estimate_local(self, data):
        src, dst = data.edge_index
        N = data.num_nodes

        mask = data.train_mask[src] & data.train_mask[dst]
        global_h = float(self.global_prior['homophily'])

        local_homo = torch.full(
            (N,), fill_value=global_h,
            dtype=torch.float, device=self.device,
        )

        homo_sum = torch.zeros(N, dtype=torch.float, device=self.device)
        count = torch.zeros(N, dtype=torch.float, device=self.device)

        if mask.sum() > 0:
            src_m = src[mask]
            dst_m = dst[mask]

            homo_edge = (data.y[src_m] == data.y[dst_m]).float().to(self.device)

            homo_sum.scatter_add_(0, src_m, homo_edge)
            count.scatter_add_(0, src_m, torch.ones_like(homo_edge))

            homo_sum.scatter_add_(0, dst_m, homo_edge)
            count.scatter_add_(0, dst_m, torch.ones_like(homo_edge))

            valid = count > 0
            local_homo[valid] = homo_sum[valid] / count[valid]

        local_homo = local_homo.clamp(0.05, 0.95)

        return torch.logit(local_homo)

    def get_edge_prior_mu(self, edge_index):
        src, dst = edge_index
        return 0.5 * (self.local_prior[src] + self.local_prior[dst])

    def kl_loss(self, mu, log_var, edge_index):
        prior_mu = self.get_edge_prior_mu(edge_index).detach()
        prior_var = self.global_prior['var']

        kl = 0.5 * (
            log_var.exp() / prior_var
            + (mu - prior_mu).pow(2) / prior_var
            - 1.0
            - log_var
            + torch.log(prior_var)
        )

        kl = kl.clamp(max=10.0)

        return kl.mean()


# ============================================================
# ConsistencyUncertaintyEstimator
# ============================================================

class ConsistencyUncertaintyEstimator:
    def __init__(self, n_samples=10):
        self.T = n_samples

    @torch.no_grad()
    def estimate(self, gate_mlp, edge_feat):
        gate_mlp.train()

        gate_samples = []
        log_var_exps = []

        for _ in range(self.T):
            gate_s, gate_mu, log_var, _ = gate_mlp(edge_feat)
            gate_samples.append(gate_mu)
            log_var_exps.append(log_var.exp().clamp(max=10.0))

        gate_samples = torch.stack(gate_samples, dim=0)
        log_var_exps = torch.stack(log_var_exps, dim=0)

        gate_mean = gate_samples.mean(dim=0)
        sigma_epistemic = gate_samples.var(dim=0)
        sigma_aleatoric = log_var_exps.mean(dim=0)

        raw_total = sigma_epistemic + sigma_aleatoric
        total_unc = raw_total / (raw_total.max().clamp(min=1e-8))

        return gate_mean, sigma_aleatoric, sigma_epistemic, total_unc

    def consistency_loss(self, gate_mlp, edge_feat, n=3):
        gate_mlp.train()
        samples = []
        for _ in range(n):
            gate_s, _, _, _ = gate_mlp(edge_feat)
            samples.append(gate_s)

        samples = torch.stack(samples, dim=0)
        return samples.var(dim=0).clamp(max=1.0).mean()


# ============================================================
# UncertaintyDrivenGraphRefiner
# ============================================================

class UncertaintyDrivenGraphRefiner:
    def __init__(self, epi_threshold=0.3, ale_threshold=0.5,
                 knn_k=5, max_repair_nodes=500, homophily=0.5,
                 use_dual_uncertainty=True):
        self.epi_threshold = epi_threshold
        self.ale_threshold = ale_threshold
        self.knn_k = knn_k
        self.max_repair_nodes = max_repair_nodes
        self.homophily = homophily
        self.use_dual_uncertainty = use_dual_uncertainty

        if homophily < 0.3:
            self.knn_k = max(1, knn_k // 2)
            print(f"[Refiner] 高异质图（homophily={homophily:.3f}），"
                  f"kNN k 自动降至 {self.knn_k}")

    def classify_edges(self, sigma_aleatoric, sigma_epistemic):
        if not self.use_dual_uncertainty:
            # ablation: w/o dual uncertainty — binary classification by total unc only
            total = sigma_aleatoric + sigma_epistemic
            total_max = total.max().clamp(min=1e-8)
            total_norm = total / total_max
            threshold = (self.epi_threshold + self.ale_threshold) / 2.0

            high_unc = total_norm > threshold

            class_A = torch.zeros_like(high_unc)
            class_B = high_unc
            class_C = torch.zeros_like(high_unc)
            class_D = ~high_unc

            return class_A, class_B, class_C, class_D

        epi_max = sigma_epistemic.max().clamp(min=1e-8)
        ale_max = sigma_aleatoric.max().clamp(min=1e-8)
        epi_norm = sigma_epistemic / epi_max
        ale_norm = sigma_aleatoric / ale_max

        high_epi = epi_norm > self.epi_threshold
        high_ale = ale_norm > self.ale_threshold

        class_A = high_epi & ~high_ale
        class_B = high_epi & high_ale
        class_C = ~high_epi & high_ale
        class_D = ~high_epi & ~high_ale

        return class_A, class_B, class_C, class_D

    def _knn_repair_with_label_filter(self, x, probs,
                                       uncertain_nodes, k, device):
        x_norm = F.normalize(x, dim=1)
        pred_labels = probs.argmax(dim=1)

        if uncertain_nodes.numel() > self.max_repair_nodes:
            perm = torch.randperm(uncertain_nodes.numel(), device=device)
            uncertain_nodes = uncertain_nodes[perm[:self.max_repair_nodes]]

        x_uncertain = x_norm[uncertain_nodes]

        batch_size = 256
        all_new_src = []
        all_new_dst = []

        k_candidates = min(k * 3, x.size(0) - 1)

        for start in range(0, uncertain_nodes.numel(), batch_size):
            end = min(start + batch_size, uncertain_nodes.numel())
            x_batch = x_uncertain[start:end]
            node_batch = uncertain_nodes[start:end]

            sim = torch.mm(x_batch, x_norm.T)

            for i, nid in enumerate(node_batch):
                sim[i, nid] = -1.0

            _, top_cand = sim.topk(k_candidates, dim=1)

            for i, nid in enumerate(node_batch):
                candidates = top_cand[i]
                same_label = pred_labels[candidates] == pred_labels[nid]
                valid_cand = candidates[same_label][:k]

                if valid_cand.numel() > 0:
                    all_new_src.append(nid.expand(valid_cand.numel()))
                    all_new_dst.append(valid_cand)

        if len(all_new_src) == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=device)

        new_src = torch.cat(all_new_src, dim=0)
        new_dst = torch.cat(all_new_dst, dim=0)

        return torch.stack([new_src, new_dst], dim=0)

    def refine(self, data, sigma_aleatoric, sigma_epistemic, probs):
        class_A, class_B, class_C, class_D = self.classify_edges(
            sigma_aleatoric, sigma_epistemic
        )

        edge_index = data.edge_index
        weights = torch.ones(edge_index.size(1), device=edge_index.device)

        weights[class_B] = 0.05

        ale_norm = sigma_aleatoric / sigma_aleatoric.max().clamp(min=1e-8)
        weights[class_C] = (1.0 - ale_norm[class_C]).clamp(0.1, 1.0)

        new_edges = None
        new_weights = None
        n_class_A = int(class_A.sum().item())

        if n_class_A > 0 and self.knn_k > 0:
            uncertain_nodes = torch.unique(
                torch.cat([
                    edge_index[0, class_A],
                    edge_index[1, class_A],
                ], dim=0)
            )

            x = data.x.to_dense() if hasattr(data.x, 'to_dense') else data.x

            new_edges = self._knn_repair_with_label_filter(
                x=x, probs=probs,
                uncertain_nodes=uncertain_nodes,
                k=self.knn_k, device=edge_index.device,
            )

            if new_edges.size(1) > 0:
                new_weights = torch.ones(
                    new_edges.size(1), device=edge_index.device
                )

        if new_edges is not None and new_edges.size(1) > 0:
            refined_edge_index = torch.cat([edge_index, new_edges], dim=1)
            refined_weights = torch.cat([weights, new_weights], dim=0)
        else:
            refined_edge_index = edge_index
            refined_weights = weights

        refine_stats = {
            'class_A': int(class_A.sum()),
            'class_B': int(class_B.sum()),
            'class_C': int(class_C.sum()),
            'class_D': int(class_D.sum()),
            'new_edges': new_edges.size(1) if new_edges is not None else 0,
            'total_edges': refined_edge_index.size(1),
        }

        return refined_edge_index, refined_weights, refine_stats


# ============================================================
# Adjacency Utilities
# ============================================================

def normalize_weighted_adj(adj_t, addself=True):
    if addself:
        adj_t = fill_diag(adj_t, 1.0)
    else:
        adj_t = remove_diag(adj_t)

    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.0)

    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))

    return adj_t


def build_refined_adjs(data, args):
    gamma = getattr(args, 'uncertainty_gamma', 1.0)

    orig_edge_index = data.edge_index
    n_orig = orig_edge_index.size(1)

    refined_edge_index = data.refined_edge_index
    refined_weights = data.edge_weights
    n_refined = refined_edge_index.size(1)

    edge_gate = data.edge_gate
    total_unc = data.edge_uncertainty
    confidence = (1.0 - total_unc).clamp(0.0).pow(gamma)

    orig_same = edge_gate * confidence
    orig_diff = (1.0 - edge_gate) * confidence

    if n_refined > n_orig:
        new_weights = refined_weights[n_orig:]
        new_edge_idx = refined_edge_index[:, n_orig:]

        same_edge_index = torch.cat([orig_edge_index, new_edge_idx], dim=1)
        same_values = torch.cat([orig_same, new_weights], dim=0)

        diff_edge_index = orig_edge_index
        diff_values = orig_diff
    else:
        same_edge_index = orig_edge_index
        same_values = orig_same

        diff_edge_index = orig_edge_index
        diff_values = orig_diff

    row_same = same_edge_index[1]
    col_same = same_edge_index[0]
    row_diff = diff_edge_index[1]
    col_diff = diff_edge_index[0]

    adj_same = SparseTensor(
        row=row_same, col=col_same, value=same_values,
        sparse_sizes=(data.num_nodes, data.num_nodes),
    )
    adj_diff = SparseTensor(
        row=row_diff, col=col_diff, value=diff_values,
        sparse_sizes=(data.num_nodes, data.num_nodes),
    )

    adj_same = normalize_weighted_adj(adj_same, addself=args.addself)
    adj_diff = normalize_weighted_adj(adj_diff, addself=args.addself)

    return adj_same, adj_diff


# ============================================================
# EdgeSoftNCSAGE
# ============================================================

class EdgeSoftNCSAGE(nn.Module):
    def __init__(self, num_features, num_classes, params):
        super().__init__()

        self.conv_same_1 = SAGEConv(num_features, params.hidden, bias=False)
        self.conv_same_2 = SAGEConv(params.hidden, params.hidden, bias=False)
        self.conv_diff_1 = SAGEConv(num_features, params.hidden, bias=False)
        self.conv_diff_2 = SAGEConv(params.hidden, params.hidden, bias=False)

        self.WX = Parameter(torch.empty(num_features, params.hidden))
        self.lam = Parameter(torch.zeros(3))

        self.dropout = Dropout(p=params.dp1)
        self.dropout2 = Dropout(p=params.dp2)
        self.finaldp = Dropout(p=0.5)

        self.lin1 = Linear(params.hidden, num_classes)
        self.act = F.relu
        self.args = params

        self._cached_adj_same = None
        self._cached_adj_diff = None

        self.reset_parameters()

    def reset_parameters(self):
        self.conv_same_1.reset_parameters()
        self.conv_same_2.reset_parameters()
        self.conv_diff_1.reset_parameters()
        self.conv_diff_2.reset_parameters()
        self.lin1.reset_parameters()
        constant_(self.lam, 0)
        xavier_uniform_(self.WX, gain=calculate_gain('relu'))
        self._cached_adj_same = None
        self._cached_adj_diff = None

    def forward(self, data):
        x = data.x.to_dense() if isinstance(data.x, SparseTensor) else data.x

        if data.update_gate or self._cached_adj_same is None:
            adj_same, adj_diff = build_refined_adjs(data, self.args)
            self._cached_adj_same = adj_same
            self._cached_adj_diff = adj_diff
            data.update_gate = False
        else:
            adj_same = self._cached_adj_same
            adj_diff = self._cached_adj_diff

        xs = self.conv_same_1(x, adj_same)
        xs = self.act(xs)
        xs = self.dropout(xs)
        xs = self.conv_same_2(xs, adj_same)

        xd = self.conv_diff_1(x, adj_diff)
        xd = self.act(xd)
        xd = self.dropout2(xd)
        xd = self.conv_diff_2(xd, adj_diff)

        xr = torch.mm(x, self.WX)

        lam_raw, lam_same, lam_diff = torch.softmax(self.lam, dim=0)

        xf = lam_raw * xr + lam_same * xs + lam_diff * xd
        xf = self.act(xf)
        xf = self.finaldp(xf)
        out = self.lin1(xf)

        return out


# ============================================================
# EdgeSoftNCGCN
# ============================================================

class EdgeSoftNCGCN(nn.Module):
    def __init__(self, num_features, num_classes, params):
        super().__init__()

        self.W1S = Parameter(torch.empty(num_features, params.hidden))
        self.W2S = Parameter(torch.empty(params.hidden, params.hidden))
        self.W1D = Parameter(torch.empty(num_features, params.hidden))
        self.W2D = Parameter(torch.empty(params.hidden, params.hidden))
        self.WX = Parameter(torch.empty(num_features, params.hidden))
        self.lam = Parameter(torch.zeros(3))

        self.dropout = Dropout(p=params.dp1)
        self.dropout2 = Dropout(p=params.dp2)
        self.finaldp = Dropout(p=0.5)

        self.lin1 = Linear(params.hidden, num_classes)
        self.act = F.relu
        self.args = params

        self._cached_adj_same = None
        self._cached_adj_diff = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        constant_(self.lam, 0)
        xavier_uniform_(self.W1S, gain=calculate_gain('relu'))
        xavier_uniform_(self.W2S, gain=calculate_gain('relu'))
        xavier_uniform_(self.W1D, gain=calculate_gain('relu'))
        xavier_uniform_(self.W2D, gain=calculate_gain('relu'))
        xavier_uniform_(self.WX, gain=calculate_gain('relu'))
        self._cached_adj_same = None
        self._cached_adj_diff = None

    def forward(self, data):
        x = data.x

        if data.update_gate or self._cached_adj_same is None:
            adj_same, adj_diff = build_refined_adjs(data, self.args)
            self._cached_adj_same = adj_same
            self._cached_adj_diff = adj_diff
            data.update_gate = False
        else:
            adj_same = self._cached_adj_same
            adj_diff = self._cached_adj_diff

        xs = matmul(adj_same, x)
        xs = matmul(xs, self.W1S)
        xs = self.act(xs)
        xs = self.dropout(xs)
        xs = torch.mm(matmul(adj_same, xs), self.W2S)

        xd = matmul(adj_diff, x)
        xd = matmul(xd, self.W1D)
        xd = self.act(xd)
        xd = self.dropout2(xd)
        xd = torch.mm(matmul(adj_diff, xd), self.W2D)

        xr = matmul(x, self.WX)

        lam_raw, lam_same, lam_diff = torch.softmax(self.lam, dim=0)

        xf = lam_raw * xr + lam_same * xs + lam_diff * xd
        xf = self.act(xf)
        xf = self.finaldp(xf)
        out = self.lin1(xf)

        return out


# ============================================================
# Single-channel adjacency (ablation: w/o Dual-channel)
# ============================================================

def build_single_adj(data, args):
    gamma = getattr(args, 'uncertainty_gamma', 1.0)

    orig_edge_index = data.edge_index
    n_orig = orig_edge_index.size(1)

    refined_edge_index = data.refined_edge_index
    refined_weights = data.edge_weights
    n_refined = refined_edge_index.size(1)

    edge_gate = data.edge_gate
    total_unc = data.edge_uncertainty
    confidence = (1.0 - total_unc).clamp(0.0).pow(gamma)

    # single channel: use gate*confidence as edge weight (no same/diff split)
    orig_values = edge_gate * confidence

    if n_refined > n_orig:
        new_weights = refined_weights[n_orig:]
        new_edge_idx = refined_edge_index[:, n_orig:]

        edge_index = torch.cat([orig_edge_index, new_edge_idx], dim=1)
        values = torch.cat([orig_values, new_weights], dim=0)
    else:
        edge_index = orig_edge_index
        values = orig_values

    adj = SparseTensor(
        row=edge_index[1],
        col=edge_index[0],
        value=values,
        sparse_sizes=(data.num_nodes, data.num_nodes),
    )

    adj = normalize_weighted_adj(adj, addself=args.addself)
    return adj


# ============================================================
# SingleChannelSAGE (ablation: w/o Dual-channel)
# ============================================================

class SingleChannelSAGE(nn.Module):
    def __init__(self, num_features, num_classes, params):
        super().__init__()

        self.conv1 = SAGEConv(num_features, params.hidden, bias=False)
        self.conv2 = SAGEConv(params.hidden, params.hidden, bias=False)

        self.WX = Parameter(torch.empty(num_features, params.hidden))
        self.lam = Parameter(torch.zeros(1))

        self.dropout = Dropout(p=params.dp1)
        self.finaldp = Dropout(p=0.5)

        self.lin1 = Linear(params.hidden, num_classes)
        self.act = F.relu
        self.args = params

        self._cached_adj = None

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin1.reset_parameters()
        xavier_uniform_(self.WX, gain=calculate_gain('relu'))
        constant_(self.lam, 0)
        self._cached_adj = None

    def forward(self, data):
        x = data.x.to_dense() if isinstance(data.x, SparseTensor) else data.x

        if data.update_gate or self._cached_adj is None:
            adj = build_single_adj(data, self.args)
            self._cached_adj = adj
            data.update_gate = False
        else:
            adj = self._cached_adj

        xg = self.conv1(x, adj)
        xg = self.act(xg)
        xg = self.dropout(xg)
        xg = self.conv2(xg, adj)

        xr = torch.mm(x, self.WX)

        lam_gate = torch.sigmoid(self.lam)
        xf = (1.0 - lam_gate) * xr + lam_gate * xg
        xf = self.act(xf)
        xf = self.finaldp(xf)
        out = self.lin1(xf)

        return out

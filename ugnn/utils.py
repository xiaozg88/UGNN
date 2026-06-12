import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score as _sklearn_f1_score

from torch_sparse import SparseTensor

from .models import (
    InformedPrior,
    ConsistencyUncertaintyEstimator,
    UncertaintyDrivenGraphRefiner,
    StochasticGateMLP,
)


# ============================================================
# Edge Gate Initialization
# ============================================================

@torch.no_grad()
def init_edge_gate(data):
    num_edges = data.edge_index.size(1)
    data.edge_gate = torch.ones(num_edges, dtype=torch.float,
                                device=data.edge_index.device)
    data.edge_weights = torch.ones(num_edges, dtype=torch.float,
                                   device=data.edge_index.device)
    data.edge_uncertainty = torch.zeros(num_edges, dtype=torch.float,
                                        device=data.edge_index.device)
    data.sigma_aleatoric = torch.zeros(num_edges, dtype=torch.float,
                                       device=data.edge_index.device)
    data.sigma_epistemic = torch.zeros(num_edges, dtype=torch.float,
                                       device=data.edge_index.device)
    data.update_gate = True
    data.refined_edge_index = data.edge_index.clone()
    return data


def get_dense_x(data):
    if isinstance(data.x, SparseTensor):
        return data.x.to_dense()
    return data.x


def build_edge_feature_batch(x, probs, edge_index, edge_ids):
    src = edge_index[0, edge_ids]
    dst = edge_index[1, edge_ids]
    edge_feat = torch.cat(
        [x[src], x[dst], probs[src], probs[dst]],
        dim=1,
    )
    return edge_feat


# ============================================================
# Gate Training
# ============================================================

def train_gate_mlp_full(
    data, probs, gate_mlp, gate_optimizer,
    informed_prior, consistency_estimator,
    epochs=20, batch_size=65536,
    kl_beta=1e-3, consist_beta=1e-2, refine_beta=1e-3,
):
    gate_mlp.train()

    edge_index = data.edge_index
    src_all = edge_index[0]
    dst_all = edge_index[1]

    train_edge_mask = data.train_mask[src_all] & data.train_mask[dst_all]
    train_edge_ids = train_edge_mask.nonzero(as_tuple=False).view(-1)

    if train_edge_ids.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    x = get_dense_x(data)
    probs = probs.detach()

    total_bce = 0.0
    total_kl = 0.0
    total_consist = 0.0
    total_refine = 0.0
    total_loss = 0.0
    total_count = 0

    for _ in range(epochs):
        perm = torch.randperm(
            train_edge_ids.numel(), device=train_edge_ids.device,
        )
        shuffled_ids = train_edge_ids[perm]

        for start in range(0, shuffled_ids.numel(), batch_size):
            batch_ids = shuffled_ids[start:start + batch_size]

            edge_feat = build_edge_feature_batch(
                x=x, probs=probs, edge_index=edge_index, edge_ids=batch_ids,
            )

            src = edge_index[0, batch_ids]
            dst = edge_index[1, batch_ids]
            edge_label = (data.y[src] == data.y[dst]).float()

            gate_optimizer.zero_grad()

            gate_sample, gate_mu, log_var, _ = gate_mlp(edge_feat)

            bce_loss = F.binary_cross_entropy(gate_sample, edge_label)

            batch_edge_index = edge_index[:, batch_ids]
            mu_logit = torch.logit(gate_mu.clamp(1e-6, 1.0 - 1e-6))

            kl_loss = informed_prior.kl_loss(
                mu=mu_logit, log_var=log_var, edge_index=batch_edge_index,
            )

            # Skip consistency computation if beta=0 (ablation: wo_consistency)
            if consist_beta > 0:
                consist_loss = consistency_estimator.consistency_loss(
                    gate_mlp=gate_mlp, edge_feat=edge_feat, n=3,
                )
            else:
                consist_loss = torch.tensor(0.0, device=edge_feat.device)

            refine_loss = torch.tensor(0.0, device=edge_feat.device)

            if (
                hasattr(data, 'refined_edge_index')
                and data.refined_edge_index.size(1) > edge_index.size(1)
            ):
                n_new = data.refined_edge_index.size(1) - edge_index.size(1)
                n_sample = min(512, n_new)
                new_start = edge_index.size(1)

                perm_r = torch.randperm(n_new, device=edge_feat.device)
                refine_ids = new_start + perm_r[:n_sample]

                ref_src_all = data.refined_edge_index[0, refine_ids]
                ref_dst_all = data.refined_edge_index[1, refine_ids]

                train_ref_mask = (
                    data.train_mask[ref_src_all]
                    & data.train_mask[ref_dst_all]
                )

                if train_ref_mask.sum() > 0:
                    refine_ids_train = refine_ids[train_ref_mask]

                    ref_feat = build_edge_feature_batch(
                        x=x, probs=probs,
                        edge_index=data.refined_edge_index,
                        edge_ids=refine_ids_train,
                    )

                    ref_src = data.refined_edge_index[0, refine_ids_train]
                    ref_dst = data.refined_edge_index[1, refine_ids_train]
                    ref_label = (data.y[ref_src] == data.y[ref_dst]).float()

                    ref_gate, _, _, _ = gate_mlp(ref_feat)
                    refine_loss = F.binary_cross_entropy(ref_gate, ref_label)

            loss = (
                bce_loss
                + kl_beta * kl_loss
                + consist_beta * consist_loss
                + refine_beta * refine_loss
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(gate_mlp.parameters(), max_norm=1.0)

            gate_optimizer.step()

            n = batch_ids.numel()
            total_bce += bce_loss.item() * n
            total_kl += kl_loss.item() * n
            total_consist += consist_loss.item() * n
            total_refine += refine_loss.item() * n
            total_loss += loss.item() * n
            total_count += n

    if total_count == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    return (
        total_loss / total_count,
        total_bce / total_count,
        total_kl / total_count,
        total_consist / total_count,
        total_refine / total_count,
    )


# ============================================================
# Gate Inference & Update
# ============================================================

@torch.no_grad()
def update_edge_gate_full(
    data, probs, gate_mlp, consistency_estimator, refiner,
    temperature=1.0, min_gate=0.0, max_gate=1.0,
    batch_size=65536, refine_freq=5, current_epoch=0,
):
    gate_mlp.eval()

    if temperature != 1.0:
        logits = torch.log(probs + 1e-12) / temperature
        probs = torch.softmax(logits, dim=-1)

    x = get_dense_x(data)
    edge_index = data.edge_index
    num_edges = edge_index.size(1)

    gate_list = []
    ale_list = []
    epi_list = []
    total_list = []

    for start in range(0, num_edges, batch_size):
        end = min(start + batch_size, num_edges)
        edge_ids = torch.arange(
            start, end, device=edge_index.device, dtype=torch.long,
        )

        edge_feat = build_edge_feature_batch(
            x=x, probs=probs, edge_index=edge_index, edge_ids=edge_ids,
        )

        gate_mean, sigma_ale, sigma_epi, total_unc = \
            consistency_estimator.estimate(gate_mlp, edge_feat)

        gate_mean = gate_mean.clamp(min=min_gate, max=max_gate)

        gate_list.append(gate_mean.detach())
        ale_list.append(sigma_ale.detach())
        epi_list.append(sigma_epi.detach())
        total_list.append(total_unc.detach())

    data.edge_gate = torch.cat(gate_list, dim=0)
    data.sigma_aleatoric = torch.cat(ale_list, dim=0)
    data.sigma_epistemic = torch.cat(epi_list, dim=0)
    data.edge_uncertainty = torch.cat(total_list, dim=0)

    refine_stats = None
    if current_epoch % refine_freq == 0:
        refined_edge_index, refined_weights, refine_stats = refiner.refine(
            data=data,
            sigma_aleatoric=data.sigma_aleatoric,
            sigma_epistemic=data.sigma_epistemic,
            probs=probs,
        )
        data.refined_edge_index = refined_edge_index
        data.edge_weights = refined_weights

    data.update_gate = True

    return data, refine_stats


# ============================================================
# Train / Eval
# ============================================================

def train_step(data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, out


@torch.no_grad()
def test(data, model):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    return int((pred[data.test_mask] == data.y[data.test_mask]).sum()) \
        / int(data.test_mask.sum())


@torch.no_grad()
def test_with_f1(data, model):
    """Evaluate on test set, returning both accuracy and macro-F1."""
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()
    acc = float((y_pred == y_true).sum()) / len(y_true)
    f1 = float(_sklearn_f1_score(y_true, y_pred, average='macro'))
    return acc, f1


@torch.no_grad()
def valid(data, model):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    return int((pred[data.val_mask] == data.y[data.val_mask]).sum()) \
        / int(data.val_mask.sum())


@torch.no_grad()
def run_full_probs(data, model, forcing=0):
    model.eval()
    out = model(data)
    probs = torch.softmax(out, dim=1)

    if forcing:
        train_onehot = torch.zeros_like(probs)
        train_onehot.scatter_(1, data.y.view(-1, 1), 1.0)
        mask = data.train_mask.view(-1, 1)
        probs = torch.where(mask, train_onehot, probs)

    return probs


# ============================================================
# Uncertainty Stats
# ============================================================

@torch.no_grad()
def log_uncertainty_stats(data):
    unc = data.edge_uncertainty
    ale = data.sigma_aleatoric
    epi = data.sigma_epistemic

    return {
        'total_mean': unc.mean().item(),
        'total_std': unc.std().item(),
        'aleatoric_mean': (ale / ale.max().clamp(min=1e-8)).mean().item(),
        'epistemic_mean': (epi / epi.max().clamp(min=1e-8)).mean().item(),
        'high_unc_ratio': (unc > 0.7).float().mean().item(),
        'low_unc_ratio': (unc < 0.3).float().mean().item(),
    }

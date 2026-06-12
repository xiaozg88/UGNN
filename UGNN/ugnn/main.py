import warnings

import torch
from tqdm import trange
from torch_sparse import SparseTensor

from .config import Config, seed_everything, parse_args
from .datasets import DataLoader, get_or_create_split
from .models import (
    EdgeSoftNCSAGE,
    EdgeSoftNCGCN,
    SingleChannelSAGE,
    StochasticGateMLP,
    NeutralPrior,
    InformedPrior,
    ConsistencyUncertaintyEstimator,
    UncertaintyDrivenGraphRefiner,
)
from .utils import (
    init_edge_gate,
    train_gate_mlp_full,
    update_edge_gate_full,
    train_step,
    test,
    test_with_f1,
    valid,
    run_full_probs,
    log_uncertainty_stats,
)


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    ablation = args.ablation

    dataset, data = DataLoader(args.dataset)

    print(f"load {args.dataset} successfully!")
    print('=' * 70)
    print(args)
    print('=' * 70)
    if ablation != 'full':
        print(f"[Ablation] Mode: {ablation}")
    print(f"[Split Protocol] split_dir={args.split_dir}")
    print(f"[Split Protocol] use_saved_splits={args.use_saved_splits}")
    print(f"[Split Protocol] save_splits={args.save_splits}")
    print('=' * 70)

    # Ablation: override hyperparameters
    if ablation == 'wo_prior':
        args.kl_beta = 0.0
    if ablation == 'wo_consistency':
        args.consist_beta = 0.0
    if ablation == 'wo_refinement':
        args.refine_beta = 0.0
        args.refine_freq = 99999
        args.knn_k = 0

    train_rate = 0.6
    val_rate = 0.2

    if args.dataset == 'penn94':
        num_nodes = torch.count_nonzero(data.y + 1).item()
    else:
        num_nodes = dataset.num_nodes

    percls_trn = int(round(train_rate * num_nodes / dataset.num_classes))
    val_lb = int(round(val_rate * num_nodes))

    accs, test_accs, test_f1s = [], [], []

    # Model selection (ablation-aware)
    if ablation == 'wo_dual_channel':
        model = SingleChannelSAGE(
            dataset.num_features, dataset.num_classes, args,
        ).to(Config.device)
    elif args.model == 'EdgeNCSAGE':
        model = EdgeSoftNCSAGE(
            dataset.num_features, dataset.num_classes, args,
        ).to(Config.device)
    elif args.model == 'EdgeNCGCN':
        model = EdgeSoftNCGCN(
            dataset.num_features, dataset.num_classes, args,
        ).to(Config.device)
    else:
        raise ValueError(f'model {args.model} not supported')

    gate_mlp = StochasticGateMLP(
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        hidden=args.gate_hidden,
        dropout=args.gate_dropout,
    ).to(Config.device)

    consistency_estimator = ConsistencyUncertaintyEstimator(
        n_samples=args.mc_samples,
    )

    data.num_nodes = dataset.num_nodes
    data.x = SparseTensor.from_dense(data.x)

    print(f"UGNN V05-fixed initialized (ablation={ablation})")
    print('=' * 70)

    for rand in trange(5):
        seed_everything(args.baseseed + rand)

        data = get_or_create_split(
            data=data,
            dataset_name=args.dataset,
            num_classes=dataset.num_classes,
            percls_trn=percls_trn,
            val_lb=val_lb,
            split_dir=args.split_dir,
            baseseed=args.baseseed,
            run_id=rand,
            use_saved_splits=bool(args.use_saved_splits),
            save_splits=bool(args.save_splits),
        ).to(Config.device)

        data = init_edge_gate(data)

        # Prior selection (ablation-aware)
        if ablation == 'wo_prior':
            informed_prior = NeutralPrior(
                num_nodes=data.num_nodes, device=Config.device,
            )
        elif ablation == 'wo_local_prior':
            informed_prior = InformedPrior(
                data=data, device=Config.device, ablation='wo_local_prior',
            )
        else:
            informed_prior = InformedPrior(data=data, device=Config.device)

        # Refiner (ablation-aware)
        use_dual = (ablation != 'wo_dual_uncertainty')
        refiner = UncertaintyDrivenGraphRefiner(
            epi_threshold=args.epi_threshold,
            ale_threshold=args.ale_threshold,
            knn_k=args.knn_k,
            max_repair_nodes=args.max_repair_nodes,
            homophily=informed_prior.global_prior['homophily'],
            use_dual_uncertainty=use_dual,
        )

        model.reset_parameters()
        gate_mlp.reset_parameters()

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wd,
        )
        gate_optimizer = torch.optim.Adam(
            gate_mlp.parameters(), lr=args.lr, weight_decay=args.wd,
        )

        best_acc = 0.0
        final_test_acc = 0.0
        final_test_f1 = 0.0
        es_count = patience = 100

        gate_loss = 0.0
        bce_loss = 0.0
        kl_loss = 0.0
        consist_loss = 0.0
        refine_loss = 0.0
        refine_stats = None

        for epoch in range(500):

            loss, out = train_step(data, model, optimizer, criterion)

            val_acc = valid(data, model)
            test_acc, test_f1 = test_with_f1(data, model)

            if val_acc > best_acc:
                es_count = patience
                best_acc = val_acc
                final_test_acc = test_acc
                final_test_f1 = test_f1
            else:
                es_count -= 1

            if epoch % args.gate_update_freq == 0:

                probs = run_full_probs(data, model, args.forcing)

                gate_loss, bce_loss, kl_loss, consist_loss, refine_loss = \
                    train_gate_mlp_full(
                        data=data,
                        probs=probs.detach(),
                        gate_mlp=gate_mlp,
                        gate_optimizer=gate_optimizer,
                        informed_prior=informed_prior,
                        consistency_estimator=consistency_estimator,
                        epochs=args.gate_epochs,
                        batch_size=args.gate_batch_size,
                        kl_beta=args.kl_beta,
                        consist_beta=args.consist_beta,
                        refine_beta=args.refine_beta,
                    )

                data, refine_stats = update_edge_gate_full(
                    data=data,
                    probs=probs.detach(),
                    gate_mlp=gate_mlp,
                    consistency_estimator=consistency_estimator,
                    refiner=refiner,
                    temperature=args.gate_temperature,
                    min_gate=args.gate_min,
                    max_gate=args.gate_max,
                    batch_size=args.gate_batch_size,
                    refine_freq=args.refine_freq,
                    current_epoch=epoch,
                )

                if args.log_uncertainty:
                    unc_stats = log_uncertainty_stats(data)
                    print(
                        f"  [unc] total={unc_stats['total_mean']:.4f}"
                        f"±{unc_stats['total_std']:.4f} "
                        f"ale={unc_stats['aleatoric_mean']:.4f} "
                        f"epi={unc_stats['epistemic_mean']:.4f} "
                        f"high={unc_stats['high_unc_ratio']:.4f} "
                        f"low={unc_stats['low_unc_ratio']:.4f}"
                    )
                    if refine_stats is not None:
                        print(
                            f"  [refine] "
                            f"A={refine_stats['class_A']} "
                            f"B={refine_stats['class_B']} "
                            f"C={refine_stats['class_C']} "
                            f"D={refine_stats['class_D']} "
                            f"new_edges={refine_stats['new_edges']} "
                            f"total={refine_stats['total_edges']}"
                        )

            print(
                f"epoch:{epoch:03d} | loss:{loss.item():.4f} | "
                f"val:{val_acc:.4f} | test:{test_acc:.4f} | "
                f"gate:{gate_loss:.4f} "
                f"(bce:{bce_loss:.4f} "
                f"kl:{kl_loss:.4f} "
                f"consist:{consist_loss:.4f} "
                f"refine:{refine_loss:.4f})"
            )

            if es_count <= 0:
                print(f"  [早停] epoch {epoch}，best_val={best_acc:.4f}")
                break

        accs.append(best_acc)
        test_accs.append(final_test_acc)
        test_f1s.append(final_test_f1)

    accs = torch.tensor(accs)
    test_accs = torch.tensor(test_accs)
    test_f1s = torch.tensor(test_f1s)

    print('=' * 70)
    print(
        f'{args.dataset} [{ablation}] valid_acc: '
        f'{100 * accs.mean().item():.2f} ± {100 * accs.std().item():.2f}'
    )
    print(
        f'{args.dataset} [{ablation}] test_acc: '
        f'{100 * test_accs.mean().item():.2f} ± {100 * test_accs.std().item():.2f}'
    )
    print(
        f'{args.dataset} [{ablation}] test_f1_macro: '
        f'{100 * test_f1s.mean().item():.2f} ± {100 * test_f1s.std().item():.2f}'
    )
    # ABLATION_RESULT tag for automated parsing
    print(f'ABLBATION_RESULT|{args.dataset}|{ablation}|'
          f'{100 * test_accs.mean().item():.2f}|{100 * test_accs.std().item():.2f}|'
          f'{100 * test_f1s.mean().item():.2f}|{100 * test_f1s.std().item():.2f}')


if __name__ == '__main__':
    main()

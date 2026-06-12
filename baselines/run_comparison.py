"""
Comparison Experiment Runner
Runs UGNN and all baselines on actor, photo, cora, chameleon, texas, pubmed,
wisconsin, cornell, squirrel datasets.

Usage:
    python baselines/run_comparison.py                       # run all
    python baselines/run_comparison.py --datasets actor      # actor only
    python baselines/run_comparison.py --models GCN GAT      # specific baselines
    python baselines/run_comparison.py --skip-ugnn           # skip UGNN (already run)
"""

import subprocess
import sys
import re
import os
import argparse
import time

BASELINES = ['GCN', 'GAT', 'GraphSAGE', 'FAGCN', 'H2GCN',
             'GPR-GNN', 'ACM-GCN', 'LINKX', 'Pro-GNN']

DATASETS = ['actor', 'photo', 'cora', 'chameleon', 'texas', 'pubmed',
            'wisconsin', 'cornell', 'squirrel', 'citeseer']

MODEL_LABELS = {
    'GCN': 'GCN',
    'GAT': 'GAT',
    'GraphSAGE': 'GraphSAGE',
    'FAGCN': 'FAGCN',
    'H2GCN': 'H2GCN',
    'GPR-GNN': 'GPR-GNN',
    'ACM-GCN': 'ACM-GCN',
    'LINKX': 'LINKX',
    'Pro-GNN': 'Pro-GNN',
    'UGNN': 'UGNN',
}


def run_ugnn(dataset):
    """Run UGNN via its own entry point."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [
        sys.executable, '-m', 'ugnn.main',
        '--dataset', dataset,
        '--ablation', 'full',
    ]
    print(f"\n{'='*70}")
    print(f"Running: UGNN | {dataset}")
    print(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
    elapsed = time.time() - t0

    match = re.search(
        r'ABLBATION_RESULT\|(\w+)\|(\w+)\|([\d.]+)\|([\d.]+)',
        result.stdout,
    )
    if match:
        acc, std = float(match.group(3)), float(match.group(4))
        print(f"  -> test_acc: {acc:.2f} ± {std:.2f}  ({elapsed:.0f}s)")
        return acc, std
    else:
        print(f"  -> FAILED ({elapsed:.0f}s)")
        if result.stderr:
            print("STDERR:", result.stderr[-500:])
        return None, None


def run_baseline(model_name, dataset):
    """Run a baseline model."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [
        sys.executable, '-c',
        f"from baselines.trainer import run_baseline; "
        f"run_baseline('{model_name}', '{dataset}')",
    ]
    print(f"\n{'='*70}")
    print(f"Running: {model_name} | {dataset}")
    print(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
    elapsed = time.time() - t0

    match = re.search(
        r'BASELINE_RESULT\|(\w+)\|(\w+)\|([\d.]+)\|([\d.]+)',
        result.stdout,
    )
    if match:
        acc, std = float(match.group(3)), float(match.group(4))
        print(f"  -> test_acc: {acc:.2f} ± {std:.2f}  ({elapsed:.0f}s)")
        return acc, std
    else:
        print(f"  -> FAILED ({elapsed:.0f}s)")
        if result.stderr:
            print("STDERR:", result.stderr[-500:])
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--models', nargs='+', default=BASELINES)
    parser.add_argument('--skip-ugnn', action='store_true',
                        help='Skip UGNN (already have results)')
    args = parser.parse_args()

    results = {}
    for ds in args.datasets:
        results[ds] = {}

    # Run UGNN first
    if not args.skip_ugnn:
        for ds in args.datasets:
            acc, std = run_ugnn(ds)
            results[ds]['UGNN'] = (acc, std)

    # Run baselines
    for ds in args.datasets:
        for model in args.models:
            acc, std = run_baseline(model, ds)
            results[ds][model] = (acc, std)

    # Print summary table
    all_models = ['UGNN'] + args.models
    print('\n\n')
    print('=' * 90)
    print('COMPARISON EXPERIMENT RESULTS (test_acc %)')
    print('=' * 90)

    header = f"{'Method':<20}"
    for ds in args.datasets:
        header += f"| {ds:^20} "
    print(header)
    print('-' * len(header))

    for model in all_models:
        label = MODEL_LABELS.get(model, model)
        row = f"{label:<20}"
        for ds in args.datasets:
            entry = results[ds].get(model, (None, None))
            if entry[0] is not None:
                row += f"| {entry[0]:.2f} ± {entry[1]:.2f}      "
            else:
                row += f"| {'N/A':^20} "
        print(row)

    print('=' * 90)

    # Markdown table
    print('\n### Markdown Table\n')
    md_header = "| Method |"
    md_sep = "|--------|"
    for ds in args.datasets:
        md_header += f" {ds} |"
        md_sep += "----------|"
    print(md_header)
    print(md_sep)

    for model in all_models:
        label = MODEL_LABELS.get(model, model)
        row = f"| {label} |"
        for ds in args.datasets:
            entry = results[ds].get(model, (None, None))
            if entry[0] is not None:
                row += f" {entry[0]:.2f} ± {entry[1]:.2f} |"
            else:
                row += " N/A |"
        print(row)


if __name__ == '__main__':
    main()

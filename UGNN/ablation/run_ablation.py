"""
Ablation Study Runner
Runs all ablation variants across datasets and prints a summary table.

Usage:
    python ablation/run_ablation.py                  # run all
    python ablation/run_ablation.py --datasets actor  # actor only
    python ablation/run_ablation.py --variants wo_prior wo_consistency  # specific variants
"""

import subprocess
import sys
import re
import os
import argparse
import time

ABLATION_VARIANTS = [
    'full',
    'wo_prior',
    'wo_local_prior',
    'wo_consistency',
    'wo_refinement',
    'wo_dual_uncertainty',
    'wo_dual_channel',
]

DATASETS = ['actor', 'photo', 'cora', 'pubmed', 'chameleon', 'wisconsin', 'texas']

VARIANT_LABELS = {
    'full': 'UGNN-full',
    'wo_prior': 'w/o Prior',
    'wo_local_prior': 'w/o Local Prior',
    'wo_consistency': 'w/o Consistency',
    'wo_refinement': 'w/o Refinement',
    'wo_dual_uncertainty': 'w/o Dual Uncertainty',
    'wo_dual_channel': 'w/o Dual-channel',
}


def run_single(dataset, variant):
    cmd = [
        sys.executable, '-m', 'ugnn.main',
        '--dataset', dataset,
        '--ablation', variant,
    ]
    print(f"\n{'='*70}")
    print(f"Running: {dataset} | {variant}")
    print(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    elapsed = time.time() - t0

    # Parse ABLATION_RESULT line (now includes F1-macro)
    # Format: ABLBATION_RESULT|dataset|variant|acc|acc_std|f1|f1_std
    match = re.search(
        r'ABLBATION_RESULT\|(\w+)\|(\w+)\|([\d.]+)\|([\d.]+)\|([\d.]+)\|([\d.]+)',
        result.stdout,
    )
    if match:
        acc = float(match.group(3))
        acc_std = float(match.group(4))
        f1 = float(match.group(5))
        f1_std = float(match.group(6))
        print(f"  -> test_acc: {acc:.2f} ± {acc_std:.2f}  |  f1_macro: {f1:.2f} ± {f1_std:.2f}  ({elapsed:.0f}s)")
        return acc, acc_std, f1, f1_std
    else:
        print(f"  -> FAILED ({elapsed:.0f}s)")
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[-300:])
        return None, None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--variants', nargs='+', default=ABLATION_VARIANTS)
    args = parser.parse_args()

    results = {}
    for ds in args.datasets:
        results[ds] = {}
        for variant in args.variants:
            acc, acc_std, f1, f1_std = run_single(ds, variant)
            results[ds][variant] = (acc, acc_std, f1, f1_std)

    # Print summary table - Accuracy
    print('\n\n')
    print('=' * 100)
    print('ABLATION STUDY RESULTS - Test Accuracy (%)')
    print('=' * 100)

    header = f"{'Variant':<25}"
    for ds in args.datasets:
        header += f"| {ds:^20} "
    print(header)
    print('-' * len(header))

    for variant in args.variants:
        label = VARIANT_LABELS.get(variant, variant)
        row = f"{label:<25}"
        for ds in args.datasets:
            acc, acc_std, _, _ = results[ds].get(variant, (None, None, None, None))
            if acc is not None:
                row += f"| {acc:.2f} ± {acc_std:.2f}      "
            else:
                row += f"| {'N/A':^20} "
        print(row)

    print('=' * 100)

    # Print summary table - F1-macro
    print('\n')
    print('=' * 100)
    print('ABLATION STUDY RESULTS - F1-macro (%)')
    print('=' * 100)

    header = f"{'Variant':<25}"
    for ds in args.datasets:
        header += f"| {ds:^20} "
    print(header)
    print('-' * len(header))

    for variant in args.variants:
        label = VARIANT_LABELS.get(variant, variant)
        row = f"{label:<25}"
        for ds in args.datasets:
            _, _, f1, f1_std = results[ds].get(variant, (None, None, None, None))
            if f1 is not None:
                row += f"| {f1:.2f} ± {f1_std:.2f}      "
            else:
                row += f"| {'N/A':^20} "
        print(row)

    print('=' * 100)

    # Also print as markdown table - Accuracy
    print('\n### Markdown Table - Test Accuracy (%)\n')
    md_header = "| Variant |"
    md_sep = "|---------|"
    for ds in args.datasets:
        md_header += f" {ds} |"
        md_sep += "-----------|"
    print(md_header)
    print(md_sep)

    for variant in args.variants:
        label = VARIANT_LABELS.get(variant, variant)
        row = f"| {label} |"
        for ds in args.datasets:
            acc, acc_std, _, _ = results[ds].get(variant, (None, None, None, None))
            if acc is not None:
                row += f" {acc:.2f} ± {acc_std:.2f} |"
            else:
                row += " N/A |"
        print(row)

    # Markdown table - F1-macro
    print('\n### Markdown Table - F1-macro (%)\n')
    md_header = "| Variant |"
    md_sep = "|---------|"
    for ds in args.datasets:
        md_header += f" {ds} |"
        md_sep += "-----------|"
    print(md_header)
    print(md_sep)

    for variant in args.variants:
        label = VARIANT_LABELS.get(variant, variant)
        row = f"| {label} |"
        for ds in args.datasets:
            _, _, f1, f1_std = results[ds].get(variant, (None, None, None, None))
            if f1 is not None:
                row += f" {f1:.2f} ± {f1_std:.2f} |"
            else:
                row += " N/A |"
        print(row)


if __name__ == '__main__':
    main()

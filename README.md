# UGNN: Uncertainty-Aware Graph Neural Network with Label-Leakage-Free Structural Priors

**Uncertainty-Aware Graph Neural Network with Label-Leakage-Free Structural Priors for Node Classification on Heterophilous Graphs**

Zhiguo Xiao, Junli Liu, Xinyao Cao

*Knowledge-Based Systems (KBS), 2026*

---

## Overview

UGNN is an uncertainty-aware graph neural network designed for semi-supervised node classification on heterophilous graphs. It addresses three key challenges:

1. **Label-Leakage-Free Structural Priors**: Estimates global and local homophily priors exclusively from training nodes and training–training edges, preventing validation/test labels from participating in structural modeling.
2. **Dual Uncertainty Estimation**: Decomposes edge-level uncertainty into aleatoric uncertainty (inherent edge noise) and epistemic uncertainty (insufficient training information), enabling differentiated structural operations.
3. **Uncertainty-Driven Structure Adjustment & Dual-Channel Message Passing**: Performs edge reweighting, structure refinement, and separates edges into homophilic/heterophilic channels for differentiated propagation.

![UGNN Architecture](https://img.shields.io/badge/Architecture-Dual_Channel-blue) ![Python](https://img.shields.io/badge/Python-3.9+-green) ![PyTorch](https://img.shields.io/badge/PyTorch-1.11+-orange) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Quick Start

```bash
# 1. Install dependencies via conda
conda env create -f requirement.yaml
conda activate pyg

# 2. Run UGNN on the Actor dataset (default)
python -m ugnn.main -D actor -M EdgeNCSAGE

# 3. Run baseline comparison across all datasets
python baselines/run_comparison.py

# 4. Run ablation study
python ablation/run_ablation.py --datasets actor chameleon
```

Datasets are automatically downloaded on first run. Results are reported over 5 independent runs with mean ± std.

---

## Project Structure

```
UGNN/
├── ugnn/                        # Core package
│   ├── __init__.py
│   ├── config.py                 # Configuration and argument parsing
│   ├── datasets.py               # Dataset loading and split management
│   ├── main.py                   # Main training loop
│   ├── models.py                 # UGNN models and components
│   └── utils.py                  # Training, evaluation, and gate utilities
├── baselines/                    # Baseline model implementations
│   ├── __init__.py
│   ├── models_gnn.py             # GCN, GAT, GraphSAGE, FAGCN
│   ├── models_hetero.py          # H2GCN
│   ├── models_gpr.py             # GPR-GNN
│   ├── models_acm.py             # ACM-GCN
│   ├── models_linkx.py           # LINKX
│   ├── models_prognn.py          # Pro-GNN
│   ├── trainer.py                # Unified baseline training loop
│   └── run_comparison.py         # Comparison experiment runner
├── ablation/
│   └── run_ablation.py           # Ablation study runner
├── requirements.txt              # Python dependencies
├── requirement.yaml              # Conda environment specification
├── pyproject.toml                # Package configuration
├── Makefile                      # Development shortcuts
├── .gitignore
├── .pre-commit-config.yaml
└── README.md
```

---

## Environment Setup

### Option 1: Conda (Recommended)

```bash
conda env create -f requirement.yaml
conda activate pyg
```

### Option 2: Pip

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Install PyTorch Geometric and extensions
pip install torch-geometric==2.1.0
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torch-sparse==0.6.15 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html

# Install other dependencies
pip install -r requirements.txt
```

> **Note**: Adjust the CUDA version (`cu113`, `cu116`, `cu117`, etc.) and PyTorch version according to your hardware. For the latest installation instructions of PyTorch Geometric, please refer to [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric#installation).

### Option 3: Install as Editable Package

```bash
make install
# or
pip install -e ".[dev]"
```

This installs the `ugnn` package in editable mode along with development tools (`ruff`, `pre-commit`).

---

## Data Preparation

All datasets are **automatically downloaded** on the first run via PyTorch Geometric's built-in downloaders. No manual data preparation is required.

### Supported Datasets

| Dataset | Nodes | Edges | Features | Classes | Homophily | Type |
|:--------|------:|------:|---------:|--------:|----------:|:-----|
| Cora | 2,708 | 10,556 | 1,433 | 7 | 0.82 | Homophilous |
| CiteSeer | 3,327 | 9,104 | 3,703 | 6 | 0.70 | Homophilous |
| PubMed | 19,717 | 88,648 | 500 | 3 | 0.79 | Homophilous |
| Photo | 7,650 | 238,162 | 745 | 8 | 0.84 | Homophilous |
| Actor | 7,600 | 30,019 | 932 | 5 | 0.20 | Heterophilous |
| Chameleon | 2,277 | 36,101 | 2,325 | 5 | 0.23 | Heterophilous |
| Squirrel | 5,201 | 217,073 | 2,089 | 5 | 0.22 | Heterophilous |
| Texas | 183 | 325 | 1,703 | 5 | 0.11 | Heterophilous |
| Cornell | 183 | 298 | 1,703 | 5 | 0.21 | Heterophilous |
| Wisconsin | 251 | 515 | 1,703 | 5 | 0.14 | Heterophilous |

### Data Download Sources

Most datasets (Cora, CiteSeer, PubMed, Photo, Actor) are automatically downloaded via PyG's built-in dataset classes.

For **Chameleon** and **Squirrel**, the code loads from local raw files. If automatic download fails, please manually download the raw data from [geom-gcn](https://github.com/graphdml-uiuc-jlu/geom-gcn) and place the files as follows:

```
data/chameleon/raw/out1_node_feature_label.txt
data/chameleon/raw/out1_graph_edges.txt
data/squirrel/raw/out1_node_feature_label.txt
data/squirrel/raw/out1_graph_edges.txt
```

For **Texas, Cornell, Wisconsin**, the code attempts to download from [geom-gcn](https://github.com/graphdml-uiuc-jlu/geom-gcn/tree/master/new_data). If download fails, please manually place the files:

```
data/texas/raw/out1_node_feature_label.txt
data/texas/raw/out1_graph_edges.txt
data/cornell/raw/out1_node_feature_label.txt
data/cornell/raw/out1_graph_edges.txt
data/wisconsin/raw/out1_node_feature_label.txt
data/wisconsin/raw/out1_graph_edges.txt
```

---

## Usage

### 1. Run UGNN

```bash
# Run on Actor dataset with EdgeNCSAGE (default)
python -m ugnn.main -D actor -M EdgeNCSAGE

# Run on Cora dataset with EdgeNCGCN
python -m ugnn.main -D cora -M EdgeNCGCN

# Run on Chameleon dataset
python -m ugnn.main -D chameleon -M EdgeNCSAGE

# Run on Texas dataset
python -m ugnn.main -D texas -M EdgeNCSAGE
```

### 2. Ablation Study

```bash
# Run all ablation variants across all datasets
python ablation/run_ablation.py

# Run ablation on specific datasets
python ablation/run_ablation.py --datasets actor chameleon

# Run specific ablation variants only
python ablation/run_ablation.py --variants wo_prior wo_consistency
```

**Available ablation modes:**

| Mode | Description |
|:-----|:------------|
| `full` | Complete UGNN (default) |
| `wo_prior` | Without data-driven informed prior |
| `wo_local_prior` | Without local prior (global prior only) |
| `wo_consistency` | Without consistency uncertainty estimation |
| `wo_refinement` | Without uncertainty-driven graph refinement |
| `wo_dual_uncertainty` | Without dual uncertainty (binary classification only) |
| `wo_dual_channel` | Without dual-channel architecture (single-channel) |

### 3. Baseline Comparison

```bash
# Run UGNN and all baselines on all datasets
python baselines/run_comparison.py

# Run on specific datasets only
python baselines/run_comparison.py --datasets actor chameleon

# Run specific baselines only
python baselines/run_comparison.py --models GCN GAT GraphSAGE

# Skip UGNN (already have results)
python baselines/run_comparison.py --skip-ugnn
```

**Available baselines:** GCN, GAT, GraphSAGE, FAGCN, H2GCN, GPR-GNN, ACM-GCN, LINKX, Pro-GNN

### 4. Makefile Shortcuts

| Command | Description |
|:--------|:------------|
| `make install` | Install package in editable mode with dev dependencies |
| `make run` | Run UGNN with default arguments |
| `make ablation` | Run all ablation experiments |
| `make comparison` | Run full baseline comparison |
| `make lint` | Run `ruff check` on the project |
| `make format` | Format code with `ruff format` |
| `make check` | Run lint + format check |
| `make clean` | Remove `__pycache__`, `.pyc`, and build artifacts |

---

## Key Arguments

### Core Arguments

| Argument | Default | Description |
|:---------|:--------|:------------|
| `-D, --dataset` | `actor` | Dataset name |
| `-M, --model` | `EdgeNCSAGE` | Model type: `EdgeNCSAGE` or `EdgeNCGCN` |
| `--ablation` | `full` | Ablation mode (see table above) |
| `-H, --hidden` | `256` | Hidden dimension |
| `--lr` | `1e-2` | Learning rate |
| `--wd` | `5e-5` | Weight decay |
| `--dp1` | `0.0` | Dropout rate for first GNN layer |
| `--dp2` | `0.9` | Dropout rate for second GNN layer |
| `--act` | `relu` | Activation function |
| `--hops` | `1` | Number of hops for aggregation |
| `--forcing` | `0` | Forcing mode (0 or 1) |
| `--addself, -A` | `1` | Add self-loops (0 or 1) |
| `--threshold, -T` | `0.3` | Threshold for edge classification |
| `--finalagg` | `add` | Final aggregation mode |
| `-S, --baseseed` | `42` | Base random seed |

### Gate Hyperparameters

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--gate-temperature` | `1.0` | Gumbel-Softmax temperature |
| `--gate-min` | `0.0` | Minimum gate value |
| `--gate-max` | `1.0` | Maximum gate value |
| `--gate-hidden` | `256` | Gate MLP hidden dimension |
| `--gate-dropout` | `0.5` | Gate MLP dropout rate |
| `--gate-epochs` | `20` | Gate MLP training epochs per update |
| `--gate-batch-size` | `65536` | Gate MLP training batch size |
| `--gate-update-freq` | `5` | Gate update frequency (epochs) |

### Contribution I: Data-Driven Prior

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--kl-beta` | `1e-2` | KL divergence prior weight (α) |

### Contribution II: Consistency Uncertainty

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--consist-beta` | `5e-3` | Consistency regularization weight (β) |
| `--mc-samples` | `10` | MC Dropout sampling times (T) |

### Contribution III: Structure Refinement

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--refine-beta` | `1e-4` | Refinement edge quality loss weight |
| `--epi-threshold` | `0.5` | Epistemic uncertainty threshold (τ_e) |
| `--ale-threshold` | `0.7` | Aleatoric uncertainty threshold (τ_a) |
| `--knn-k` | `3` | KNN refinement neighbor count (k) |
| `--refine-freq` | `10` | Structure refinement frequency (epochs) |
| `--max-repair-nodes` | `500` | Maximum nodes per refinement step |

### Auxiliary

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--uncertainty-gamma` | `1.0` | Uncertainty scaling factor |
| `--log-uncertainty` | `0` | Print uncertainty statistics during training (0 or 1) |

### Data Split Management

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--split-dir` | `./splits` | Directory for saving/loading data splits |
| `--use-saved-splits` | `1` | Load existing splits if available (0 or 1) |
| `--save-splits` | `1` | Save generated splits to disk (0 or 1) |

---

## Training Protocol

- **Epochs**: Maximum 500 training epochs per run
- **Early Stopping**: Patience of 100 epochs on validation accuracy
- **Independent Runs**: 5 runs with seeds `baseseed + {0,1,2,3,4}`
- **Data Split**: 60%/20%/20% train/val/test split
- **Metrics**: Test Accuracy (mean ± std) and Macro F1-Score (mean ± std)
- **Model Selection**: Best validation accuracy determines the final test result

---

## Expected Output

A typical training run produces output like:

```
======================================================================
Namespace(dataset='actor', model='EdgeNCSAGE', baseseed=42, hidden=256, ...)
======================================================================
load actor successfully!
======================================================================
UGNN V05-fixed initialized (ablation=full)
======================================================================
  0%|          | 0/5 [00:00<?, ?it/s]
epoch:000 | loss:1.4523 | val:0.5234 | test:0.5102 | gate:0.3241 (bce:0.2891 kl:0.0123 consist:0.0089 refine:0.0012)
epoch:001 | loss:1.3987 | val:0.5412 | test:0.5287 | gate:0.2987 (bce:0.2654 kl:0.0112 consist:0.0078 refine:0.0009)
...
epoch:150 | loss:0.8234 | val:0.6123 | test:0.5987 | gate:0.1823 (bce:0.1456 kl:0.0089 consist:0.0045 refine:0.0003)
  [早停] epoch 150, best_val=0.6234
...
======================================================================
actor [full] valid_acc: 62.34 ± 1.23
actor [full] test_acc: 59.87 ± 0.98
actor [full] test_f1_macro: 58.45 ± 1.12
```

The ablation and comparison runners additionally output summary tables in both plain text and Markdown format.

---

## Reproducing Paper Results

To reproduce the main results in Tables 2 and 3 of the paper:

```bash
# Homophilous datasets
python -m ugnn.main -D cora -M EdgeNCSAGE
python -m ugnn.main -D citeseer -M EdgeNCSAGE
python -m ugnn.main -D pubmed -M EdgeNCSAGE
python -m ugnn.main -D photo -M EdgeNCSAGE

# Heterophilous datasets
python -m ugnn.main -D actor -M EdgeNCSAGE
python -m ugnn.main -D chameleon -M EdgeNCSAGE
python -m ugnn.main -D squirrel -M EdgeNCSAGE
python -m ugnn.main -D texas -M EdgeNCSAGE
python -m ugnn.main -D cornell -M EdgeNCSAGE
python -m ugnn.main -D wisconsin -M EdgeNCSAGE
```

All experiments use a 60%/20%/20% train/val/test split with 5 independent runs, reporting mean accuracy and standard deviation. The data splits are automatically saved to `splits/` and reused across UGNN and baselines for fair comparison.

---

## Architecture Details

### Model Variants

| Model | Description |
|:------|:------------|
| `EdgeNCSAGE` | Dual-channel GraphSAGE-based model with uncertainty-aware edge gating (default) |
| `EdgeNCGCN` | Dual-channel GCN-based model with uncertainty-aware edge gating |
| `SingleChannelSAGE` | Single-channel ablation variant (no same/diff separation) |

### Core Components

- **StochasticGateMLP**: Variational edge gate network that predicts edge-wise gate values with uncertainty estimates via reparameterization trick
- **InformedPrior**: Computes global and local homophily priors from training–training edges only
- **NeutralPrior**: Ablation prior with neutral (0.5) homophily assumption
- **ConsistencyUncertaintyEstimator**: MC Dropout-based epistemic uncertainty estimation
- **UncertaintyDrivenGraphRefiner**: Classifies edges into 4 categories (A/B/C/D) and performs kNN-based structure repair for high-uncertainty edges

### Edge Classification

The refiner classifies each edge based on normalized aleatoric (τ_a) and epistemic (τ_e) uncertainty thresholds:

| Class | Epistemic | Aleatoric | Treatment |
|:------|:----------|:----------|:----------|
| A | High | Low | kNN repair (add same-label edges) |
| B | High | High | Heavily down-weighted (weight=0.05) |
| C | Low | High | Moderately down-weighted (1-ale_norm) |
| D | Low | Low | Keep original weight |

---

## Key Design Principles

### Label Leakage Prevention

UGNN strictly prevents label leakage in the semi-supervised setting:

- **Structural priors**: Global and local homophily priors are estimated exclusively from training–training edges (both endpoints in the training set).
- **Edge semantic supervision**: Edge homophily labels are constructed only on training–training edges.
- **Structure refinement**: New edges are added based on model predictions and feature similarity, without accessing validation/test labels.
- **Evaluation**: Validation labels are used only for model selection and early stopping; test labels are never accessed during training.

### Unified Data Splits

All methods (UGNN and baselines) share the same data splits, saved to `splits/` directory. This ensures:
- Fair comparison between methods
- Reproducibility across experiments
- Consistent evaluation protocol

---

## Troubleshooting

### CUDA Out of Memory

Reduce `--gate-batch-size` (default: 65536) or `--hidden` dimension:

```bash
python -m ugnn.main -D actor --gate-batch-size 10248 --hidden 128
```

### PyTorch Geometric Installation Fails

Ensure your PyTorch version matches the `torch-scatter` and `torch-sparse` wheel. Check available wheels at:
- https://data.pyg.org/whl/

### Chameleon/Squirrel Data Download Fails

Manually download from [geom-gcn](https://github.com/graphdml-uiuc-jlu/geom-gcn) and place files in:
```
data/chameleon/raw/
data/squirrel/raw/
```

### Reproducibility Issues

Set `--baseseed` to a fixed value. Each run uses `baseseed + {0,1,2,3,4}` for the 5 independent runs. Disable cuDNN nondeterminism by setting `torch.backends.cudnn.deterministic = True` in your environment.

---


---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Format your code: `make format`
4. Run lint checks: `make check`
5. Commit your changes (`git commit -m 'Add your feature'`)
6. Push to the branch (`git push origin feature/your-feature`)
7. Open a Pull Request

Before submitting a PR, please ensure:
- All existing tests pass
- New code follows the existing style (`make format`)
- You've added appropriate comments for non-obvious logic

---

## Acknowledgments

We thank the authors of the following open-source projects for making their code available:

- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [GPR-GNN](https://github.com/jianhao2016/GPRGNN)
- [H2GCN](https://github.com/GemsLab/H2GCN)
- [FAGCN](https://github.com/bdy9527/FAGCN)
- [ACM-GCN](https://github.com/132709/ACM-GCN)
- [LINKX](https://github.com/CUAI/LinkHomo)
- [Pro-GNN](https://github.com/ChandlerBang/Pro-GNN)
- [geom-gcn](https://github.com/graphdml-uiuc-jlu/geom-gcn)

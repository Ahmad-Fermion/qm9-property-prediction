# QM9 Molecular Property Prediction with Graph Neural Networks

A compact PyTorch Geometric project for predicting small-molecule quantum-chemical properties from the QM9 dataset.

This project trains a message-passing graph neural network on QM9 molecular graphs to predict the HOMO-LUMO gap. The dataset contains DFT-computed quantum-chemical properties for approximately 134k small organic molecules composed of C, H, O, N, and F.

## Why this project

This repository demonstrates:

- molecular graph learning with PyTorch Geometric
- small-molecule quantum-property prediction
- RDKit/PyG-based molecular featurization
- reproducible dataset splitting, training, evaluation, and plotting
- clean workflow structure suitable for local or HPC training

## Dataset

Default dataset loader: `torch_geometric.datasets.QM9`.

The code downloads and processes QM9 automatically the first time it runs.

Useful source links:

- PyTorch Geometric QM9 documentation: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html
- Original QM9 collection: https://doi.org/10.6084/m9.figshare.c.978904.v5
- MoleculeNet datasets: https://moleculenet.org/datasets-1
- DeepChem QM9 CSV mirror: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv
- DeepChem QM9 tar mirror: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.tar.gz

Do not commit the downloaded dataset to GitHub. It is ignored in `.gitignore`.

## Installation

### Option 1: pip

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torch_geometric rdkit numpy pandas matplotlib scikit-learn tqdm
```

If `rdkit-pypi` fails, use `rdkit`, not `rdkit-pypi`.

### Option 2: conda/mamba

```bash
mamba create -n qm9-gnn python=3.11 -y
mamba activate qm9-gnn
pip install torch torch_geometric rdkit numpy pandas matplotlib scikit-learn tqdm
```

## Quick smoke test

Run this first on your workstation. It uses only 2,000 molecules and 2 epochs.

```bash
python src/train.py --subset 2000 --epochs 2 --batch-size 64 --hidden-dim 64
```

Expected outputs:

```text
checkpoints/best_model.pt
figures/learning_curve.png
figures/predicted_vs_actual.png
results/metrics.json
```

## More serious local run

```bash
python src/train.py --subset 20000 --epochs 50 --batch-size 64 --hidden-dim 128
```

## Full QM9 training

Run this on a GPU node or HPC system:

```bash
python src/train.py --subset -1 --epochs 100 --batch-size 128 --hidden-dim 128
```

## Targets

Default target is the HOMO-LUMO gap.

PyG QM9 target indices include:

| Index | Property |
|---:|---|
| 2 | HOMO energy |
| 3 | LUMO energy |
| 4 | HOMO-LUMO gap |
| 5 | Electronic spatial extent |
| 12 | Zero-point vibrational energy |
| 13 | Internal energy at 0 K |
| 16 | Free energy at 298.15 K |
| 17 | Heat capacity |

Example:

```bash
python src/train.py --target-idx 13 --target-name U0 --subset 20000 --epochs 50
```

## Repository structure

```text
qm9-property-prediction/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── src/
│   ├── model.py
│   └── train.py
└── scripts/
    └── run_hpc3.slurm
```

## Notes

This is a lightweight demonstration project, not a production benchmark. The goal is to show a clean molecular ML workflow: dataset loading, target selection, graph neural network training, evaluation, and reproducible outputs.

For better performance, next steps include:

- using 3D equivariant models
- adding multiple target prediction
- hyperparameter sweeps
- scaffold or molecule-size split analysis
- uncertainty estimation

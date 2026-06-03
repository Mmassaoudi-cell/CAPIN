# CAPIN — Constraint-Aware Physics-Informed Neural Network for Smart Grid Intrusion Detection

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.7-3c9.svg)](https://pytorch-geometric.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Conference](https://img.shields.io/badge/IECON-2026%20(submitted)-orange.svg)](https://www.ieee-ies.org/)


Official implementation of the paper:

> **A Constraint-Aware Physics-Informed Neural Network for Enhanced Intrusion Detection in Cyber-Physical Smart Grids**  
> Mohamed Massaoudi, Maymouna Ez Eddin  
> *Sustainable Energy, Grids and Networks Journal, Elsevier (under review)*
![The CAPIN Architecture](Flowshart.png)
---

## Overview

CAPIN is a hybrid intrusion-detection framework for cyber-physical smart grids. It addresses the core limitation of purely statistical classifiers: a measurement vector can look normal in isolation even when its voltage, current, impedance, frequency, and relay-log values are mutually inconsistent according to fundamental electrical laws.

CAPIN addresses this by injecting physical knowledge at three points in the training pipeline:

1. **Physics-informed feature engineering** — raw PMU and relay measurements are transformed into physically meaningful residuals (Ohm-law mismatch, Kirchhoff current balance, power imbalance, frequency stability, relay coordination).
2. **Constraint-guided sample weighting** — training samples whose measurements violate physical residuals receive higher loss weight, sharpening the decision boundary around physics-inconsistent attack vectors.
3. **Validation-optimised ensemble fusion** — a neural branch (MLP) and three tree-based models (XGBoost, Random Forest, Extra Trees) are combined with weights optimised directly on the validation log-loss.

### Key results on the Mississippi State ICS dataset (stratified test split, 15,676 samples)

| Metric | CAPIN | Best baseline (RF) |
|--------|------:|-------------------:|
| Accuracy | 91.1 % | 89.7 % |
| Precision | 96.4 % | 95.0 % |
| Recall | 90.9 % | 90.3 % |
| F1 | 93.6 % | 92.6 % |
| ROC-AUC | 97.5 % | 96.6 % |
| PR-AUC | 98.9 % | 98.4 % |
| FPR | 8.3 % | 11.7 % |

---

## Repository structure

```
.
├── capinn.py          # Full CAPIN implementation (single self-contained script)
├── README.md          # This file
└── data/              # Place the dataset CSV files here (not included)
    ├── data1.csv
    ├── data2.csv
    └── ...
```

---

## Dataset

CAPIN uses the **Mississippi State University ICS Cyberattack Dataset** (binary split), publicly available from the Critical Infrastructure Protection Center:

> Morris, T. et al. (2015). *Industrial Control System (ICS) Cyber Attack Datasets.*  
> https://www.ece.msstate.edu/~pvs/files/research/ICS_Cyberattack_Dataset.html

The binary variant (`binaryAllNaturalPlusNormalVsAttacks`) contains **78,377 samples** across 37 scenarios:
- **22,714 Normal** — line faults, maintenance switching, load variation
- **55,663 Attack** — data injection, remote tripping, relay-setting changes, coordinated multi-vector attacks

Each sample has **128 features**: 116 PMU electrical measurements (4 relay groups × voltage magnitude/angle, current magnitude/angle, frequency, frequency derivative, impedance, status) and 12 cyber/control/log channels (control-panel logs, relay logs, Snort alerts).

After downloading, place all `data*.csv` files in a single folder (e.g. `data/`) and pass it to `--data_dir`.

---

## Installation

```bash
pip install numpy pandas scikit-learn scipy xgboost
```

`xgboost` is optional; CAPIN falls back to an additional Random Forest if it is not available.

Python **3.9 or later** is required.

---

## Quick start

```bash
# Run the full experimental suite (baselines, ablation, intensity analysis)
python capinn.py --data_dir ./data

# Faster run — skip the MLP branch and intensity sweep
python capinn.py --data_dir ./data --no_nn --skip_intensity

# Save all metric tables as CSV files
python capinn.py --data_dir ./data --out_dir ./results

# Reproduce the paper's exact numbers
python capinn.py --data_dir ./data --seed 2026
```

---

## Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `./data` | Folder with the dataset CSV files |
| `--seed` | `2026` | Master random seed for all splits and models |
| `--out_dir` | *(none)* | If set, save result CSVs to this directory |
| `--no_nn` | off | Disable the MLP branch (tree ensemble only) |
| `--skip_baselines` | off | Skip conventional baseline evaluation |
| `--skip_ablation` | off | Skip ablation study |
| `--skip_intensity` | off | Skip attack-intensity sensitivity analysis |
| `--skip_data_dependency` | off | Skip training-fraction experiment |
| `--skip_scenario_block` | off | Skip scenario-block holdout validation |

---

## Experiments

### Model comparison
Evaluates CAPIN against five conventional baselines (ANN, Logistic Regression, Random Forest, AdaBoost, XGBoost) on the same stratified 64/16/20 train/validation/test split. All models use the same raw features; only CAPIN receives physics features and constraint-guided weights.

### Ablation study
Quantifies the contribution of each CAPIN component by systematically disabling:
- Physics-informed features
- Constraint-guided sample weighting
- The neural (MLP) branch
- The tree-based ensemble

### Data-dependency analysis
Trains CAPIN on 10 %, 25 %, 50 %, 75 %, and 100 % of the training partition. The held-out test set (15,676 samples) is fixed for all fractions.

### Attack-intensity analysis
Evaluates the trained CAPIN model at four attack prevalence levels (α = 5 %, 10 %, 20 %, 30 %) by subsampling attack test samples while retaining all normal samples. Each level is repeated with three random seeds (2026–2028) and results are reported as mean ± std.

### Scenario-block holdout validation
An 80/20 split on complete data-acquisition scenario blocks (rather than individual rows) to assess generalization when entire recording sessions are withheld from training.

---

## Architecture

```
Raw PMU / relay measurements  (128 features)
         │
         ├──► Physics feature engineering
         │         Ohm residual, KCL proxy, power balance error,
         │         frequency stability, voltage/current imbalance,
         │         log-electrical cross-terms
         │                 │
         │         Normalized constraint residuals
         │         (used as input features AND sample weights)
         │
         └──────────────────────────┐
                                    ▼
                    Augmented feature matrix  z_i = [x_i, ψ(x_i), c̄(x_i)]
                    Sample weights  ω_i = ω_class × (1 + 0.35 × C̄)
                                    │
                    ┌───────────────┼──────────────┐
                    ▼               ▼              ▼
                 XGBoost     Random Forest    Extra Trees
                    │               │              │
                    └───────────────┼──────────────┘
                                    │
                               MLP branch
                     (32 → 16, ReLU, early stopping)
                                    │
                  Validation-optimised ensemble weights α
                  (SLSQP: min log-loss, Σα=1, α≥0)
                                    │
                       Combined probability p̂_i
                                    │
                       Validation-tuned threshold τ
                    (max 0.5·F1 + 0.5·balanced_acc - 0.15·FPR)
                                    │
                            Binary prediction ŷ_i
```

---

## Physics constraints

| Constraint | Formula | Physical meaning |
|------------|---------|-----------------|
| Power balance | σ(P_r) / (μ(\|P_r\|) + ε) | Inter-relay power inconsistency |
| Ohm-law residual | \|V − I·Z\| / (\|V\| + ε) | Voltage–current–impedance mismatch |
| Kirchhoff current | \|Σ I_j\| / (Σ\|I_j\| + ε) | Current-balance violation proxy |
| Frequency stability | max(f_r) − min(f_r) | PMU frequency disagreement |
| dF/dt indicator | max_r \|DF_r\| | Fast frequency-change indicator |
| Voltage imbalance | σ(V) / (μ(\|V\|) + ε) | Voltage measurement dispersion |
| Current imbalance | σ(I) / (μ(\|I\|) + ε) | Current measurement dispersion |
| Protection activity | 1{Σ logs > 0} | Relay / control / Snort log activation |

---

## Reproducibility

All random operations use fixed seeds. To reproduce the numbers in the paper exactly:

```bash
python capinn.py --data_dir ./data --seed 2026
```

The stratified split produces **50,160 training / 12,541 validation / 15,676 test** samples. Attack-intensity experiments use seeds 2026, 2027, and 2028.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{massaoudi2025capin,
  title     = {A Constraint-Aware Physics-Informed Neural Network for Enhanced
               Intrusion Detection in Cyber-Physical Smart Grids},
  author    = {Massaoudi, Mohamed and Ez Eddin, Maymouna},
  journal   = {Sustainable Energy, Grids and Networks},
  publisher = {Elsevier},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

This code is released for research and educational use. For commercial use, please contact the authors.

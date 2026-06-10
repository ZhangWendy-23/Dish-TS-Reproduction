# Dish-TS-Reproduction

[![AAAI 2023](https://img.shields.io/badge/AAAI-2023-blue?style=flat-square)](https://ojs.aaai.org/index.php/AAAI/article/view/25913)
[![arXiv](https://img.shields.io/badge/arXiv-2302.14829-b31b1b?style=flat-square)](https://arxiv.org/abs/2302.14829)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.1%2B-76b900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> **Dish-TS (AAAI 2023) — Paper Reproduction Coursework**
>
> Original Paper: [Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/25913)
>
> Based on [official Dish-TS code](https://github.com/weifantt/Dish-TS), with all parameters strictly aligned.
> Dish-TS proposes a Dual-Conet framework to address both intra-space and inter-space distribution shift in time series forecasting, achieving 20%+ average improvement.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation & Quick Start](#installation--quick-start)
3. [Project Structure](#project-structure)
4. [Paper Reproduction Guide](#paper-reproduction-guide)
   - [Experiment Overview (4 Tables)](#experiment-overview-4-tables)
   - [Table 1: Univariate Forecasting](#table-1-univariate-forecasting)
   - [Table 2: Multivariate Forecasting](#table-2-multivariate-forecasting)
   - [Table 3: RevIN vs Dish-TS Comparison](#table-3-revin-vs-dishts-comparison)
   - [Table 4: Long-horizon Forecasting](#table-4-long-horizon-forecasting)
   - [Quick Verification](#quick-verification)
   - [Simplified Experiments](#simplified-experiments)
   - [Preventing SSH Disconnection](#preventing-ssh-disconnection)
   - [Monitoring Progress](#monitoring-progress)
   - [Collecting Results](#collecting-results)
5. [Configuration Reference](#configuration-reference)
   - [CLI Arguments](#cli-arguments)
   - [Model Hyperparameters](#model-hyperparameters)
   - [Data Splits](#data-splits)
   - [Batch Size Rules](#batch-size-rules)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Expected Results](#expected-results)
8. [Citation](#citation)
9. [Acknowledgments](#acknowledgments)
10. [Notes](#notes)

---

## Requirements

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA RTX 3090 24GB (paper standard) or RTX 3080 Ti 12GB |
| **RAM** | >= 16 GB |
| **Disk** | >= 5 GB |
| **Python** | >= 3.8 |
| **PyTorch** | >= 1.8.0 |
| **CUDA** | >= 11.1 |

---

## Installation & Quick Start

Since the repository is public, you can clone directly without SSH keys.

### Option 1: Clone via HTTPS (recommended, no SSH required)

```bash
git clone https://github.com/ZhangWendy-23/Dish-TS-Reproduction.git
cd Dish-TS-Reproduction
pip install -r requirements.txt
```

### Option 2: Clone via SSH

```bash
git clone git@github.com:ZhangWendy-23/Dish-TS-Reproduction.git
cd Dish-TS-Reproduction
pip install -r requirements.txt
```

### Verify Installation

```bash
nvidia-smi                         # Should show RTX 3090 (24576 MiB) or similar
ls -lh data/                       # Should list 5 CSV files
python -c "from backbones import Autoformer; from DishTS import DishTS; print('Installation OK')"
```

> **SSH disconnection will kill your experiments.** Always use `screen` / `tmux`
> (see [Preventing SSH Disconnection](#preventing-ssh-disconnection)).

### Run All Paper Experiments

```bash
chmod +x run_paper_exps.sh
./run_paper_exps.sh all 2>&1 | tee logs/master_all.log
```

Full reproduction (542 experiments) takes ~2-5 days on a single RTX 3090.
See below for running individual tables.

---

## Project Structure

```
Dish-TS-Reproduction/
├── train.py                     # Main training script (CLI args, paper-aligned)
├── Model.py                     # Unified model wrapper (forecast backbone + norm model)
├── DishTS.py                    # Dish-TS core module (Dual-Conet with GELU activation)
├── REVIN.py                     # RevIN baseline implementation
├── .gitignore
├── requirements.txt
│
├── run_paper_exps.sh            # ★ Paper Table 1-4 one-click runner (recommended)
├── run_simplified_exps.sh       # Simplified 18-experiment subset (~1-3 hours)
│
├── backbones/                   # Forecasting backbone models
│   ├── __init__.py
│   ├── Autoformer.py            # Autoformer (auto-correlation mechanism)
│   ├── Informer.py              # Informer (prob-sparse self-attention)
│   ├── Transformer.py           # Vanilla Transformer
│   └── layers/                  # Reusable neural network components
│       ├── __init__.py
│       ├── Embed.py             # Data/temporal/positional embeddings
│       ├── Autoformer_EncDec.py # Autoformer-specific encoder/decoder with series decomposition
│       ├── Transformer_EncDec.py# Standard Transformer encoder/decoder
│       ├── AutoCorrelation.py   # Auto-correlation attention mechanism
│       ├── SelfAttention_Family.py # FullAttention + ProbAttention
│       ├── masking.py           # Causal masking and ProbMask
│       └── utils.py             # Utility functions
│
├── utils/                       # Utility modules
│   ├── __init__.py              # Random seed configuration
│   ├── dataset.py               # Data loader (M/S mode, 6:2:2 or 7:1:2 split)
│   ├── earlystop.py             # Early stopping mechanism
│   └── metric.py                # MSE/MAE/RMSE/MAPE/MSPE metrics
│
├── data/                        # 5 benchmark datasets (tracked in version control)
│   ├── ETTm2.csv  (~9.3 MB)    # 15-min level, 7 features, ~69,680 timestamps
│   ├── ETTh1.csv  (~2.5 MB)    # Hourly, 7 features, ~17,420 timestamps
│   ├── ECL.csv    (~92 MB)     # Hourly, 321 features, ~26,304 timestamps
│   ├── WTH.csv    (~7.0 MB)    # 10-min level, 21 features, ~52,696 timestamps
│   └── ILI.csv    (~67 KB)     # Weekly, 7 features, ~966 timestamps
│
├── results/                     # Result collection and analysis tools
│   └── collect_results.py       # Parse logs -> CSV + LaTeX tables + plots
│
├── logs/                        # Experiment logs (tracked for reproducibility)
│   ├── simplified/              # Simplified 18-experiment results
│   └── backup/                  # Historical backup
│
├── README.md                    # This file
└── IMPROVEMENTS.md              # Complete modification log (16 items)
```

---

## Paper Reproduction Guide

### Experiment Overview (4 Tables)

| Table | Description | seq_len | pred_len | Datasets | Models | Norms | Seeds | Total Exp. |
|-------|-------------|---------|----------|----------|--------|-------|-------|-------------|
| **1** | Univariate forecasting | = pred_len | {24, 48, 96, 168, 336} | All 5 | 3 | 3 | 1 | 225 |
| **2** | Multivariate forecasting | 96 | {24, 48, 96, 168, 336} | All 5 | 3 | 3 | 1 | 225 |
| **3** | RevIN vs. Dish-TS | 96 | {24, 168, 336} | 4 (excl. ILI) | Autoformer | 2 | 3 | 72 |
| **4** | Long-horizon forecasting | 96 | {336, 420, 540, 600, 720} | ECL, ETTh1 | Autoformer | 2 | 1 | 20 |

**Recommended execution order:**

| Order | Command | Why First | Experiments | Estimated Time |
|-------|---------|-----------|-------------|----------------|
| 1 | `./run_paper_exps.sh quick` | Verify environment works | 6 | ~10 min |
| 2 | `./run_paper_exps.sh table3` | Core result: RevIN vs Dish-TS | 72 | 4-6 hours |
| 3 | `./run_paper_exps.sh table2` | Main multivariate results | 225 | 18-24 hours |
| 4 | `./run_paper_exps.sh table1` | Univariate results | 225 | 12-20 hours |
| 5 | `./run_paper_exps.sh table4` | Long-horizon results | 20 | 5-8 hours |

### Table 1: Univariate Forecasting

Univariate setting (`--features S`). Input length equals prediction length.

```bash
./run_paper_exps.sh table1 2>&1 | tee logs/table1_master.log
```

### Table 2: Multivariate Forecasting

Core experiment. Fixed `seq_len=96`, `--features M`, 5 datasets.

```bash
./run_paper_exps.sh table2 2>&1 | tee logs/table2_master.log
```

### Table 3: RevIN vs. Dish-TS

Autoformer backbone, 3 seeds (2023/2024/2025), results reported as mean +/- std.

```bash
./run_paper_exps.sh table3 2>&1 | tee logs/table3_master.log
```

### Table 4: Long-horizon Forecasting

Prediction length gradually increases to test extrapolation (Autoformer only).

```bash
./run_paper_exps.sh table4 2>&1 | tee logs/table4_master.log
```

### Quick Verification

Run a smoke test (~10 minutes) to verify the environment is correctly configured before launching full experiments:

```bash
./run_paper_exps.sh quick
```

This runs 6 experiments on ETTm2 with Autoformer across 3 normalization methods and 2 prediction lengths (24, 96).

### Simplified Experiments

For limited time or compute resources, a simplified 18-experiment subset provides the main findings in 1-3 hours:

```bash
chmod +x run_simplified_exps.sh
nohup ./run_simplified_exps.sh > logs/simplified_master.log 2>&1 &
```

**Matrix**: ETTm2 x {Autoformer, Informer} x {none, revin, dishts} x {24, 96, 336}

### Collecting Results

After experiments complete (or at any point to check progress):

```bash
# Generate result summary
./run_paper_exps.sh summarize

# Generate LaTeX tables and comparison plots
python results/collect_results.py
```

Output files:

| File | Description |
|------|-------------|
| `results/paper_summary.csv` | Raw results for all completed experiments |
| `results/summary_aggregated.csv` | Multi-seed mean ± standard deviation |
| `results/table2_multivariate.tex` | LaTeX table for Table 2 |
| `results/table3_revin_vs_dishts.tex` | LaTeX table for Table 3 |
| `results/*.png` | MSE vs. prediction length comparison plots |

### Preventing SSH Disconnection

Always use a persistent session to protect experiments when SSH drops.

```bash
# screen (AutoDL: use -U to avoid encoding issues)
screen -U -S dishts          # start session
# Run experiments, then: Ctrl+A D → detach
screen -r dishts             # reconnect
screen -ls                   # list sessions

# tmux (install: apt-get install -y tmux -qq)
tmux new -s dishts           # start session
# Run experiments, then: Ctrl+B D → detach
tmux attach -t dishts        # reconnect
tmux ls                      # list sessions
```

`nohup` is a fallback but some cloud environments may still kill orphaned processes.

### Monitoring Progress

```bash
# Currently running experiments
ps aux | grep "train.py" | grep -v grep

# Latest log output
tail -50 logs/table3_master.log

# GPU utilization
nvidia-smi

# Summarize completed results at any time
./run_paper_exps.sh summarize
```

---

## Configuration Reference

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | `ETTm2` | Dataset: `ETTm2`, `ETTh1`, `ECL`, `WTH`, `ILI` |
| `--model` | str | `Transformer` | Forecast model: `Autoformer`, `Informer`, `Transformer` |
| `--norm` | str | `none` | Normalization: `none`, `revin`, `dishts` |
| `--features` | str | `M` | Feature mode: `M` (multivariate), `S` (univariate, Table 1) |
| `--seq_len` | int | `96` | Input/history window length |
| `--label_len` | int | `0` | Decoder input length. `0` = auto-compute as `max(48, pred_len//2)`, capped at seq_len |
| `--pred_len` | int | `96` | Prediction/horizon window length |
| `--batch_size` | int | `0` | `0` = auto-select per paper: Informer=256, Autoformer/Transformer=128; ECL=64 regardless |
| `--lr` | float | `1e-3` | Learning rate (paper search range: [1e-4, 1e-3]) |
| `--patience` | int | `7` | Early stopping patience epochs |
| `--seed` | int | `2023` | Random seed (paper uses 3 seeds for Table 3: 2023/2024/2025) |
| `--alpha` | float | `0.5` | Dish-TS prior knowledge guidance weight (paper search: 0 to 1) |
| `--dish_init` | str | `standard` | DishTS initialization: `standard` (GELU), `avg`, `uniform` |
| `--affine` | int | `1` | RevIN affine transformation: `1` = enabled, `0` = disabled |
| `--gpu` | int | `0` | GPU device ID |

### Model Hyperparameters

Hard-coded in `train.py`, identical to the original paper:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 512 | Hidden dimension |
| `n_heads` | 8 | Number of attention heads |
| `d_ff` | 2048 | Feed-forward network dimension |
| `e_layers` | 2 | Number of encoder layers |
| `d_layers` | 1 | Number of decoder layers |
| `dropout` | 0.05 | Dropout rate |
| `activation` | `gelu` | Activation function |
| `embed_type` | 3 | Embedding type (no temporal embedding) |
| `moving_avg` | 25 | Autoformer moving average kernel size |
| `factor` | 3 | Informer prob-sparse attention factor |
| `distil` | True | Informer distillation |

### Data Splits

Automatically selected based on dataset name, following the original paper:

| Datasets | Train | Validation | Test |
|----------|-------|------------|------|
| ETTm2, ETTh1, ILI | 60% | 20% | 20% |
| ECL, WTH | 70% | 10% | 20% |

All experiments are conducted on raw data without global normalization, consistent with the paper setting.

### Batch Size Rules

When `--batch_size 0` is specified (auto-selection):

| Model | Default Batch Size | ECL Dataset |
|-------|-------------------|-------------|
| Informer | 256 | 64 |
| Autoformer | 128 | 64 |
| Transformer | 128 | 64 |

Additionally, for `pred_len > 168`, batch size is automatically reduced to 64 to avoid CUDA out-of-memory errors.

---

## Evaluation Metrics

All metrics are computed on the original (unnormalized) scale.

| Metric | Formula | Range | Lower is Better? |
|--------|---------|-------|------------------|
| **MSE** | mean((y\_pred - y\_true)^2) | [0, +inf) | Yes |
| **MAE** | mean(\|y\_pred - y\_true\|) | [0, +inf) | Yes |
| **RMSE** | sqrt(MSE) | [0, +inf) | Yes |
| **MAPE** | mean(\|(y\_pred - y\_true) / y\_true\|) | [0, +inf) | Yes |
| **MSPE** | mean(((y\_pred - y\_true) / y\_true)^2) | [0, +inf) | Yes |

---

## Expected Results

The paper reports the following average improvements of Dish-TS over the baseline (no normalization):

- **Univariate forecasting**: **28.6%** average MSE reduction
- **Multivariate forecasting**: **21.9%** average MSE reduction

Reproduction success criterion: Dish-TS (`--norm dishts`) should achieve significantly lower MSE than both `none` (raw) and `revin` (RevIN baseline), especially at longer prediction horizons (pred_len >= 96).

---

## Citation

If you use this code or the Dish-TS method in your research, please cite the original paper:

```bibtex
@inproceedings{fan2023dish,
  title     = {Dish-{TS}: A General Paradigm for Alleviating Distribution Shift
               in Time Series Forecasting},
  author    = {Fan, Wei and Wang, Pengyang and Wang, Dongkun and
               Wang, Dongjie and Zhou, Yuanchun and Fu, Yanjie},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {37},
  number    = {6},
  pages     = {7522--7529},
  year      = {2023}
}
```

---

## Acknowledgments

This reproduction is based on the official [Dish-TS repository](https://github.com/weifantt/Dish-TS). We thank the authors of [Autoformer](https://github.com/thuml/Autoformer), [Informer](https://github.com/zhouhaoyi/Informer2020), and [RevIN](https://github.com/ts-kim/RevIN) for open-sourcing their code and datasets.

For a detailed record of all modifications made in this reproduction, see [IMPROVEMENTS.md](IMPROVEMENTS.md).

---

## Notes

- **Commit message encoding**: Some early commits display `???` in their messages. This is a known Git encoding artifact caused by Emoji/Unicode characters in early commit messages during initial setup. These are purely cosmetic and do not affect any code, results, or reproducibility.

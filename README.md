# Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting

[![AAAI 2023](https://img.shields.io/badge/AAAI-2023-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/25913)
[![arXiv](https://img.shields.io/badge/arXiv-2302.14829-b31b1b.svg)](https://arxiv.org/abs/2302.14829)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Torch](https://img.shields.io/badge/PyTorch-1.8%2B-ee4c2c.svg)](https://pytorch.org/)

> Official implementation of **"Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting"** (AAAI 2023).

Dish-TS is a **model-agnostic** neural paradigm that consistently boosts time series forecasting models by over **20%** on average, by addressing both **intra-space** and **inter-space** distribution shift through a Dual-Conet framework.

---

## Table of Contents

- [Background](#background)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Reproducing Paper Experiments](#reproducing-paper-experiments)
- [Arguments](#arguments)
- [Results](#results)
- [Citation](#citation)

---

## Background

### Problem

Time series forecasting (TSF) suffers from **distribution shift**: the statistical properties of time series change over time, which hinders model generalization. Existing works have two key limitations:

1. **Unreliable distribution quantification** — Fixed statistics (mean, std.) from observational data cannot reliably represent the true underlying distribution, especially under varying sampling frequencies.
2. **Neglected inter-space shift** — Most methods (e.g., RevIN) assume the lookback window (input-space) and horizon window (output-space) share the same distribution, ignoring the gap between them.

### Our Approach

We systematically categorize distribution shift into two types:

| Type | Definition | Addressed by Existing Work? |
|------|-----------|---------------------------|
| **Intra-space shift** | Distribution of lookback windows changes over time | Partially |
| **Inter-space shift** | Distribution differs between lookback and horizon windows | No |

Dish-TS proposes a **Dual-CoNet** framework:

- **BackCoNet** — Learns the distribution of the input-space (lookback window) for normalization.
- **HoriCoNet** — Learns the distribution of the output-space (horizon window) for denormalization.

**Pipeline**: `Input → BackCoNet Normalize → Forecast Model → HoriCoNet Denormalize → Output`

This design is **model-agnostic** and can be plugged into any forecasting architecture (Transformer, Autoformer, Informer, N-BEATS, etc.).

### Key Contributions

- Systematically define **intra-space** and **inter-space** distribution shift in TSF.
- Propose **Dish-TS**, a general neural paradigm with Dual-CoNet for alleviating both types of shift.
- Provide a **simple and effective CoNet instance** with prior knowledge-induced training.
- Achieve **28.6%** average improvement in univariate and **21.9%** in multivariate forecasting across 5 datasets.

---

## Project Structure

```
Dish-TS/
├── train.py                     # Main training script
├── Model.py                     # Unified model wrapper (forecast + normalization)
├── DishTS.py                    # Dish-TS normalization module (core contribution)
├── REVIN.py                     # RevIN baseline normalization
├── requirements.txt             # Python dependencies
│
├── backbones/                   # Forecasting model backbones
│   ├── __init__.py
│   ├── Transformer.py           # Vanilla Transformer
│   ├── Autoformer.py            # Autoformer (auto-correlation mechanism)
│   ├── Informer.py              # Informer (prob-sparse attention)
│   └── layers/                  # Reusable neural network components
│       ├── Embed.py             # Data/temporal/positional embeddings
│       ├── Transformer_EncDec.py # Transformer encoder & decoder
│       ├── Autoformer_EncDec.py  # Autoformer encoder & decoder
│       ├── SelfAttention_Family.py # FullAttention & ProbAttention
│       ├── AutoCorrelation.py   # Auto-correlation mechanism
│       ├── masking.py           # Causal & probabilistic masks
│       └── utils.py             # Utility functions
│
├── utils/                       # Utility modules
│   ├── __init__.py              # Random seed configuration
│   ├── dataset.py               # TSForecastDataset data loader
│   ├── earlystop.py             # Early stopping mechanism
│   └── metric.py                # Evaluation metrics (MAE, MSE, RMSE, MAPE, MSPE)
│
└── dataset/                     # Datasets directory
    └── ETTm2.csv                # ETTm2 dataset (15-minutely, 7 variables)
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.8+

### Setup

```bash
# Clone the repository
git clone https://github.com/weifantt/Dish-TS.git
cd Dish-TS

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from backbones import Autoformer; from DishTS import DishTS; print('Ready!')"
```

### Dependencies

| Package   | Version   | Purpose          |
|-----------|-----------|------------------|
| torch     | >= 1.8.0  | Deep learning framework |
| numpy     | >= 1.19.0 | Numerical computation |
| pandas    | >= 1.2.0  | Data manipulation |

---

## Data Preparation

### Dataset Format

Each dataset should be a CSV file under `dataset/` with a `date` column followed by feature columns:

```csv
date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
2016-07-01 00:00:00,41.13,12.48,36.54,9.35,4.42,1.31,38.66
2016-07-01 00:15:00,39.62,11.31,35.54,8.55,3.21,1.26,38.22
...
```

### Datasets Used in the Paper

| Dataset      | Granularity  | Variables | Timesteps | Download Link |
|-------------|-------------|-----------|-----------|---------------|
| **ETTm2**   | 15 minutes  | 7         | ~69,680   | Included (`dataset/ETTm2.csv`) |
| **ETTh1**   | 1 hour      | 7         | ~17,420   | [Autoformer Repo](https://github.com/thuml/Autoformer) |
| **Electricity** | 1 hour  | 321       | ~26,304   | [Autoformer Repo](https://github.com/thuml/Autoformer) |
| **Weather** | 10 minutes  | 21        | ~52,696   | [Autoformer Repo](https://github.com/thuml/Autoformer) |
| **Illness** | Weekly      | 7         | ~966      | [Autoformer Repo](https://github.com/thuml/Autoformer) |

To download additional datasets:

```bash
mkdir -p dataset
# Download from Autoformer's official repository:
# https://github.com/thuml/Autoformer/tree/main/dataset
```

### Data Splitting

The data is split chronologically:

| Split   | Ratio | Purpose                    |
|---------|-------|----------------------------|
| Train   | 70%   | Model training             |
| Val     | 10%   | Early stopping & validation |
| Test    | 20%   | Final evaluation           |

---

## Quick Start

Train a model with Dish-TS on ETTm2 using the default settings:

```bash
python train.py --data ETTm2 --model Transformer --norm dishts --gpu 0
```

Expected output:
```
epoch:0, train_loss:0.xxxxx, val_loss:0.xxxxx
epoch:1, train_loss:0.xxxxx, val_loss:0.xxxxx
...
Early stopping with best_score:0.xxxxx
       0           1          2     3   4   5       6       7       8
ETTm2  Transformer  dishts  2023  96  96  0.xxxxx  0.xxxxx  0.xxxxx
```

---

## Reproducing Paper Experiments

### 1. Preparation

```bash
# (1) Install dependencies
pip install -r requirements.txt

# (2) Verify dataset exists
ls dataset/ETTm2.csv

# (3) Download other datasets from Autoformer repo if needed
# Place them under dataset/ as {name}.csv
```

### 2. Experiment 1: Main Results — Compare Normalization Methods

Compare **none** (no normalization), **RevIN** (baseline), and **Dish-TS** across different forecast models and prediction lengths.

```bash
# Fix pred_len=96, test 3 forecast models × 3 normalization methods
for model in Transformer Informer Autoformer; do
  for norm in none revin dishts; do
    python train.py --data ETTm2 --model $model --norm $norm \
      --pred_len 96 --seq_len 96 --seed 2023 --gpu 0
  done
done

# Repeat for pred_len = 192, 336, 720
```

### 3. Experiment 2: Different Prediction Lengths

```bash
# Test prediction lengths: 96, 192, 336, 720
for pred_len in 96 192 336 720; do
  python train.py --data ETTm2 --model Autoformer --norm dishts \
    --pred_len $pred_len --seq_len 96 --seed 2023 --gpu 0
done
```

### 4. Experiment 3: Multi-Dataset Evaluation

```bash
# After downloading all datasets:
for data in ETTh1 ETTm2 Electricity Weather Illness; do
  python train.py --data $data --model Autoformer --norm dishts \
    --pred_len 96 --seq_len 96 --seed 2023 --gpu 0
done
```

### 5. Experiment 4: Dish-TS Initialization Methods

```bash
# Compare standard, avg, and uniform initialization
for init in standard avg uniform; do
  python train.py --data ETTm2 --model Autoformer --norm dishts \
    --dish_init $init --pred_len 96 --seq_len 96 --seed 2023 --gpu 0
done
```

### 6. Multi-Seed Evaluation (for Paper-Level Results)

The paper reports **mean ± std** across 5 random seeds:

```bash
# Run with 5 seeds and collect results
for seed in 2023 2024 2025 2026 2027; do
  python train.py --data ETTm2 --model Autoformer --norm dishts \
    --pred_len 96 --seq_len 96 --seed $seed --gpu 0 >> results.log
done

# Extract MSE values and compute mean ± std
grep "ETTm2" results.log | awk '{print $7}'
```

### 7. Full Automation Script

Save the following as `run_experiments.sh`:

```bash
#!/bin/bash
# run_experiments.sh — Full Dish-TS experiment suite

GPU=0
DATA=ETTm2
SEEDS=(2023 2024 2025 2026 2027)
NORMS=(none revin dishts)
MODELS=(Transformer Informer Autoformer)
PRED_LENS=(96 192 336 720)

echo "=== Main Comparison: Models × Norms (pred_len=96) ==="
for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    for norm in "${NORMS[@]}"; do
      echo "Running: $model + $norm, seed=$seed"
      python train.py --data $DATA --model $model --norm $norm \
        --pred_len 96 --seq_len 96 --seed $seed --gpu $GPU
    done
  done
done

echo "=== Long-term Forecasting (Autoformer + DishTS) ==="
for pred_len in "${PRED_LENS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    python train.py --data $DATA --model Autoformer --norm dishts \
      --pred_len $pred_len --seq_len 96 --seed $seed --gpu $GPU
  done
done

echo "=== All experiments complete ==="
```

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

---

## Arguments

### Forecast Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | `ETTm2` | Dataset name (file: `dataset/{data}.csv`) |
| `--model` | str | `Transformer` | Forecast backbone: `Transformer`, `Informer`, `Autoformer` |
| `--seq_len` | int | `96` | Lookback window length |
| `--label_len` | int | `48` | Known future length for decoder input |
| `--pred_len` | int | `96` | Horizon (prediction) length |
| `--features` | str | `M` | Feature type: `M` (multivariate), `S` (univariate) |

### Normalization Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--norm` | str | `none` | Normalization: `none`, `revin`, `dishts` |
| `--affine` | int | `1` | Use affine layers in RevIN (1=yes, 0=no) |
| `--dish_init` | str | `standard` | DishTS initialization: `standard`, `avg`, `uniform` |

### Training Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch_size` | int | `128` (`64` if pred_len > 168) | Batch size |
| `--lr` | float | `1e-3` | Learning rate |
| `--patience` | int | `5` | Early stopping patience |
| `--seed` | int | `2023` | Random seed |
| `--gpu` | int | `0` | GPU device ID |

### Model Hyperparameters (Fixed in `train.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 512 | Model hidden dimension |
| `n_heads` | 8 | Number of attention heads |
| `d_ff` | 2048 | Feed-forward network dimension |
| `e_layers` | 2 | Number of encoder layers |
| `d_layers` | 1 | Number of decoder layers |
| `dropout` | 0.05 | Dropout rate |
| `embed_type` | 3 | Embedding type (no temporal embedding) |
| `activation` | `gelu` | Activation function |
| `moving_avg` | 25 | Moving average kernel size (Autoformer) |
| `factor` | 3 | Prob-sparse factor (Informer) |
| `distil` | True | Use distilling (Informer) |

---

## Results

### Main Results: MSE on ETTm2 (Example Format)

Results format from the paper. Reproduce by running the multi-seed experiments above.

| Model        | Norm  | pred_len=96 | pred_len=192 | pred_len=336 | pred_len=720 |
|-------------|-------|-------------|--------------|--------------|--------------|
| Transformer | none  | _TBD_       | _TBD_        | _TBD_        | _TBD_        |
| Transformer | revin | _TBD_       | _TBD_        | _TBD_        | _TBD_        |
| Transformer | dishts| _TBD_       | _TBD_        | _TBD_        | _TBD_        |
| Autoformer  | none  | _TBD_       | _TBD_        | _TBD_        | _TBD_        |
| Autoformer  | revin | _TBD_       | _TBD_        | _TBD_        | _TBD_        |
| Autoformer  | dishts| _TBD_       | _TBD_        | _TBD_        | _TBD_        |
| Informer    | none  | _TBD_       | _TBD_        | _TBD_        | _TBD_        |
| Informer    | revin | _TBD_       | _TBD_        | _TBD_        | _TBD_        |
| Informer    | dishts| _TBD_       | _TBD_        | _TBD_        | _TBD_        |

*Replace _TBD_ with your experimental results after running.*

### Expected Improvement

Per paper results, Dish-TS achieves:

- **Univariate forecasting**: Average **28.6%** improvement over baselines
- **Multivariate forecasting**: Average **21.9%** improvement over baselines

### Metrics Explanation

| Metric | Formula | Range | Lower is Better? |
|--------|---------|-------|------------------|
| **MSE** | Mean Squared Error | [0, +∞) | Yes |
| **MAE** | Mean Absolute Error | [0, +∞) | Yes |
| **RMSE** | Root Mean Squared Error | [0, +∞) | Yes |
| **MAPE** | Mean Absolute Percentage Error | [0, +∞) | Yes |
| **MSPE** | Mean Squared Percentage Error | [0, +∞) | Yes |

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{fan2023dish,
  title     = {Dish-TS: A General Paradigm for Alleviating Distribution Shift 
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

## License

This project is released under the MIT License.

---

## Acknowledgements

We thank the authors of [Autoformer](https://github.com/thuml/Autoformer), [Informer](https://github.com/zhouhaoyi/Informer2020), and [RevIN](https://github.com/ts-kim/RevIN) for their open-source code and datasets, which form the backbone components and baselines used in this work.

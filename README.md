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
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
├── run_experiments.sh           # Automated experiment runner
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
git clone git@github.com:ZhangWendy-23/Dish-TS.git
# or via HTTPS:
# git clone https://github.com/ZhangWendy-23/Dish-TS.git
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

| Dataset      | File Name   | Granularity | Variables | Timesteps | Source |
|-------------|-------------|-------------|-----------|-----------|--------|
| **ETTm2**   | `ETTm2.csv` | 15 minutes  | 7         | ~69,680   | Included in this repo |
| **ETTh1**   | `ETTh1.csv` | 1 hour      | 7         | ~17,420   | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| **Electricity** | `ECL.csv` | 1 hour      | 321       | ~26,304   | [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) |
| **Weather** | `WTH.csv`  | 10 minutes  | 21        | ~52,696   | [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) |
| **Illness** | `ILI.csv`  | Weekly      | 7         | ~966      | [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) |

### Download Instructions

**ETT 系列（ETTm2 + ETTh1）— 通过命令行即可下载：**

```bash
cd dataset

# ETTh1（已内置 ETTm2，只需下载此文件）
wget -O ETTh1.csv "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"

# 验证
wc -l ETTh1.csv    # 应输出 17421
```

**Electricity / Weather / Illness — 需手动从 Google Drive 下载：**

1. 打开 Google Drive 链接：
   **https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy**

2. 下载以下三个文件到 `dataset/` 目录：

   | Google Drive 中文件名 | 保存为 | `--data` 参数值 |
   |----------------------|--------|-----------------|
   | `electricity.csv` | `ECL.csv` | `ECL` |
   | `weather.csv` | `WTH.csv` | `WTH` |
   | `national_illness.csv` | `ILI.csv` | `ILI` |

3. 验证所有数据集已就绪：

```bash
ls -lh dataset/*.csv
# 应看到 5 个文件: ETTm2.csv, ETTh1.csv, ECL.csv, WTH.csv, ILI.csv
```

> **注意**：即使没有 Electricity/Weather/Illness，也可以使用 ETTm2 和 ETTh1 完成大部分论文实验。这两个 ETT 数据集是论文中最重要的 benchmark。

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

### Quickest Way: Use the Automation Script

The repository includes `run_experiments.sh` which automates all paper experiments in 5 phases:

```bash
chmod +x run_experiments.sh

./run_experiments.sh phase1   # ETTm2: 3 models x 3 norms x 4 pred_lens (36 runs)
./run_experiments.sh phase2   # ETTh1: 3 models x 3 norms x 4 pred_lens (36 runs)
./run_experiments.sh phase3   # ECL/WTH/ILI: Autoformer x 2 norms x 4 pred_lens (24 runs)
./run_experiments.sh phase4   # Multi-seed paper-level results (60 runs)
./run_experiments.sh phase5   # DishTS initialization ablation (12 runs)

./run_experiments.sh all      # Run everything (~168 runs, several hours)
```

Results are saved to `logs/` directory. Phase 4 auto-computes mean +- std across seeds.

### Step-by-Step Manual Execution

If you prefer running experiments manually, here are the exact commands:

#### Phase 1: ETTm2 Full Comparison (1 seed)

ETTm2 is the primary benchmark included in this repo. Test all combinations of 3 forecast models x 3 normalization methods across 4 prediction lengths:

```bash
cd Dish-TS
mkdir -p logs

for model in Transformer Informer Autoformer; do
  for norm in none revin dishts; do
    for pred_len in 96 192 336 720; do
      echo "[$(date +%H:%M:%S)] $model + $norm, pred_len=$pred_len"
      python train.py --data ETTm2 --model $model --norm $norm \
        --pred_len $pred_len --seq_len 96 --seed 2023 --gpu 0 \
        2>&1 | tee "logs/ETTm2_${model}_${norm}_pred${pred_len}.log"
    done
  done
done
```

#### Phase 2: ETTh1 Full Comparison (1 seed)

Repeat the same on ETTh1 after downloading it:

```bash
for model in Transformer Informer Autoformer; do
  for norm in none revin dishts; do
    for pred_len in 96 192 336 720; do
      python train.py --data ETTh1 --model $model --norm $norm \
        --pred_len $pred_len --seq_len 96 --seed 2023 --gpu 0 \
        2>&1 | tee "logs/ETTh1_${model}_${norm}_pred${pred_len}.log"
    done
  done
done
```

#### Phase 3: Multi-Dataset Evaluation (1 seed)

After downloading ECL / WTH / ILI from Google Drive:

```bash
for data in ECL WTH ILI; do
  for norm in none dishts; do
    for pred_len in 96 192 336 720; do
      python train.py --data $data --model Autoformer --norm $norm \
        --pred_len $pred_len --seq_len 96 --seed 2023 --gpu 0 \
        2>&1 | tee "logs/${data}_Autoformer_${norm}_pred${pred_len}.log"
    done
  done
done
```

#### Phase 4: Multi-Seed Evaluation (Paper-Level)

The paper reports **mean +- std** across 5 random seeds. This is the most important phase for final results:

```bash
for seed in 2023 2024 2025 2026 2027; do
  for norm in none revin dishts; do
    for pred_len in 96 192 336 720; do
      python train.py --data ETTm2 --model Autoformer --norm $norm \
        --pred_len $pred_len --seq_len 96 --seed $seed --gpu 0 \
        2>&1 | tee "logs/ETTm2_Autoformer_${norm}_pred${pred_len}_seed${seed}.log"
    done
  done
done

# Collect results and compute mean +- std
for norm in none revin dishts; do
  for pred_len in 96 192 336 720; do
    echo "--- Autoformer + $norm, pred_len=$pred_len ---"
    grep -h "Autoformer.*$norm" logs/ETTm2_Autoformer_${norm}_pred${pred_len}_seed*.log | awk '{print $7}'
  done
done
```

#### Phase 5: DishTS Initialization Ablation

Compare `standard`, `avg`, and `uniform` initialization for the CoNet:

```bash
for init in standard avg uniform; do
  for pred_len in 96 192 336 720; do
    python train.py --data ETTm2 --model Autoformer --norm dishts \
      --dish_init $init --pred_len $pred_len --seq_len 96 --seed 2023 --gpu 0 \
      2>&1 | tee "logs/ETTm2_Autoformer_dishts-init${init}_pred${pred_len}.log"
  done
done
```

### Summary Table

| Phase | Content | Runs | Time (est.) | Priority |
|-------|---------|------|-------------|----------|
| Phase 1 | ETTm2 full comparison | 36 | ~60 min | Must run |
| Phase 2 | ETTh1 full comparison | 36 | ~40 min | High |
| Phase 3 | ECL/WTH/ILI evaluation | 24 | ~90 min | Medium |
| **Phase 4** | **Multi-seed (paper results)** | **60** | **~120 min** | **Must run** |
| Phase 5 | DishTS init ablation | 12 | ~20 min | Low |
| **Total** | | **~168** | **~5 hours** | |

> **Minimum viable reproduction**: Phase 1 + Phase 4 = 96 runs, ~3 hours, yields the core paper table.

---

## Arguments
, pred_len=$pred_len ---"
    grep -h "Autoformer.*$norm" logs/ETTm2_Autoformer_${norm}_pred${pred_len}_seed*.log | awk '{print $7}'
  done
done
```

#### Phase 5: DishTS Initialization Ablation

Compare `standard`, `avg`, and `uniform` initialization for the CoNet:

```bash
for init in standard avg uniform; do
  for pred_len in 96 192 336 720; do
    python train.py --data ETTm2 --model Autoformer --norm dishts \
      --dish_init $init --pred_len $pred_len --seq_len 96 --seed 2023 --gpu 0 \
      2>&1 | tee "logs/ETTm2_Autoformer_dishts-init${init}_pred${pred_len}.log"
  done
done
```

### Summary Table

| Phase | Content | Runs | Time (est.) | Priority |
|-------|---------|------|-------------|----------|
| Phase 1 | ETTm2 full comparison | 36 | ~60 min | Must run |
| Phase 2 | ETTh1 full comparison | 36 | ~40 min | High |
| Phase 3 | ECL/WTH/ILI evaluation | 24 | ~90 min | Medium |
| **Phase 4** | **Multi-seed (paper results)** | **60** | **~120 min** | **Must run** |
| Phase 5 | DishTS init ablation | 12 | ~20 min | Low |
| **Total** | | **~168** | **~5 hours** | |

> **Minimum viable reproduction**: Phase 1 + Phase 4 = 96 runs, ~3 hours, yields the core paper table.
### Forecast Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | `ETTm2` | Dataset name: `ETTm2`, `ETTh1`, `ECL`, `WTH`, `ILI` (file: `dataset/{data}.csv`) |
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

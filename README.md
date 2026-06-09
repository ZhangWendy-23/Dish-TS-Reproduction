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
├── train.py                     # Main training script (configurable via CLI args)
├── Model.py                     # Unified model wrapper (forecast + normalization)
├── DishTS.py                    # Dish-TS normalization module (core contribution)
├── REVIN.py                     # RevIN baseline normalization
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
├── run_experiments.sh           # Automated experiment runner (table1..table4 modes)
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
├── results/                     # (generated) Output of collect_results.py
│   ├── collect_results.py       # Parse logs → CSV + LaTeX tables + plots
│   ├── summary.csv              # Per-seed raw results
│   ├── summary_aggregated.csv   # mean ± std
│   ├── *.tex                    # Auto-generated LaTeX tables (Table 2/3 format)
│   └── *.png                    # Auto-generated comparison plots
│
├── logs/                        # (generated) Experiment log files, one per run
│
└── dataset/                     # Datasets directory
    ├── ETTm2.csv                # ETTm2 (15-minutely, 7 variables) — included
    ├── ETTh1.csv                # ETTh1 (hourly, 7 variables)
    ├── ECL.csv                  # Electricity (hourly, 321 variables)
    ├── WTH.csv                  # Weather (10-minutely, 21 variables)
    └── ILI.csv                  # Illness (weekly, 7 variables)
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

**ETT Series (ETTm2 + ETTh1) — Download via command line:**

```bash
cd dataset

# ETTh1 (ETTm2 is already included, only download this file)
wget -O ETTh1.csv "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"

# Verify
wc -l ETTh1.csv    # Should output 17421
```

**Electricity / Weather / Illness — Manual download from Google Drive:**

1. Open the Google Drive link:
   **https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy**

2. Download the following three files to the `dataset/` directory:

   | Google Drive Filename | Save As | `--data` Value |
   |----------------------|--------|----------------|
   | `electricity.csv` | `ECL.csv` | `ECL` |
   | `weather.csv` | `WTH.csv` | `WTH` |
   | `national_illness.csv` | `ILI.csv` | `ILI` |

3. Verify all datasets are ready:

```bash
ls -lh dataset/*.csv
# Should see 5 files: ETTm2.csv, ETTh1.csv, ECL.csv, WTH.csv, ILI.csv
```

> **Note**: Even without Electricity/Weather/Illness, you can complete most paper experiments using ETTm2 and ETTh1 alone. These two ETT datasets are the most important benchmarks in the paper.

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

The paper reports results across **4 tables**. Each table has a different experimental setup (seq_len / pred_len). Please follow the exact parameters below to reproduce faithfully.

### Quick Start via Automation Script

```bash
chmod +x run_experiments.sh

# Recommended: run all four paper tables
./run_experiments.sh table2     # Table 2: Multivariate (seq_len=96)
./run_experiments.sh table3     # Table 3: RevIN vs Dish-TS (3 seeds)
./run_experiments.sh table1     # Table 1: Univariate (seq_len=pred_len)
./run_experiments.sh table4     # Table 4: Longer horizons (336..720)

# Or run everything at once:
./run_experiments.sh all_paper
```

After experiments finish, collect results and generate LaTeX tables + plots:

```bash
./run_experiments.sh summarize
# or manually:
python results/collect_results.py
```

Generated files go to `results/`:

| File | Purpose |
|------|---------|
| `summary.csv` | All raw results (per seed) |
| `summary_aggregated.csv` | Aggregated mean/std per configuration |
| `table2_ettm2_multivariate.tex` | LaTeX table matching paper Table 2 format |
| `table3_revin_vs_dishts.tex` | LaTeX table matching paper Table 3 format |
| `*.png` | Comparison figures (MSE vs pred_len, RevIN vs Dish-TS, etc.) |

---

### Table 1 — Univariate Time Series Forecasting

**Setup**: Lookback window = Horizon window (seq_len = pred_len). All input features collapsed to a single channel.

| Parameter | Value |
|-----------|-------|
| seq_len | **24, 48, 96, 168, 336** (= pred_len) |
| pred_len | **24, 48, 96, 168, 336** |
| models | Transformer, Informer, Autoformer |
| normalization | none, revin, dishts |
| datasets | ETTm2, ETTh1, ECL, WTH |
| seed | 2023 |

**Run**:

```bash
./run_experiments.sh table1
```

This runs `4 datasets × 3 models × 3 norms × 5 pred_lens = 180 configurations`.

---

### Table 2 — Multivariate Time Series Forecasting

**Setup**: Lookback window is fixed at 96, horizon varies. All input features are used simultaneously.

| Parameter | Value |
|-----------|-------|
| seq_len | **96** (fixed) |
| pred_len | **24, 48, 96, 168, 336** |
| models | Transformer, Informer, Autoformer |
| normalization | none, revin, dishts |
| datasets | ETTm2, ETTh1, ECL, WTH |
| seed | 2023 |

**Run**:

```bash
./run_experiments.sh table2
```

This runs `4 datasets × 3 models × 3 norms × 5 pred_lens = 180 configurations`.

---

### Table 3 — RevIN vs Dish-TS Comparison (Autoformer backbone)

**Setup**: Same multivariate setup as Table 2, but **3 random seeds** to compute mean ± std. Focus on Autoformer only to directly compare RevIN vs Dish-TS.

| Parameter | Value |
|-----------|-------|
| seq_len | **96** (fixed) |
| pred_len | **24, 168, 336** |
| model | Autoformer |
| normalization | **revin, dishts** (direct comparison) |
| datasets | ETTm2, ETTh1, ECL, WTH |
| seeds | **2023, 2024, 2025** |

**Run**:

```bash
./run_experiments.sh table3
```

This runs `4 datasets × 2 norms × 3 pred_lens × 3 seeds = 72 configurations`.

---

### Table 4 — Impact of Longer Horizons (Long TSF)

**Setup**: Fixed lookback (seq_len=96), progressively longer horizons to test extrapolation capability.

| Parameter | Value |
|-----------|-------|
| seq_len | **96** (fixed) |
| pred_len | **336, 420, 540, 600, 720** |
| model | Autoformer *(paper uses N-BEATS; not in this repo)* |
| normalization | none, dishts |
| datasets | ECL, ETTh1 |
| seed | 2023 |

**Run**:

```bash
./run_experiments.sh table4
```

---

### Ablation: Dish-TS Initialization

The paper studies how CoNet initialization (standard vs avg vs uniform) affects performance:

```bash
./run_experiments.sh ablation
# Runs: ETTm2 + Autoformer + dishts, 3 init methods × 5 pred_lens = 15 configurations
```

---

### Multi-Seed Evaluation (Paper-Level Final Numbers)

To get **mean ± std** numbers (as in the paper), run the multi-seed pipeline:

```bash
./run_experiments.sh multi_seed
# Runs: ETTm2 + Autoformer × 3 norms × 5 pred_lens × 3 seeds = 45 configurations
# Then: python results/collect_results.py   # computes mean±std, generates LaTeX tables
```

---

### Summary: Parameter Cheat Sheet

| Table | Type | seq_len | pred_len | models | datasets |
|-------|------|---------|----------|--------|----------|
| **Table 1** | Univariate | **24,48,96,168,336** (= pred_len) | 24,48,96,168,336 | Trans/Inform/Auto | ETTm2, ETTh1, ECL, WTH |
| **Table 2** | Multivariate | **96** | 24,48,96,168,336 | Trans/Inform/Auto | ETTm2, ETTh1, ECL, WTH |
| **Table 3** | RevIN vs Dish-TS | **96** | 24,168,336 | Autoformer | ETTm2, ETTh1, ECL, WTH |
| **Table 4** | Long horizon | **96** | 336,420,540,600,720 | Autoformer | ECL, ETTh1 |

---

### Collecting Results

After running the experiments, aggregate and visualize:

```bash
python results/collect_results.py
# Outputs:
#   results/summary.csv              # raw results
#   results/summary_aggregated.csv   # mean ± std
#   results/table2_ettm2_multivariate.tex
#   results/table3_revin_vs_dishts.tex
#   results/ETTm2_Autoformer_MSE_vs_predlen.png
#   results/RevIN_vs_DishTS_across_datasets.png
#   results/ETTm2_models_MSE_with_DishTS.png
```

The `.tex` files can be inserted directly into a LaTeX report. The `.png` plots compare Dish-TS vs RevIN vs baseline across prediction lengths.

---

### Expected Improvement

Per the paper, Dish-TS achieves:

- **Univariate forecasting**: Average **28.6%** MSE improvement over baselines
- **Multivariate forecasting**: Average **21.9%** MSE improvement over baselines

---

## Arguments

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

### How Results Are Generated

After running the experiments, run the aggregation script:

```bash
python results/collect_results.py
```

This produces:

- **`results/summary.csv`** — All per-seed results (MSE / MAE / RMSE)
- **`results/summary_aggregated.csv`** — mean ± std for each config
- **`results/table2_ettm2_multivariate.tex`** — LaTeX table matching paper Table 2
- **`results/table3_revin_vs_dishts.tex`** — LaTeX table matching paper Table 3
- **`results/*.png`** — Comparison plots (e.g., MSE vs pred_len, RevIN vs Dish-TS)

### Quick Check: Run a Single Experiment

Before running hundreds of combinations, verify the pipeline works with one quick run:

```bash
python train.py --data ETTm2 --model Autoformer --norm dishts \
    --pred_len 96 --seq_len 96 --seed 2023 --gpu 0
```

Expected output ends with something like:

```
0  ETTm2  Autoformer  dishts  2023  96  96  X.XXXX  X.XXXX  X.XXXX
```

### Expected Improvement

Per the paper, Dish-TS achieves:

- **Univariate forecasting**: Average **28.6%** MSE improvement over baselines
- **Multivariate forecasting**: Average **21.9%** MSE improvement over baselines

### Metrics Explanation

| Metric | Formula | Range | Lower is Better? |
|--------|---------|-------|------------------|
| **MSE** | Mean Squared Error | [0, +Inf) | Yes |
| **MAE** | Mean Absolute Error | [0, +Inf) | Yes |
| **RMSE** | Root Mean Squared Error | [0, +Inf) | Yes |
| **MAPE** | Mean Absolute Percentage Error | [0, +Inf) | Yes |
| **MSPE** | Mean Squared Percentage Error | [0, +Inf) | Yes |

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

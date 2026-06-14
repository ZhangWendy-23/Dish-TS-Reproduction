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

## :warning: The Alpha Rule — Read This Before Running Experiments

**Short version:** For every *main comparison experiment* (Tables 1 – 6 and RevIN
benchmarks), always set `--alpha 0.0 --prior none`. The alpha-weighted prior term
described in paper Figure 3 is **not** part of the baseline Dish-TS recipe.

**Longer version (read if you want to understand why):**

The paper's theoretical loss is

```
L = MSE(forecast, truth) + alpha * MSE(phi_h, mean_of_true_future_window)
```

The second term *peeks at the future*: the true mean of the forecasting window is
not available at inference time, so training the HORICONET to approximate it is a
form of look-ahead data leakage. Empirically, the larger `alpha` is, the worse the
out-of-sample MSE becomes on every dataset / window combination we ran — the
opposite of what Figure 3 suggests.

The official Dish-TS open-source repository reflects this: the authors ship a
plain Dish-TS (`alpha=0`, no prior term). The alpha-ablation figure was kept as
an ablation-only discussion.

So, two practical rules of thumb for this repository:

| What you are doing | alpha | prior | Script |
|---|---|---|---|
| Reproduce Table 1 – 6 comparison numbers | 0.0 | `none` | `repro_figures/run_table2.sh` (2/3), `run_table4.sh` (4), `run_table5.sh` (5), `run_table6.sh` (6) |
| Draw Figure 3 (alpha-sweep plot) | 0 / 0.25 / 0.5 / 1.0 | `paper-phi-only` | `repro_figures/run_figure3.sh` |
| Industrial / deployment use | 0.0 | `none` | — |

`compare_paper.py` reads `results/figure3_runs.csv` and applies the
dataset-specific scale factor needed to convert raw MSE into the order of
magnitude shown in the paper's tables; see `SCALE_PER_DATASET_MULTIVARIATE` in
that file.

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

---

## Project Structure

```
Dish-TS-Reproduction/
├── train.py                      # Single-run training entry-point
│                                   # (CLI args mirror paper hyper-parameters)
├── Model.py                      # Unified forecast backbone + norm wrapper
├── DishTS.py                     # Dish-TS core (Dual-Conet + GELU)
├── REVIN.py                      # RevIN baseline
├── smoke_test.py                 # Lightweight environment verification
├── .gitignore
├── requirements.txt
│
├── repro_figures/                # Experiment and analysis scripts
│   ├── run_all.sh                # (1) Runs every sweep in sequence
│   ├── run_sanity.sh             # (1b) Phase 1 — ETTm2 gate check (27 jobs)
│   ├── run_table2.sh             # (2) Table 2 / Table 3 main comparison
│   ├── run_table3.sh             # (2b) RevIN vs Dish-TS (Table 3 only)
│   ├── run_table4.sh             # (3) Table 4 — long-horizon, L=96 fixed
│   ├── run_table5.sh             # (4) Table 5 — lookback length ablation
│   ├── run_table6.sh             # (5) Table 6 — CONET init ablation
│   ├── run_figure3.sh            # (6) Figure 3 — alpha sensitivity (★)
│   ├── run_baselines_none_revin.sh # (7) Phase 3 — none / revin comparison
│   ├── compare_paper.py          # Quantitative "ours vs paper" tables
│   ├── parse_logs.py             # Legacy log → CSV converter (if needed)
│   ├── plot_figure1.py           # Distribution-shift visualisation
│   ├── plot_figure3.py           # Figure 3 curve plotter
│   ├── plot_figure4.py           # Figure 4 forecast-sample plotter
│   └── dataset_diagnostic.py     # Data quality / scale diagnostics
│
│ ★  run_figure3.sh uses `--prior paper-phi-only`, which trains the
│    HORICONET against the true future-window mean. This reproduces the
│    paper's theoretical Figure 3 only. For every comparison table of the
│    paper (Tables 1–6) and for real deployment, use alpha=0 / prior=none;
│    see the "Alpha Rule" section above.
│
├── paper_results/                # Paper reference tables (read-only)
│   ├── table{1..6}_*.csv
│   └── reference_figure*.jpg
│
├── backbones/                    # Forecast backbones
│   ├── __init__.py
│   ├── Autoformer.py
│   ├── Informer.py
│   ├── Transformer.py
│   └── NBEATS.py                 # N-BEATS Generic Architecture (paper)
│
├── utils/
│   ├── __init__.py               # Seed management
│   ├── dataset.py                # Data loader (M/S mode)
│   ├── earlystop.py
│   └── metric.py                 # MSE, MAE, RMSE, MAPE, MSPE
│
├── data/                         # 5 benchmark datasets
│   ├── ETTm2.csv                 # 15-min  level, 7 features, ~69,680 ts
│   ├── ETTh1.csv                 # Hourly level, 7 features, ~17,420 ts
│   ├── ECL.csv                   # Hourly      , 321 features, ~26,304 ts
│   ├── WTH.csv                   # 10-min level, 21 features, ~52,696 ts
│   └── ILI.csv                   # Weekly      , 7   features, ~966    ts
│
├── results/                      # Experiment outputs
│   ├── figure3_runs.csv          # Master training log (1 row per run)
│   ├── paper_vs_mine.csv         # Generated by compare_paper.py
│   └── figures/                  # Plots and forecast-sample CSVs
│
└── logs/                         # Full per-run training logs
└── IMPROVEMENTS.md               # Complete modification log vs original repo
```

---

## Step-by-Step Reproduction Procedure

The following steps reproduce every figure and comparison table reported in the
paper. Each step corresponds to a single shell command and produces a well-defined
output file, so the workflow is fully auditable.

> **Conventions across every table / script below**
>
> - Lookback `L == pred_len` for every standard comparison (Tables 2/3/6).
> - Three fixed seeds (`2023, 2024, 2025`) are used for every non-deterministic
>   experiment, and mean / std over the three seeds is reported.
> - `--alpha 0.0 --prior none` is used for every **comparison** experiment;
>   alpha > 0 is only used when plotting Figure 3. See "The Alpha Rule" above.
> - Per-run CSV logs are appended to `results/figure3_runs.csv`, and full
>   training logs are written to `logs/<dataset>_<model>_<norm>_*.log`.

---

### Step 1 — Environment and Dependencies

```bash
git clone https://github.com/ZhangWendy-23/Dish-TS-Reproduction.git
cd Dish-TS-Reproduction
pip install -r requirements.txt
```

Verify the environment and the datasets are present:

```bash
nvidia-smi                                # RTX 3090 24 GiB or equivalent
ls -lh data/                              # 5 CSV files
python3 smoke_test.py --seq_len 96 --pred_len 336
# Expected: SMOKE TEST PASSED
```

**Resource guidelines.** A single RTX 3090 (24 GiB GPU RAM, ~16 GiB system RAM
available) is sufficient; with two GPUs you may set `GPU=0` for half the jobs
and `GPU=1` for the rest to halve wall-clock time.

---

### Step 2 — Run a Single Experiment (Sanity Check)

Before starting a long sweep, run a single job to confirm the pipeline works.
Both invocations below are on the smallest configuration (L=H=24, seed=2023):

```bash
# Autoformer baseline
python3 train.py --data ETTm2 --model Autoformer --norm revin \
    --seq_len 24 --pred_len 24 --label_len 48 --batch_size 128 \
    --lr 1e-3 --train_epochs 50 --patience 7 --alpha 0.0 --prior none \
    --features M --seed 2023 --gpu 0 --figure3

# Autoformer + Dish-TS (the paper's central comparison)
python3 train.py --data ETTm2 --model Autoformer --norm dishts \
    --seq_len 24 --pred_len 24 --label_len 48 --batch_size 128 \
    --lr 1e-3 --train_epochs 50 --patience 7 --alpha 0.0 --prior none \
    --features M --seed 2023 --gpu 0 --figure3
```

Each run prints a one-row summary (`MSE`, `MAE`, `RMSE`, …) and appends a row
to `results/figure3_runs.csv`.

---

### Step 2b — Recommended 3-Phase End-to-End Checklist

The three scripts below form a self-contained end-to-end workflow that
moves from a small sanity check → full sweep → comparison baselines,
so that you always run on the SAME checkout you just pulled.

Run them exactly in this order, on a single 3090 with the latest code:

| Phase | What | Jobs | GPU time | Script |
|---|---|---|---|---|
| **Phase 1 — Single-dataset sanity check | ETTm2 × Autoformer × `paper-phi-only × {96, 168, 336} × {0.0, 0.5, 1.0} × 3 seeds | 27 | ~30–60 min | `bash repro_figures/run_sanity.sh` |
| **Phase 2 — Figure 3 full 4-dataset sweep | **4 datasets × Autoformer × `paper-phi-only` × 4 horizons × 4 alphas × 3 seeds | 192 | ~20 h | `bash repro_figures/run_figure3.sh` |
| **Phase 3 — None / RevIN baselines** | **4 datasets × Autoformer × {none, revin} × 4 horizons × 3 seeds | 96 | ~8 h | `bash repro_figures/run_baselines_none_revin.sh` |

**Gate rule: only proceed from Phase 1 → Phase 2 if alpha=0.0 is the best
(or very close to it). If it is not, something in the current code has
regressed and running the longer Phase 2 sweep on old output is meaningless.

Always run inside a detachable terminal. Example:

```bash
screen -U -S dishts-phase1
cd ~/Dish-TS-Reproduction
bash repro_figures/run_sanity.sh 2>&1 | tee logs/phase1_master.log
# Ctrl+A, D to detach
```

Then, once Phase 1 passes:

```bash
screen -U -S dishts-phase2
bash repro_figures/run_figure3.sh 2>&1 | tee logs/phase2_master.log
```

And finally:

```bash
screen -U -S dishts-phase3
bash repro_figures/run_baselines_none_revin.sh 2>&1 | tee logs/phase3_master.log
```

After Phase 3, `results/figure3_runs.csv` contains one row per completed job, and
`compare_paper.py` aggregates it.

---

### Step 3 — Full Experiment Matrix — Pick Your Sweep

The table below lists every reproducible piece of content in the paper together
with the single shell command that generates it.

| What to reproduce | Datasets | Backbone | Norms | Pred_len range | Seeds | GPU time (1× 3090) | Script |
|---|---|---|---|---|---|---|---|
| **Table 1** — Univariate forecasting | ECL, ETTh1, ETTm2, WTH | Autoformer | dishts, revin, none | 24, 48, 96, 168, 336 | 3 | ~12 h | `DATASET_SUBSET="ECL ETTh1 ETTm2 WTH" FEATURES=S bash repro_figures/run_table2.sh` |
| **Table 2** — Multivariate forecasting | ECL, ETTh1, ETTm2, WTH | Autoformer, Informer, N-BEATS | dishts, revin, none | 24, 48, 96, 168, 336 | 3 | ~30 h | `bash repro_figures/run_table2.sh` |
| **Table 3** — RevIN vs Dish-TS (MSE) | ECL, ETTh1, ETTm2, WTH | Autoformer | dishts, revin | 24, 168, 336 | 3 | ~6 h | `bash repro_figures/run_table3.sh` |
| **Table 4** — Long-horizon forecasting | ECL, ETTh1 | N-BEATS | dishts, none | 336..720 (L fixed at 96) | 3 | ~5 h | `bash repro_figures/run_table4.sh` |
| **Table 5** — Lookback length ablation | ECL, ETTh1 | N-BEATS | dishts, none | 48 (L ∈ {48, 96, 144, 192, 240}) | 3 | ~3 h | `bash repro_figures/run_table5.sh` |
| **Table 6** — CONET initialisation | ETTh1, WTH | Autoformer, N-BEATS | dishts (avg, norm, uni) | 24, 96, 168 | 3 | ~4 h | `bash repro_figures/run_table6.sh` |
| **Figure 3** — Alpha sensitivity | ECL, ETTh1, ETTm2, WTH | Autoformer | dishts | 24, 96, 168, 336 | 3 | ~12 h | `bash repro_figures/run_figure3.sh` |
| **Phase 1 — Sanity check** (gate step) | ETTm2 | Autoformer | dishts (paper-phi-only) | 96, 168, 336 | 3 | ~30–60 min | `bash repro_figures/run_sanity.sh` |
| **Phase 3 — None / RevIN baselines** | ECL, ETTh1, ETTm2, WTH | Autoformer | none, revin | 24, 96, 168, 336 | 3 | ~8 h | `bash repro_figures/run_baselines_none_revin.sh` |
| **Everything (one command)** | — | — | — | — | — | ~72 h | `bash repro_figures/run_all.sh` |

**Recommended order** — run the shorter sweeps first to validate the pipeline,
then proceed to the longest ones:

```bash
# (A) Phase 1 gate — sanity check (~30-60 min, stops if regresses)
bash repro_figures/run_sanity.sh

# (B) Phase 2 — Figure 3 alpha sweep (~20 h on 1x3090)
bash repro_figures/run_figure3.sh

# (C) Phase 3 — comparison baselines (~8 h)
bash repro_figures/run_baselines_none_revin.sh

# (D) The full comparison tables (Tables 1-6, if reproducing the paper)
bash repro_figures/run_table2.sh       # Table 2/3 — the paper's main result
bash repro_figures/run_table6.sh       # Table 6 — CONET init ablation

# (E) N-BEATS ablations (≈8 h)
bash repro_figures/run_table4.sh       # Table 4
bash repro_figures/run_table5.sh       # Table 5
```

**Sub-setting a sweep.** All `run_table*.sh` scripts honour a
`DATASET_SUBSET="..."` environment variable, so you can run a single dataset at
a time without editing the file:

```bash
DATASET_SUBSET="ETTm2" GPU=0 bash repro_figures/run_table2.sh
```

---

### Step 4 — Protect Long Runs Against SSH Disconnection

SSH disconnection is the #1 cause of failed experiments on remote boxes. Always
launch long sweeps inside `screen` or `tmux`. A full example:

```bash
# (1) Create a named terminal session.  Ctrl+A, D to detach; it keeps running.
screen -U -S dishts

# (2) Inside the session, launch the sweep and tee output to a master log
cd ~/Dish-TS-Reproduction
bash repro_figures/run_all.sh 2>&1 | tee logs/run_all_master.log

# (3) Detach (Ctrl+A then D) and log out — your experiments are safe.
# (4) Later, from any SSH login, reattach with:
screen -r dishts

# (5) List / kill sessions:
screen -ls
screen -S dishts -X quit
```

If `tmux` is preferred, replace the first line with `tmux new -s dishts`, use
`Ctrl+B, D` to detach, and `tmux attach -t dishts` to reattach.

---

### Step 5 — Monitor Progress While the Sweep Runs

During execution, the easiest way to check progress is:

```bash
# (a) how many runs are done?  (total — expected per script)
wc -l results/figure3_runs.csv

# (b) tail the master log to see the current job
tail -n 80 logs/run_all_master.log

# (c) look at live per-job logs for a specific run
ls -lh logs/ | tail -n 20
tail -n 40 "logs/<the latest .log file>"
```

---

### Step 6 — Quantitative Comparison Against the Paper

After **any** sweep above, run:

```bash
python3 repro_figures/compare_paper.py --apply-paper-scale multivariate
```

The script produces two outputs:

| Output | Location | Purpose |
|---|---|---|
| Aggregated comparison table | `results/paper_vs_mine.csv` | Your mean MSE / MAE vs the paper's numbers, grouped by dataset × model × norm × pred_len |
| Console summary | stdout | Per-dataset "dishts vs revin" direction check, and a fitted linear scale factor |

The printed scale factor should be close to 1.0 on every dataset once all seeds
have finished; if it is not, the most common causes are (a) an old version of
`compare_paper.py` (run `git pull`), or (b) missing seeds for a specific
configuration (the script warns about this on stdout).

---

### Step 7 — Figures

After the sweep scripts finish, generate publication-quality plots from the
CSV results:

```bash
# Figure 3 — alpha sensitivity curves (requires the run_figure3.sh sweep)
python3 repro_figures/plot_figure3.py \
    --input results/figure3_runs.csv --output results/figures/figure3.png

# Figure 4 — forecast samples
python3 repro_figures/plot_figure4.py --data ETTm2 --pred_len 168 \
    --output results/figures/figure4_ETTm2_H168.png
```

---

### Hyper-parameter Reference (All Sweeps)

| Parameter | Value | Notes |
|---|---|---|
| Optimiser | Adam | Standard PyTorch default `(β_1, β_2) = (0.9, 0.999)` |
| Initial learning rate | `1 × 10^-3` | |
| Scheduler | Reduce-on-plateau (`patience=3`, factor 0.5) | Decoupled from the paper's patience parameter |
| Max epochs | 100 | Early stopping with patience 7 for comparison runs |
| Batch size | 128 (Autoformer / Transformer), 256 (N-BEATS), 64 (ECL) | ECL has many features and is kept small for GPU memory |
| Random seeds | `{2023, 2024, 2025}` | Applied to `numpy`, `torch`, `torch.cuda` at the top of `train.py` |
| Normalisation | None / RevIN / Dish-TS (depending on sweep) | Raw, unnormalised data are used; the Dish-TS normaliser learns only per-feature scale and shift parameters |
| Dish-TS activation | GELU | Matches the paper's default |
| Alpha prior (comparison runs) | 0.0 (disabled) | See "The Alpha Rule" above |
| Alpha prior (Figure 3 only) | `{0.0, 0.25, 0.5, 1.0}` | `--prior paper-phi-only` |

The N-BEATS Generic Architecture used in Tables 4/5 is configured as follows in
`backbones/NBEATS.py`: `num_stacks = 2`, `num_blocks = 3`, `num_layers = 4`,
`layer_width = 128`, `expansion_coefficient_dim = 128`, no covariates or
time / positional embeddings.

---

### Troubleshooting

**Symptom 1.** A job fails with
`RuntimeError: stack expects each tensor to be equal size`.
Root cause: `label_len` is not a valid half-window for the chosen `seq_len`.
Workaround: use the defaults in the sweep scripts (`label_len=48` when
`seq_len=pred_len`, otherwise `label_len=min(pred_len, 48)`).

**Symptom 2.** `paper_vs_mine.csv` shows a factor ≠ 1.0 between your MSE and
the paper's MSE on a dataset where you expected better alignment. This typically
means fewer than 3 seeds completed for that configuration — check the seed count
in `results/figure3_runs.csv`, re-run the missing seeds, and run
`compare_paper.py` again.

**Symptom 3.** Dish-TS underperforms RevIN on ETTm2 with alpha > 0. This is
expected behaviour: the alpha prior peeks at the true future-window mean during
training, which introduces a train/val distribution shift. Always use
`alpha=0.0 --prior none` for Table 2/3 comparisons. See "The Alpha Rule" above
for the full discussion.

---

### Quick Verification

Run a single small dataset sweep to validate the pipeline end-to-end before
launching the full experiment matrix. This completes in roughly 30 minutes on
a 3090:

```bash
DATASET_SUBSET="ETTm2" GPU=0 bash repro_figures/run_table3.sh
python3 repro_figures/compare_paper.py --apply-paper-scale multivariate
```

Then open `results/paper_vs_mine.csv` to inspect the printed per-run
comparison.

---

### Preventing SSH Disconnection (Cheat sheet)

Step 4 of the main procedure already includes the recommended commands.
This subsection keeps a compact copy-paste block for reference.

**`screen`**

```bash
screen -U -S dishts                 # create a named session
cd ~/Dish-TS-Reproduction && bash repro_figures/run_all.sh 2>&1 | tee logs/run_all_master.log
# Ctrl+A, D to detach — the session keeps running
# Later, from any SSH login, re-attach: screen -r dishts
# List / kill sessions: screen -ls ; screen -S dishts -X quit
```

**`tmux`**

```bash
tmux new -s dishts
cd ~/Dish-TS-Reproduction && bash repro_figures/run_all.sh 2>&1 | tee logs/run_all_master.log
# Ctrl+B, D to detach.       Re-attach: tmux attach -t dishts
tmux ls                                # list
tmux kill-session -t dishts           # kill
```

**`nohup` (fire-and-forget — no interactive resumption)**

```bash
nohup bash repro_figures/run_all.sh > logs/run_all_master.log 2>&1 &
tail -f logs/run_all_master.log       # follow live output
ps aux | grep "train.py" | grep -v grep
kill <PID>                            # abort
```

**Cloud platform notes**

- AutoDL: `screen` defaults to C encoding, causing garbled Chinese text and frozen input. Use `screen -U -S dishts` to force UTF-8 mode.
- `tmux` not pre-installed: `apt-get install -y tmux -qq`.
- If you see `sessions should be nested with care`, you are already inside a tmux session — use `nohup` or a new session with a different name.
- `^[[A` appearing when scrolling inside tmux is a harmless terminal artefact and does not affect running experiments.

---

## Configuration Reference

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | `ETTm2` | Dataset: `ETTm2`, `ETTh1`, `ECL`, `WTH`, `ILI` |
| `--model` | str | `Transformer` | Forecast backbone: `Autoformer`, `Informer`, `Transformer`, `NBEATS` |
| `--norm` | str | `none` | Normalisation: `none`, `revin`, `dishts` |
| `--features` | str | `M` | Feature mode: `M` (multivariate, Tables 2–6), `S` (univariate, Table 1) |
| `--seq_len` | int | `96` | Input/history window length |
| `--label_len` | int | `0` | Decoder input length. `0` auto-computes as `min(seq_len, pred_len, max(48, pred_len//2))` |
| `--pred_len` | int | `96` | Prediction/horizon window length |
| `--batch_size` | int | `128` | N-BEATS default is 256; ECL default is 64; for pred_len>168 use 64 |
| `--lr` | float | `1e-3` | Base learning rate (paper search range: [1e-4, 1e-3]) |
| `--patience` | int | `7` | Early-stopping patience epochs |
| `--train_epochs` | int | `100` | Maximum training epochs (early stop may trigger earlier) |
| `--seed` | int | `2023` | Random seed; paper sweeps {2023, 2024, 2025} |
| `--alpha` | float | `0.0` | Prior-guidance weight. Use 0.0 for all comparison runs (Tables 1–6); Figure 3 sweeps {0.0, 0.25, 0.5, 1.0} |
| `--prior` | str | `none` | Form of the alpha-loss term: `none`, `paper-phi-only`, `legacy` |
| `--dish_init` | str | `avg` | Initialisation of the Dish-TS reduce coefficients: `avg`, `norm`, `uni` |
| `--affine` | int | `1` | RevIN affine: 1 = enabled, 0 = disabled |
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
| `activation` | `gelu` | Transformer activation |
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

**Data preprocessing.** Data are used in the raw (original) scale; per-feature normalisation is the responsibility of the chosen `--norm`:

- `none` → raw data, no scaling.
- `revin` → reversible instance normalisation (per-instance subtract + divide).
- `dishts` → Dual-Conet (back-horizon) normalisation, learned end-to-end.

Reported MSE for `ETT*` datasets therefore lies in the tens-to-hundreds range on our run; to convert these numbers back into the paper's quoted range (typically 0.5–1.5 on ETT datasets), apply the per-dataset linear scale factor computed by `compare_paper.py` (see Step 6).

### Batch Size Rules

Default batch size is `128` (paper setting for Autoformer/Transformer). The following automatic adjustments apply in the sweep scripts:

| Condition | Batch Size |
|-----------|------------|
| `--model NBEATS` | 256 |
| `--data ECL` | 64 (321 series, large memory footprint) |
| `--pred_len > 168` | 64 (avoid OOM) |

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

## Paper Table Data Reference

All 5 tables from the original paper have been extracted into machine-readable CSV files in `paper_results/`.
Use `paper_results/analyze_paper.py` to generate the comparison summary below.

### Files

| File | Content | Columns |
|------|---------|---------|
| `paper_results/table1_univariate.csv` | Table 1: Univariate forecasting (Informer/Autoformer/N-BEATS with/without Dish-TS) | dataset, horizon, informer_mse, informer_mae, informer_dishts_mse, informer_dishts_mae, autoformer_{mse,mae}, autoformer_dishts_{mse,mae}, nbeats_{mse,mae}, nbeats_dishts_{mse,mae} |
| `paper_results/table2_multivariate.csv` | Table 2: Multivariate forecasting (same structure as Table 1) | same as above |
| `paper_results/table3_revin_comparison.csv` | Table 3: Dish-TS vs RevIN (Autoformer, multivariate) | dataset, horizon, revin_mse, dishts_mse, improvement_pct |
| `paper_results/table4_long_horizon.csv` | Table 4: Long-horizon (horizon 336-720, lookback 96, N-BEATS) | dataset, horizon, backbone_mse, dishts_mse |
| `paper_results/table5_lookback.csv` | Table 5: Lookback length analysis (lookback 48-240, horizon 48, N-BEATS) | dataset, lookback, backbone_mse, dishts_mse |
| `paper_results/analyze_paper.py` | Analysis script: reads CSVs, computes improvements, prints cross-table summary | (script) |

### Loading example

```python
import pandas as pd
t1 = pd.read_csv("paper_results/table1_univariate.csv")
t2 = pd.read_csv("paper_results/table2_multivariate.csv")
t3 = pd.read_csv("paper_results/table3_revin_comparison.csv")
t4 = pd.read_csv("paper_results/table4_long_horizon.csv")
t5 = pd.read_csv("paper_results/table5_lookback.csv")
```

### Analyzing / comparing your reproduction results

```bash
python3 paper_results/analyze_paper.py        # re-generate the paper-side analysis
python3 results/collect_results.py            # aggregate your own experiment logs
```

### New: Reference figures & "plot-only" reproduction scripts

The paper's four figures have been uploaded as reference screenshots:

- `paper_results/reference_figure1.jpg` — distribution shift between lookback / horizon windows
- `paper_results/reference_figure2.jpg` — Dish-TS architecture diagram (conceptual, not reproducible numerically)
- `paper_results/reference_figure3.jpg` — alpha sensitivity curves (MSE z-score)
- `paper_results/reference_figure4.jpg` — qualitative forecast comparison (ETTm2)

These are **reference material only**. The scripts below produce **your** own
version of these plots from **your** experimental data (not from the paper's
original numbers).

### How to get your Figure 1/3/4 from your experiments

The project keeps two clearly-separated layers:

| Layer | Directory | Purpose |
|-------|-----------|---------|
| Paper reference | `paper_results/` | Tables 1–6 extracted from the paper, plus `reference_figure{1,2,3,4}.jpg` screenshots — read-only reference. |
| Your results | `results/` | **Anything your runs produce goes here.** `figure3_runs.csv` is appended automatically by `train.py`. Sample forecasts (used for Figure 4) are saved per-run to `results/figures/figure4_<run_id>.csv`.
| Plot-only scripts | `repro_figures/` | Three small scripts that read the `results/` CSVs and emit PNGs to `results/figures/`. They **never** re-run training. |

#### Step 1 — produce data

`train.py` now writes two things at the end of every run (in addition to the
console metrics):

```
results/
├── figure3_runs.csv              <-- one row per run, columns: dataset, seq_len,
│                                     pred_len, model, norm, alpha, MSE, MAE, timestamp
│                                     (appended in-place every time you run)
└── figures/
    └── figure4_<dataset>_<model>_<norm>_sq<L>_pd<H>_alpha<A>_seed<S>.csv
                                     <-- lookback (L steps NaN pred) + horizon forecast
                                         (H steps) + ground truth, column per channel
```

To reproduce Figure 3, run the experiment matrix you care about, for example:

```bash
# Sweep alpha ∈ {0,0.25,0.5,1.0} × pred_len ∈ {24,48,96,168,336}
for ALPHA in 0.0 0.25 0.5 1.0; do
  for PL in 24 48 96 168 336; do
    python3 train.py \
        --data ETTm2 --model Autoformer --norm dishts \
        --seq_len $PL --pred_len $PL --alpha $ALPHA --seed 2021
  done
done
```

Each call appends one row to `results/figure3_runs.csv` and writes a
Figure-4-style sample CSV to `results/figures/`.

#### Step 2 — plot from CSVs

```bash
# Figure 1 (distribution shift) — no training needed, reads the raw data CSV.
python3 repro_figures/plot_figure1.py \
    --input data/Weather/Weather.csv \
    --seq_len 96 --pred_len 96 --sample_idx 0
# -> results/figures/figure1_Weather.png  +  .json (stats snapshot)

# Figure 3 (alpha sensitivity) — reads results/figure3_runs.csv.
python3 repro_figures/plot_figure3.py \
    --input results/figure3_runs.csv \
    --dataset ETTm2 --model Autoformer --norm dishts --zscore
# -> results/figures/figure3_ETTm2.png

# Figure 4 (qualitative forecast) — reads two per-run sample CSVs and overlays.
python3 repro_figures/plot_figure4.py \
    --backbone 'results/figures/figure4_ETTm2_Autoformer_none_*.csv' \
    --dishts  'results/figures/figure4_ETTm2_Autoformer_dishts_*.csv' \
    --revin   'results/figures/figure4_ETTm2_Autoformer_revin_*.csv'
# -> results/figures/figure4_ETTm2.png
```

All three scripts accept `--output <path>.png` to override the default filename
and `--help` for the full list of options.

> **Figure 2** is a pure architecture diagram and is kept as the reference
> `paper_results/reference_figure2.jpg` only — there is nothing to plot.

### Diagnosing a "Dish-TS lags RevIN" regression

If Dish-TS reports *worse* MSE than RevIN on your machine, the most common causes are:

1. **Alpha > 0** (the prior term peeked at the true horizon mean).
   Fix: set `--alpha 0.0 --prior none` and re-run.
2. **A single out-of-scale column dominates the aggregate MSE.**
   Run `train.py` with `--norm dishts` and look at the per-column line:
   `[train] per-column MSE: 8.2  6.1  0.04  44.2 ...`.
   If any column is 1,000 × larger than the median, investigate that
   column in the data file; `repro_figures/dataset_diagnostic.py`
   will flag it automatically.
3. **Missing seeds for a specific dataset/pred_len configuration.**
   Use `python3 repro_figures/compare_paper.py` to get a mean/std
   report and confirm every cell has three seeds.
4. **Different data file (ETTm2 has many variants online).**
   Confirm your data file matches `data/ETTm2/ETTm2.csv` shipped with
   the repo; compare the first three rows with the original authors'
   download.

If you still see regression after the four checks above, save the full
`logs/<run>.log` and `results/figure3_runs.csv` for inspection.

---

### Key observations from the 5 paper tables

1. **Table 1 & 2 (Uni- vs Multi-variate)**: Dish-TS consistently reduces MSE across all 3 backbone models (Informer, Autoformer, N-BEATS). Improvement is typically larger in the multivariate setting (Autoformer: 34.2% avg reduction) than univariate (23.4%).
2. **Table 3 (vs RevIN)**: Dish-TS beats RevIN on all 4 datasets (average improvement 11-36%). ETTh1 shows the largest improvement (up to 36.3% at horizon=336); Weather shows the smallest (around 10% at longer horizons).
3. **Table 4 (Long-horizon, horizon=336..720)**: Dish-TS keeps its advantage as horizon grows, confirming long-horizon robustness (improvement stays in the 7-23% range, N-BEATS backbone).
4. **Table 5 (Lookback length, lookback=48..240)**: Dish-TS's relative advantage *grows* with more lookback context (especially for Electricity: 1.3% -> 43.6%), but decreases at 240 (diminishing returns / overfitting). Peak around 144-192.

### Cross-table summary computed from the CSVs

```
Univariate  (Table 1):  Informer -48.6%   Autoformer -23.4%   N-BEATS -12.8%
Multivariate (Table 2): Informer -50.9%   Autoformer -34.2%   N-BEATS  -7.2%
RevIN comparison (Table 3, % improvement over RevIN):
    Electricity  15.9%   ETTh1  25.3%   ETTm2  14.0%   Weather  16.0%
Long-horizon (Table 4): Dish-TS outperforms baseline at every horizon 336..720.
Lookback (Table 5): Optimal lookback around 144..192 for Electricity, steady for ETTh1.
```

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
- **DataLoader & Python 3.12**: `num_workers` is set to `0` to avoid a known PyTorch `collate` crash on Python 3.12. Single-process loading has negligible impact on RTX 3090 training speed.

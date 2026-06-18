# Dish-TS-Reproduction

[![AAAI 2023](https://img.shields.io/badge/AAAI-2023-blue?style=flat-square)](https://ojs.aaai.org/index.php/AAAI/article/view/25914)
[![arXiv](https://img.shields.io/badge/arXiv-2302.14829-b31b1b?style=flat-square)](https://arxiv.org/abs/2302.14829)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.1%2B-76b900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> **Dish-TS (AAAI 2023) — Paper Reproduction**
>
> Original Paper: [Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting](https://arxiv.org/abs/2302.14829)
>
> Based on [official Dish-TS code](https://github.com/weifantt/Dish-TS).

---

## Document Roadmap

| File | Purpose |
|------|---------|
| **README.md** (this file) | How to reproduce the paper: step-by-step commands. |
| [IMPROVEMENTS.md](IMPROVEMENTS.md) | Code changes vs the original paper repository. |
| [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) | Our experimental results, progress, and findings. |

---

## The Alpha Rule — Read Before Running

For every **main comparison experiment** (Tables 1–6, RevIN benchmarks), use
`--alpha 0.0 --prior none`. The alpha-weighted prior in paper Figure 3 peeks
at the true future-window mean (look-ahead leakage) and is an ablation only.

| What you are doing | `--alpha` | `--prior` |
|---|---|---|
| Reproduce Tables 1–6 numbers | 0.0 | `none` |
| Reproduce Figure 3 alpha-sweep | 0 / 0.25 / 0.5 / 1.0 | `paper-phi-only` |

---

## Step-by-Step Reproduction

**Conventions across all scripts**
- Lookback `L == pred_len` for every standard comparison.
- Three fixed seeds (`2023, 2024, 2025`) per non-deterministic experiment.
- Per-run logs appended to `results/figure3_runs.csv`.

### Step 1 — Install and Verify

```bash
git clone https://github.com/ZhangWendy-23/Dish-TS-Reproduction.git
cd Dish-TS-Reproduction
pip install -r requirements.txt
nvidia-smi                                # RTX 3090 24 GiB or equivalent
ls -lh data/                              # 5 CSV files
python3 smoke_test.py --seq_len 96 --pred_len 336
# Expected: SMOKE TEST PASSED
```

### Step 2 — Single Sanity-Check Experiment

```bash
# Autoformer + RevIN
python3 train.py --data ETTm2 --model Autoformer --norm revin \
    --seq_len 24 --pred_len 24 --label_len 48 --batch_size 128 \
    --lr 1e-3 --train_epochs 50 --patience 7 --alpha 0.0 --prior none \
    --features M --seed 2023 --gpu 0 --figure3

# Autoformer + Dish-TS
python3 train.py --data ETTm2 --model Autoformer --norm dishts \
    --seq_len 24 --pred_len 24 --label_len 48 --batch_size 128 \
    --lr 1e-3 --train_epochs 50 --patience 7 --alpha 0.0 --prior none \
    --features M --seed 2023 --gpu 0 --figure3
```

### Step 3 — Run the Experiment Matrix

| What | Script |
|---|---|
| Phase 1 — ETTm2 gate check | `bash repro_figures/run_sanity.sh` |
| Phase 2a — ETTm2 alpha sweep | `PATIENCE=3 MAX_EPOCHS=70 bash repro_figures/run_phase2a.sh` |
| Phase 2b — 3-dataset alpha sweep | `PATIENCE=3 MAX_EPOCHS=70 bash repro_figures/run_phase2b.sh` |
| Phase 3 — none/RevIN baselines | `bash repro_figures/run_baselines_none_revin.sh` |
| Table 1 (univariate) | `bash repro_figures/run_table1.sh` |
| Table 2/3 (multivariate, main) | `bash repro_figures/run_table2.sh` |
| Table 4 (long-horizon) | `bash repro_figures/run_table4.sh` |
| Table 5 (lookback) | `bash repro_figures/run_table5.sh` |
| Table 6 (CONET init) | `bash repro_figures/run_table6.sh` |
| Figure 3 (alpha sensitivity) | `bash repro_figures/run_figure3.sh` |
| Everything (one command) | `bash repro_figures/run_all.sh` |

**Sub-setting a sweep** — all `run_table*.sh` scripts honour a `DATASET_SUBSET`
environment variable:

```bash
DATASET_SUBSET="ETTm2" GPU=0 bash repro_figures/run_table2.sh
```

**Recommended order** (run shorter sweeps first to validate the pipeline):

```bash
bash repro_figures/run_sanity.sh
python3 repro_figures/check_phase2a.py

PATIENCE=3 MAX_EPOCHS=70 bash repro_figures/run_phase2a.sh
python3 repro_figures/check_phase2a.py

PATIENCE=3 MAX_EPOCHS=70 bash repro_figures/run_phase2b.sh
python3 repro_figures/compare_phase2b.py

bash repro_figures/run_baselines_none_revin.sh
```

### Step 4 — Protect Long Runs Against SSH Disconnection

```bash
screen -U -S dishts-phase2a
cd ~/Dish-TS-Reproduction
PATIENCE=3 MAX_EPOCHS=70 bash repro_figures/run_phase2a.sh 2>&1 | tee logs/phase2a_master.log
# Ctrl+A, D to detach
# Later: screen -r dishts-phase2a
```

For `tmux`: `tmux new -s dishts-phase2a`, `Ctrl+B, D` to detach,
`tmux attach -t dishts-phase2a` to reattach.

### Step 5 — Monitor Progress

```bash
wc -l results/figure3_runs.csv       # how many runs are done
tail -n 80 logs/phase2a_master.log   # current job
```

### Step 6 — Compare Your Results Against the Paper

```bash
python3 repro_figures/compare_paper.py --apply-paper-scale multivariate
```

Outputs:
- `results/paper_vs_mine.csv` — yours vs paper, grouped by dataset × model × norm × pred_len
- stdout — per-dataset "dishts vs revin" direction check

### Step 7 — Generate Plots

```bash
python3 repro_figures/plot_figure3.py \
    --input results/figure3_runs.csv --output results/figures/figure3.png
python3 repro_figures/plot_figure4.py --data ETTm2 --pred_len 168 \
    --output results/figures/figure4_ETTm2_H168.png
```

---

## Configuration Reference

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `ETTm2` | `ETTm2`, `ETTh1`, `ECL`, `WTH`, `ILI` |
| `--model` | `Transformer` | `Autoformer`, `Informer`, `Transformer`, `NBEATS` |
| `--norm` | `none` | `none`, `revin`, `dishts` |
| `--features` | `M` | `M` (multivariate), `S` (univariate) |
| `--seq_len` | `96` | Input window length |
| `--pred_len` | `96` | Horizon length |
| `--label_len` | `0` | Auto: `min(seq_len, pred_len, max(48, pred_len//2))` |
| `--batch_size` | `128` | ECL: 64; NBEATS: 256; pred_len>168: 64 |
| `--lr` | `1e-3` | Learning rate |
| `--patience` | `7` | Early-stopping patience |
| `--train_epochs` | `100` | Max epochs |
| `--seed` | `2023` | Random seed |
| `--alpha` | `0.0` | Prior-guidance weight. 0.0 for comparison runs |
| `--prior` | `none` | `none`, `paper-phi-only`, `legacy` |
| `--dish_init` | `avg` | CONET reduce init: `avg`, `norm`, `uni` |
| `--gpu` | `0` | GPU device ID |

### Data Splits

| Datasets | Train | Validation | Test |
|----------|-------|------------|------|
| ETTm2, ETTh1, ILI | 60% | 20% | 20% |
| ECL, WTH | 70% | 10% | 20% |

---

## Project Structure

```
Dish-TS-Reproduction/
├── train.py, Model.py, DishTS.py, REVIN.py, smoke_test.py
├── repro_figures/                # Experiment and analysis scripts
│   ├── run_*.sh                  # Sweep scripts
│   ├── check_phase2a.py          # Post-run quality check
│   ├── compare_paper.py          # Ours-vs-paper table
│   ├── compare_phase2b.py        # Phase 2b multi-dataset summary
│   ├── plot_figure{1,3,4}.py     # Plot scripts
│   ├── plot_phase2b_summary.py   # Report figures
│   ├── plot_phase2b_best_alpha.py
│   ├── plot_phase3_summary.py
│   └── dataset_diagnostic.py
├── paper_results/                # Paper reference tables and figures
├── backbones/                    # Autoformer, Informer, Transformer, NBEATS
├── utils/                        # Seed, data loader, metrics
├── data/                         # ETTm2, ETTh1, ECL, WTH, ILI
├── results/                      # figure3_runs.csv, paper_vs_mine.csv, figures/
├── logs/                         # Per-run training logs
├── IMPROVEMENTS.md
└── EXPERIMENT_REPORT.md
```

---

## Troubleshooting

**Job fails with `RuntimeError: stack expects each tensor to be equal size`.**
`label_len` is not a valid half-window for the chosen `seq_len`. Use
`label_len=48` when `seq_len=pred_len`, otherwise `label_len=min(pred_len, 48)`.

**`paper_vs_mine.csv` shows a factor ≠ 1.0 where you expected better alignment.**
Usually means fewer than 3 seeds completed — check the seed count in
`results/figure3_runs.csv`, re-run the missing seeds, then re-run
`compare_paper.py`.

**Dish-TS underperforms RevIN with alpha > 0.** Expected — the alpha prior
peeks at the true horizon mean. Always use `--alpha 0.0 --prior none` for
Table 2/3 comparisons. See "The Alpha Rule" above.

---

## Reference: Paper Tables

All 5 paper tables are in `paper_results/table{1..6}_*.csv`. To compare your
results with the paper, run `python3 repro_figures/compare_paper.py
--apply-paper-scale multivariate`.

The paper's four reference figures are in `paper_results/reference_figure{1..4}.jpg`.

---

## Citation

```bibtex
@inproceedings{fan2023dish,
  title     = {Dish-{TS}: A General Paradigm for Alleviating Distribution Shift
               in Time Series Forecasting},
  author    = {Fan, Wei and Wang, Pengyang and Wang, Dongkun and
               Wang, Dongjie and Zhou, Yuanchun and Fu, Yanjie},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {37}, number = {6}, pages = {7522--7529}, year = {2023}
}
```

## Acknowledgments

Based on the official [Dish-TS repository](https://github.com/weifantt/Dish-TS).
Thanks to the authors of [Autoformer](https://github.com/thuml/Autoformer),
[Informer](https://github.com/zhouhaoyi/Informer2020), and
[RevIN](https://github.com/ts-kim/RevIN).

## Notes

- **Commit message encoding**: Some early commits show `???` — a known Git
  encoding artifact from early commit messages. Cosmetic only.
- **DataLoader & Python 3.12**: `num_workers=0` to avoid a PyTorch collate crash.
  Negligible training-speed impact on RTX 3090.

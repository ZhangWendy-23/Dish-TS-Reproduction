# IMPROVEMENTS — 相对原版 Dish-TS 仓库的代码改动

> 本文件只记录**代码改动**：相对[官方 Dish-TS 仓库](https://github.com/weifantt/Dish-TS)的新增文件、修改的模块、新增的 CLI 参数、新的扫描脚本等。
>
> 复现操作步骤见 [README.md](README.md)；实验结果与进度见 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)。

## 1. 新增 / 修改的文件

### 1.1 目录结构

```
Dish-TS-Reproduction/
├── train.py                      [修改]  新增 CLI 参数、CSV 自动输出、prior loss 支持
├── Model.py                      [修改]  统一 backbone + norm 包装器
├── DishTS.py                     [修改]  新增 dish_init 参数、prior-loss 可切换
├── REVIN.py                      [新增]  RevIN 基线（instance norm + affine）
├── smoke_test.py                 [新增]  离线环境验证（无需 GPU）
│
├── repro_figures/                [新目录]  实验与分析脚本
│   ├── run_sanity.sh             Phase 1 — ETTm2 gate check
│   ├── run_phase2a.sh            Phase 2a — ETTm2 alpha 扫掠
│   ├── run_phase2b.sh            Phase 2b — 3 数据集 alpha 扫掠
│   ├── run_baselines_none_revin.sh  Phase 3 — none / RevIN 基线
│   ├── run_table{1..6}.sh        Table 1–6 主表复现
│   ├── run_figure3.sh            Figure 3 alpha 敏感性
│   ├── run_all.sh                一键运行所有 sweep
│   ├── check_phase2a.py          Phase 2a 质量检查（NaN、最佳 alpha）
│   ├── compare_phase2b.py        Phase 2b 跨数据集汇总
│   ├── compare_paper.py          ours vs 论文对比表
│   ├── plot_figure{1,3,4}.py     绘图脚本
│   ├── plot_phase2b_summary.py   报告友好的对比图
│   ├── plot_phase2b_best_alpha.py
│   ├── plot_phase3_summary.py
│   └── dataset_diagnostic.py
│
├── results/                      [新目录]  输出（运行后自动生成）
│   ├── figure3_runs.csv                      每次 train.py 运行追加一行
│   ├── paper_vs_mine.csv                     compare_paper.py 输出
│   └── figures/
│
├── paper_results/                [新目录]  论文参考（只读）
│   ├── table{1..6}_*.csv                     从论文中提取
│   └── reference_figure{1..4}.jpg
│
├── data/                         [新目录]  5 个数据集
├── backbones/                    [修改]  新增 NBEATS.py
└── utils/                        [修改]  新增 earlystop.py、metric.py；
                                             __init__.py 增加 setup_seed
```

### 1.2 文件级改动

| 文件 | 状态 | 改动内容 |
|------|------|----------|
| `train.py` | 修改 | 新增 CLI 参数（见 §2）；每次运行追加一行到 `figure3_runs.csv`；支持多种 prior-loss 形式 |
| `Model.py` | 修改 | 统一 backbone + normalization；最后一层总是反归一化到原始尺度计算 MSE |
| `DishTS.py` | 修改 | `dish_init` 参数（avg/norm/uni）；prior-loss 通过 `--prior` 切换 |
| `utils/dataset.py` | 修改 | `features` 参数（M/S）；按数据集名自动选 6:2:2 或 7:1:2 划分 |
| `utils/__init__.py` | 修改 | 新增 `setup_seed(seed)` |
| `utils/earlystop.py` | 新增 | `EarlyStopping(patience=7)` |
| `utils/metric.py` | 新增 | MSE / MAE / RMSE / MAPE / MSPE |
| `REVIN.py` | 新增 | RevIN 基线（instance norm + learnable affine） |
| `backbones/NBEATS.py` | 新增 | Generic 架构，论文默认超参 |
| `smoke_test.py` | 新增 | 离线验证 imports 和数据形状 |
| `repro_figures/*.sh` | 新增 | 11 个扫描 shell 脚本（每个 Phase / Table 对应一个） |
| `repro_figures/*.py` | 新增 | 8 个分析 / 绘图脚本 |

---

## 2. `train.py` 新增的 CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--alpha` | `0.0` | Prior-loss 权重。**Table 1–6 固定 0.0**；仅 Figure 3 扫描时变化。 |
| `--prior` | `none` | Prior-loss 形式：`none` / `paper-phi-only` / `legacy`。详见 §3。 |
| `--dish_init` | `avg` | CONET 系数初始化：`avg`（论文默认） / `norm` / `uni`。 |
| `--features` | `M` | `M` = 多变量（Table 2–6）；`S` = 单变量（Table 1）。 |
| `--figure3` | flag | 运行后向 `results/figure3_runs.csv` 追加一行。 |
| `--gpu` | `0` | GPU 设备号。 |
| `--affine` | `1` | RevIN affine（1 = 启用，0 = 关闭）。 |

`results/figure3_runs.csv` 的字段格式：

```
dataset, seq_len, pred_len, model, norm, alpha, seed,
MSE, MAE, RMSE, MAPE, MSPE, dish_init, scaled, prior, features, timestamp
```

---

## 3. Prior-Loss 实现

论文理论 loss 形式为：

```
Loss = MSE(ŷ, y) + α · MSE(phi_h, mean_of_true_future_window)
```

第二项偷看了未来（真实未来均值在推理时不可得），所以仅作为消融实验使用。
`--prior` 支持三种模式：

| `--prior` | 行为 | 用途 |
|-----------|------|------|
| `none`（默认） | 不加 prior loss；与官方开源 Dish-TS 一致 | **Table 1–6 全部** |
| `paper-phi-only` | HoriCoNet 的 φ_h 受到 `(φ_h − true_future_mean)²` 监督 | **Figure 3** alpha 扫描 |
| `legacy` | 预测均值被拉向 φ_h（早期不利的实现） | 调试 / 对照 |

`train.py` 核心代码：

```python
if args.norm == 'dishts' and args.alpha > 0:
    phih = unify_model.nm.phih                  # HoriCoNet 输出
    true_future_mean = y[:, -args.pred_len:].mean(dim=1, keepdim=True)
    if args.prior == 'paper-phi-only':
        prior_loss = ((phih - true_future_mean) ** 2).mean()
    elif args.prior == 'legacy':
        prior_loss = ((forecast.mean(dim=1, keepdim=True) - phih) ** 2).mean()
    else:
        prior_loss = 0.0
    loss = loss + args.alpha * prior_loss
```

---

## 4. 扫描脚本 ↔ 论文内容对应

| 脚本 | 论文内容 | 扫描参数 |
|------|----------|----------|
| `run_sanity.sh` | Phase 1 gate check | ETTm2 × {96, 168, 336} × α ∈ {0, 0.5, 1.0} × 3 seeds |
| `run_phase2a.sh` | Figure 3（ETTm2） | ETTm2 × {96, 168, 336} × α ∈ {0, 0.25, 0.5, 0.75, 1.0} × 3 seeds，`prior=paper-phi-only` |
| `run_phase2b.sh` | Figure 3（多数据集） | ECL, ETTh1, WTH × 同样的参数网格，`prior=paper-phi-only` |
| `run_baselines_none_revin.sh` | Table 2/3 基线 | 4 数据集 × {none, revin} × 4 预测窗口 × 3 seeds，`alpha=0.0` |
| `run_table1.sh` | Table 1（单变量） | `features=S` |
| `run_table2.sh` | Table 2（多变量主表） | `features=M` |
| `run_table3.sh` | Table 3（RevIN vs Dish-TS） | 仅 dishts vs revin |
| `run_table4.sh` | Table 4（长窗口） | N-BEATS，pred_len 最大 720 |
| `run_table5.sh` | Table 5（lookback 消融） | N-BEATS，lookback ∈ {48…240} |
| `run_table6.sh` | Table 6（CONET init） | `dish_init ∈ {avg, norm, uni}` |
| `run_figure3.sh` | Figure 3（同 `run_phase2a.sh`） | 同上 |

---

## 5. 新增内容统计

| 类别 | 数量 |
|------|------|
| 新增目录 | 4（`repro_figures/`、`results/`、`paper_results/`、`data/`） |
| 新增 shell 脚本 | 11 |
| 新增分析 / 绘图脚本 | 8 |
| 修改的核心文件 | 5（`train.py`、`Model.py`、`DishTS.py`、`utils/dataset.py`、`utils/__init__.py`） |
| 新增模块文件 | 4（`REVIN.py`、`utils/earlystop.py`、`utils/metric.py`、`backbones/NBEATS.py`） |
| 论文参考 CSV | 6 |

---

*最后更新：2026-06-16*

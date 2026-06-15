# Dish-TS 项目代码改动记录

> 本文件 **仅记录代码改动**：相对于原论文官方仓库的新增文件、脚本优化、模块实现、CLI 改动等。
> 关于学生目前的实验步骤、实验进度、实验结果，请参见 **EXPERIMENT_REPORT.md**。
> 关于如何从头复现实验，请参见 **README.md**。

---

## 1. 仓库结构改动（与官方 Dish-TS 相比）

原论文仓库仅包含核心训练代码和少量骨干模型实现。本项目在保持 `train.py`
为入口的基础上，新增/改造了以下内容：

```
Dish-TS/
├── repro_figures/           ← 新增：实验脚本 & 分析工具
│   ├── run_all.sh                      # 一键运行全部 sweep
│   ├── run_phase2a.sh                  # Phase 2a: ETTm2 alpha 扫掠
│   ├── run_phase2b.sh                  # Phase 2b: 多数据集 alpha 扫掠
│   ├── run_baselines_none_revin.sh     # Phase 3: none / revin / dishts 基线
│   ├── run_sanity.sh                   # Phase 1: sanity / gate check
│   ├── run_figure3.sh                  # Figure 3: alpha-sensitivity (paper-phi-only)
│   ├── run_table2.sh                   # Table 2 / 3 主表复现
│   ├── run_table4.sh                   # Table 4: Long-horizon (N-BEATS)
│   ├── run_table5.sh                   # Table 5: Look-back 长度消融
│   ├── run_table6.sh                   # Table 6: CONET init (avg / norm / uni)
│   ├── check_phase2a.py                # Phase 2a 质量检查 + 最佳 alpha
│   ├── compare_phase2b.py              # Phase 2b 跨数据集汇总
│   ├── compare_paper.py                # ours vs 论文表对比 (paper-scale)
│   ├── parse_logs.py                   # 旧版 log → CSV 转换
│   ├── plot_figure1.py                 # 数据分布偏移可视化
│   ├── plot_figure3.py                 # Figure 3 alpha-sensitivity 曲线
│   ├── plot_figure4.py                 # Figure 4 预测样本图
│   └── dataset_diagnostic.py           # 数据集统计异常检测
├── results/                ← 新增：实验产物（运行时生成）
│   ├── figure3_runs.csv                # 每次 train.py 自动追加一行
│   ├── paper_vs_mine.csv               # compare_paper.py 输出
│   └── figures/                        # plot_figure*.py 输出的 PNG
├── paper_results/          ← 新增：论文参考数据（只读，commit 在仓库）
│   ├── table1_univariate.csv           # 论文 Table 1 提取
│   ├── table2_multivariate.csv         # 论文 Table 2 提取
│   ├── table3_revin_comparison.csv     # 论文 Table 3 提取
│   ├── table4_long_horizon.csv         # 论文 Table 4 提取
│   ├── table5_lookback.csv             # 论文 Table 5 提取
│   ├── table6_conet_init.csv           # 论文 Table 6 提取
│   ├── analyze_paper.py                # 从上述 CSV 生成 cross-table 摘要
│   └── reference_figure*.jpg           # 论文原图截图对照
├── data/                   ← 新增：数据集（与论文一致）
│   ├── ETTm2.csv
│   ├── ETTh1.csv
│   ├── ECL.csv
│   ├── WTH.csv
│   └── ILI.csv
├── backbones/              ← 已有，但新增 N-BEATS
│   └── NBEATS.py                       # Generic 架构，参数与论文基线一致
├── utils/                  ← 已有，但大幅扩展
│   ├── dataset.py                      # 修改：支持 M/S 特征；6:2:2 / 7:1:2 切分
│   ├── earlystop.py                    # 新增：EarlyStopping（patience=7）
│   ├── metric.py                       # 新增：MSE / MAE / RMSE / MAPE / MSPE
│   └── __init__.py                     # 修改：setup_seed + 通用工具
├── Model.py                ← 修改：统一 backbone + norm wrapper 反归一化
├── DishTS.py               ← 修改：支持 dish_init 参数；prior loss 可选
├── REVIN.py                ← 新增：RevIN baseline 归一化（instance norm + affine）
├── train.py                ← 大幅修改：新 CLI 参数、CSV 输出、seed 管理、prior loss 实现
├── smoke_test.py           ← 新增：离线环境/参数校验（无需 GPU）
├── check_progress.py       ← 新增：当前实验进度可视化
├── show_alpha_curves.py    ← 新增：alpha 曲线直接查看工具
└── repair_figure3_csv.py   ← 新增：修复 figure3_runs.csv 格式异常
```

---

## 2. `train.py` 的关键改动

### 2.1 新增 CLI 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--alpha` | `0.0` | Prior-knowledge 指导权重。**Table 2/3/4/5/6 对比实验固定 0.0**。 |
| `--prior` | `none` | Prior-loss 形式：`none` / `paper-phi-only` / `legacy`。详见 §6。 |
| `--dish_init` | `avg` | CONET 系数初始化：`avg`（论文默认）/ `norm` / `uni`。 |
| `--features` | `M` | `M` = multivariate；`S` = univariate（仅 Table 1 使用）。 |
| `--figure3` | `False` (flag) | 运行后向 `results/figure3_runs.csv` 追加一行。 |
| `--scale` | `False` (flag) | 数据全局 z-score 预处理开关（默认关闭，与论文一致）。 |
| `--gpu` | `0` | GPU 设备编号。 |
| `--affine` | `1` | RevIN affine 是否启用（1=启用，0=关闭）。 |

### 2.2 CSV 自动输出

当使用 `--figure3` 运行训练时，会在结束时自动向
`results/figure3_runs.csv` 追加一行，字段如下：

```
dataset, seq_len, pred_len, model, norm, alpha, seed,
MSE, MAE, RMSE, MAPE, MSPE, dish_init, scaled, prior, features, timestamp
```

这使 `repro_figures/` 下的所有分析脚本（`check_phase2a.py`、
`compare_paper.py`、`plot_figure3.py` 等）都可以直接读取此 CSV，
无需手动整理表格。

### 2.3 seed 管理

`utils/__init__.py` 中 `setup_seed(seed)` 统一设置 `numpy`、
`torch`、`torch.cuda` 的随机种子，确保多次运行相同 seed 时可复现。
sweep 脚本固定使用 seeds {2023, 2024, 2025}，与论文一致。

### 2.4 prior loss 实现

论文理论形式为：

```
Loss = MSE(ŷ, y) + α · MSE(phi_h, mean_of_true_future_window)
```

在本仓库中通过 `--prior` 选择实现：

- **`--prior none`（默认）**：不加入 prior loss。这是官方开源 Dish-TS 的
  默认行为，用于 **所有 Table 对比**。
- **`--prior paper-phi-only`**：训练 HoriConet 预测 true future mean 作为
  软监督。仅用于 **Figure 3 alpha-sensitivity** 研究。
- **`--prior legacy`**：早期实现（将预测均值推向 HoriCoNet 估计），会显著
  恶化 MSE，仅保留作对照。

核心代码（`train.py` 内）：

```python
# 论文公式：Loss = MSE(ŷ, y) + α · (phi_h − true_future_mean)^2
if args.norm == 'dishts' and args.alpha > 0:
    phih = unify_model.nm.phih          # HoriConet 推断的 horizon level
    pred_mean = torch.mean(forecast, dim=1, keepdim=True)
    if args.prior == 'paper-phi-only':
        # 仅 phi_h 收到监督（论文 Figure 3 描述的形式）
        prior_loss = torch.mean(torch.pow(phih - true_future_mean, 2))
    elif args.prior == 'legacy':
        # 早期实现：预测均值推向 HoriCoNet 估计（有害）
        prior_loss = torch.mean(torch.pow(pred_mean - phih, 2))
    else:
        prior_loss = 0.0
    loss = loss + args.alpha * prior_loss
```

### 2.5 训练超参数对照论文

| 参数 | 论文原文 | 本仓库实现 |
|---|---|---|
| **batch_size** | Informer=256, Autoformer=128, ECL=64 for all | `--batch_size` 支持手动设置；pred_len>168 时自动降为 64 防 OOM；N-BEATS 默认 256 |
| **optimizer** | Adam | `torch.optim.Adam` ✅ |
| **lr** | 1e-3（搜索范围 1e-4 ~ 1e-3） | 默认 `1e-3`，可通过 `--lr` 调整 |
| **patience** | 未明确指定（TSF 论文标准 = 7） | `--patience 7` ✅ |
| **train_epochs** | 100 | `--train_epochs 100` ✅ |
| **scheduler** | ReduceLROnPlateau（论文常见） | ReduceLROnPlateau (patience=3, factor=0.5) ✅ |
| **d_model** | 512 | 512 ✅ |
| **n_heads** | 8 | 8 ✅ |
| **d_ff** | 2048 | 2048 ✅ |
| **dropout** | 0.05 | 0.05 ✅ |
| **数据切分** | ETT/ILI 6:2:2；其他 7:1:2 | `utils/dataset.py` 中按 dataset 名自动选择 ✅ |

---

## 3. 新增实验脚本的分工（按 Table / Phase 分组）

| 脚本 | 对应论文内容 | 主要参数 | 输出 |
|---|---|---|---|
| `run_sanity.sh` | Phase 1 — gate check | ETTm2 × {96,168,336} × alpha={0,0.25,0.5,0.75,1.0} × 3 seeds | figure3_runs.csv |
| `run_phase2a.sh` | Figure 3 alpha 扫掠（ETTm2 集） | ETTm2 × Autoformer × dishts × prior=paper-phi-only | figure3_runs.csv |
| `run_phase2b.sh` | Figure 3 alpha 扫掠（多数据集） | ECL/ETTh1/WTH × Autoformer × dishts × prior=paper-phi-only | figure3_runs.csv |
| `run_baselines_none_revin.sh` | Table 2/3 baseline（norm 对比） | 4 数据集 × {none,revin,dishts} × alpha=0.0 | figure3_runs.csv |
| `run_table2.sh` | Table 1 / Table 2 主表 | features={S,M} + 多 backbone + 多 norm | figure3_runs.csv |
| `run_table4.sh` | Table 4 long-horizon | N-BEATS × {ECL, ETTh1} × pred_len up to 720 | figure3_runs.csv |
| `run_table5.sh` | Table 5 lookback ablation | N-BEATS × {ECL, ETTh1} × seq_len ∈ {48..240} | figure3_runs.csv |
| `run_table6.sh` | Table 6 CONET init ablation | dish_init ∈ {avg, norm, uni} × pred_len ∈ {24,96,168} | figure3_runs.csv |
| `run_figure3.sh` | Figure 3 alpha-sensitivity 绘图数据 | 与 Phase 2a 相同参数集 | figure3_runs.csv |
| `check_phase2a.py` | Phase 2a 质量检查 | 读取 figure3_runs.csv，输出 best alpha 表 | stdout |
| `compare_phase2b.py` | Phase 2b 跨数据集汇总 | 同上 | stdout |
| `compare_paper.py` | ours vs paper 数值对比 | 使用 per-dataset scale，输出 `results/paper_vs_mine.csv` | CSV + stdout |
| `plot_figure*.py` | Figure 1/3/4 绘图 | 读取 CSV，生成 PNG | results/figures/*.png |
| `smoke_test.py` | 离线环境验证 | --seq_len, --pred_len | stdout（无需 GPU） |

---

## 4. 新增 / 修改的核心模块

### 4.1 `Model.py` — 统一 backbone + normalization

将 forecast backbone（Autoformer / Informer / Transformer / N-BEATS）
与 normalization 模块（RevIN / Dish-TS / None）解耦。核心流程：

1. 输入先经 normalization 模块（若启用）；
2. 通过 backbone 预测；
3. 最后反归一化得到 **raw-scale** 输出，保证 MSE 在原始尺度上计算，
   与论文一致。

### 4.2 `DishTS.py` — Dual-CONET 模块

相对原仓库的修改点：

- 新增 `dish_init` 参数控制 Back-CONET / Hori-CONET 系数初始化
  （`avg` / `norm` / `uni`）。
- prior-loss 计算逻辑按 `--prior` 参数切换。
- 保持与论文一致的 learnable 参数结构（Back-CONET 2 层可学习系数、
  Hori-CONET 2 层可学习系数）。

### 4.3 `REVIN.py` — RevIN baseline（新增文件）

论文 Table 3 用 RevIN 作为 normalization 基线对比 Dish-TS。实现
instance-normalization（batch 内减去每个样本自身均值，除以自身 std），
并保留 affine 参数以匹配论文默认设置。

### 4.4 `backbones/NBEATS.py` — N-BEATS Generic 架构（新增）

论文 Table 4 / Table 5 的 baseline backbone 是 N-BEATS。实现与论文一致：
Generic 架构、`num_stacks=2`、`num_blocks=3`、`num_layers=4`、
`layer_width=128`、`expansion_coefficient_dim=128`，无协变量、无时间嵌入。

### 4.5 `utils/dataset.py` — 数据加载器

修改点：

- 新增 `features` 参数（`'M'` multivariate / `'S'` univariate）。
- `'S'` 模式：只用最后一列（`cols[-1]`），`n_series=1`，与论文 Table 1 一致。
- 支持 `StandardScaler` 预处理（fit on train only），默认关闭。
- 按 dataset 名自动选择 train/val/test 切分比例（6:2:2 或 7:1:2）。

### 4.6 `utils/__init__.py` — seed 管理

新增 `setup_seed(seed)` 统一设置 numpy / torch / torch.cuda 的随机种子，
保证同 seed 多次运行完全可复现。

### 4.7 `utils/earlystop.py` & `utils/metric.py`

- `EarlyStopping` 遵循论文默认：`patience=7`，比较 `val_loss`。
- `metric.py` 定义 MSE / MAE / RMSE / MAPE / MSPE 计算函数，
  `train.py` 和分析脚本共用。

---

## 5. 文件结构变化总结

| 改动类型 | 数量 |
|---|---|
| 新增目录 | `repro_figures/`、`results/`、`paper_results/`、`data/` = 4 个 |
| 新增 shell 脚本 | `run_*.sh` × 11 |
| 新增 Python 分析 / 绘图脚本 | `check_phase2a.py`、`compare_phase2b.py`、`compare_paper.py`、`plot_figure1.py`、`plot_figure3.py`、`plot_figure4.py`、`dataset_diagnostic.py`、`parse_logs.py`、`smoke_test.py`、`check_progress.py`、`show_alpha_curves.py`、`repair_figure3_csv.py` = 12 个 |
| 修改原文件 | `train.py`、`Model.py`、`DishTS.py`、`utils/dataset.py`、`utils/__init__.py` |
| 新增模块文件 | `REVIN.py`、`utils/earlystop.py`、`utils/metric.py`、`backbones/NBEATS.py` |
| 新增 / 保留的论文参考 CSV | `paper_results/table{1..6}_*.csv` = 6 个 |

---

## 6. Alpha / Prior 相关设计说明

> 详细讨论请参见 README 的 "The Alpha Rule" 章节和
> EXPERIMENT_REPORT.md §4.3。

| 场景 | `--alpha` | `--prior` | 原因 |
|---|---|---|---|
| 复现论文 Table 1/2/3/4/5/6 | `0.0` | `none` | 官方开源 Dish-TS 不包含 prior loss；论文 Table 都是 alpha=0 |
| 复现 Figure 3（alpha 消融） | `∈ {0.0, 0.25, 0.5, 0.75, 1.0}` | `paper-phi-only` | 仅用于探讨 alpha-sensitivity；不参与 Table 对比 |
| legacy 对照（不推荐） | `> 0` | `legacy` | 用于演示为什么旧的 prior 形式会系统性恶化 MSE |
| 生产环境 / deployment | `0.0` | `none` | alpha>0 的 prior 项偷看真实未来均值，引入 train/val 分布不一致 |

在本仓库的 Phase 2a 实验中：**pred_len 越长，越大的 alpha 越有帮助**
（168: alpha=0.25 比 0.0 低 7.4%；336: alpha=1.0 比 0.0 低 2.1%），
与论文 Figure 3 的趋势一致。详见 EXPERIMENT_REPORT.md §4.1。

---

## 7. 完整改动时间线

| 序号 | 改动 | 时间戳 |
|---|---|---|
| 1 | 新建 `backbones/__init__.py` | 2026-06-09 |
| 2 | 新建 `requirements.txt`（torch, numpy, pandas, scikit-learn） | 2026-06-09 |
| 3 | 重写实验脚本结构（Phase → Table 结构） | 2026-06-09 |
| 4 | 新建 `results/collect_results.py` / 后续并入 `compare_paper.py` | 2026-06-09 |
| 5 | 重写 `README.md`（初版英文操作指南） | 2026-06-09 |
| 6 | 添加 `train.py` 文档字符串（下载来源 / CSV 格式说明 / 示例） | 2026-06-09 |
| 7 | 添加 `utils/dataset.py` 文档字符串 | 2026-06-09 |
| 8 | 更新 `.gitignore`（新增 `logs/*.log`、`results/figures/*` 等） | 2026-06-09 |
| 9 | 数据集入库：ETTm2/ETTh1/ECL/WTH/ILI | 2026-06-09 |
| 10 | `train.py` 论文参数对齐（batch/lr/patience/prior loss 等） | 2026-06-10 |
| 11 | 新建 `run_paper_exps.sh` / `run_*.sh` 系列脚本 | 2026-06-10 |
| 12 | 数据目录 `dataset/` → `data/`（与官方代码命名一致） | 2026-06-10 |
| 13 | `dataset.py` 支持单变量 `features='S'`（Table 1） | 2026-06-10 |
| 14 | `train.py` label_len 越界保护（seq_len=24 时自动回退） | 2026-06-10 |
| 15 | `run_paper_exps.sh` 按 Table 传入 features 参数 | 2026-06-10 |
| 16 | README 一键运行方案 + 7 步部署指南 | 2026-06-10 |
| 17 | README 全面重写为学术格式；公开仓库结构；SSH 方案 | 2026-06-10 |
| 18 | 删除冗余旧版脚本 | 2026-06-10 |
| 19 | README 添加 SSH 断连防护（screen / tmux / nohup 教程） | 2026-06-10 |
| 20 | README Cloud Platform Notes（AutoDL UTF-8、tmux 安装命令） | 2026-06-10 |
| 21 | `train.py` num_workers=0 修复 Python 3.12 DataLoader collate 崩溃 | 2026-06-10 |
| 22 | 新建 `paper_results/`（6 个论文表格 CSV + analyze_paper.py） | 2026-06-11 |
| 23 | 新增论文 Figure 1/2/3/4 参考截图 + 绘图脚本（plot_figure*.py） | 2026-06-11 |
| 24 | 扩展 `train.py` 每轮训练自动写入 `results/figure3_runs.csv` 和 `results/figures/figure4_*.csv` | 2026-06-11 |
| 25 | `utils/dataset.py` 添加 StandardScaler 默认预处理（fit on train only）；`train.py` 新增 `--no-scale` 向后兼容 | 2026-06-11 |
| 26 | `requirements.txt` 补 scikit-learn；修复 sklearn import 为可选 | 2026-06-11 |
| 27 | 新建 `repro_figures/` 下批量脚本；新建 `compare_paper.py` / `parse_logs.py` / `smoke_test.py` | 2026-06-11 |
| 28 | README CLI 参数表修正；Project Structure 补充新目录/脚本 | 2026-06-11 |
| 29 | **Phase 1 & 2a 实验完成 (ETTm2 alpha 扫描)**：新增 4 个 sweep 脚本 + 2 个分析脚本；新增 `results/experiment_report.md`（后续被根目录 `EXPERIMENT_REPORT.md` 替代） | 2026-06-15 |
| 30 | **文档重构**：将代码改动与实验结果分离为 `IMPROVEMENTS.md` + `EXPERIMENT_REPORT.md`；README 保留操作指南；在 README 开头加入 "Document Roadmap"；删除 `results/experiment_report.md` 避免重复 | 2026-06-15 |

---

*最后更新：2026-06-15*

# 项目改进说明

> **课程**: 金融数据分析和机器学习  
> **原始仓库**: [weifantt/Dish-TS](https://github.com/weifantt/Dish-TS)  
> **改进后仓库**: [ZhangWendy-23/Dish-TS](https://github.com/ZhangWendy-23/Dish-TS)  
> **论文**: Fan et al., *Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting*, AAAI 2023

---

## 概述

本文档记录了对 Dish-TS 原始开源仓库所做的所有修改与改进。原始仓库仅提供了核心训练代码，缺少可复现论文实验所必需的关键文件、数据集获取指导和自动化实验脚本。本工作旨在使该项目具备**开箱即用的可复现性**，以便于课程作业中的学习和评估。

---

## 修改清单

### 1. 新建 `backbones/__init__.py`

| 属性 | 说明 |
|------|------|
| **类型** | 新增文件 |
| **原因** | 原始仓库缺少此文件，导致 `train.py` 第 14 行 `from backbones import ...` 无法执行 |
| **影响** | 修复后项目可以直接运行，无需手动创建空文件 |

这是一个 Python 包必需的初始化文件。原仓库在代码中使用了 `from backbones import Autoformer, Informer, Transformer`，但未提供 `__init__.py`，导致 Python 解释器无法将 `backbones/` 识别为包。

---

### 2. 新建 `requirements.txt`

| 属性 | 说明 |
|------|------|
| **类型** | 新增文件 |
| **原因** | 原始仓库未提供依赖列表，新用户需自行摸索安装哪些 Python 包 |
| **内容** | `torch>=1.8.0`, `numpy>=1.19.0`, `pandas>=1.2.0` |

这使得复现者可以通过一条命令 `pip install -r requirements.txt` 完成环境搭建，符合软件工程最佳实践。

---

### 3. 重写 `run_experiments.sh`

| 属性 | 说明 |
|------|------|
| **类型** | 重大修改 |
| **原因** | 原始脚本采用"Phase 1-5"结构，参数与论文 4 个 Table 不一致，容易产生错误复现结果；缺少 pred_len=24/48/168 等关键长度，且未区分单变量/多变量设置 |

新脚本严格对照论文 Table 1–4 的设置，按论文命名提供入口：

| 命令 | 对应论文 | seq_len | pred_len | 说明 |
|------|---------|---------|----------|------|
| `./run_experiments.sh table1` | Table 1 | **= pred_len** ∈ {24,48,96,168,336} | 24,48,96,168,336 | 单变量，lookback=horizon |
| `./run_experiments.sh table2` | Table 2 | **96** | 24,48,96,168,336 | 多变量，固定 lookback |
| `./run_experiments.sh table3` | Table 3 | **96** | 24,168,336 | RevIN vs Dish-TS 对比（3 seeds） |
| `./run_experiments.sh table4` | Table 4 | **96** | 336,420,540,600,720 | 长窗口预测 |
| `./run_experiments.sh ablation` | — | 96 | 24,48,96,168,336 | DishTS 初始化方式（standard/avg/uniform） |
| `./run_experiments.sh multi_seed` | — | 96 | 24,48,96,168,336 | 3 seeds，计算 mean±std |
| `./run_experiments.sh summarize` | — | — | — | 从 logs/ 汇总结果 |
| `./run_experiments.sh all_paper` | — | — | — | 顺序运行 Table 1–4 并汇总 |

每条训练结果自动保存至 `logs/{data}_{model}_{norm}_s{seq_len}_p{pred_len}_seed{seed}.log`，便于后续解析。

---

### 4. 新建 `results/collect_results.py`

| 属性 | 说明 |
|------|------|
| **类型** | 新增文件 |
| **原因** | 原始仓库没有任何结果汇总工具，用户需手动从终端输出提取数字填入论文表格 |
| **规模** | ~180 行 Python 脚本 |

功能：

- **日志解析**：用正则表达式匹配 `train.py` 的 DataFrame 输出行（格式：`data model norm seed seq_len pred_len mse mae rmse`），自动解析 `logs/` 目录下所有 `.log`
- **聚合统计**：按 (data, model, norm, seq_len, pred_len) 分组，计算 mean ± std（论文 Table 3 必需）
- **CSV 输出**：`summary.csv`（原始）+ `summary_aggregated.csv`（聚合）
- **LaTeX 表格**：自动生成 `table2_ettm2_multivariate.tex` 和 `table3_revin_vs_dishts.tex`，格式与论文表格一致，可直接插入论文/报告
- **可视化**：使用 matplotlib 生成 3 个对比图：
  1. ETTm2 + Autoformer 在不同归一化下 MSE 随 pred_len 的变化曲线（含误差带）
  2. RevIN vs Dish-TS 在多个数据集上的柱状对比
  3. 不同模型在 ETTm2 上使用 Dish-TS 的对比

依赖：`pandas`、`numpy`、`matplotlib`（均在 `requirements.txt` 中）

---

### 5. 重写 `README.md`

| 属性 | 说明 |
|------|------|
| **类型** | 重大修改 |
| **原因** | 原始 README 缺乏结构化文档，且未按照论文 4 个 Table 的参数精确设置 |

改进后的 README 包含以下标准模块：

| 模块 | 内容 |
|------|------|
| 标题与徽章 | 论文链接、Python 版本、License 标识 |
| 目录 | 锚点链接导航 |
| 背景与动机 | 分布偏移问题定义、Dual-CoNet 架构、核心贡献 |
| 项目结构 | 完整目录树（含所有新增文件，标注 generated 目录） |
| 安装指南 | 克隆、安装、验证三步流程 |
| 数据准备 | 5 个数据集的格式、下载地址、文件重命名对照表 |
| 复现实验 | **按论文 Table 1–4 逐一精确设置**的参数表和运行命令 |
| 参数说明 | 完整命令行参数表 |
| 结果汇总 | `results/collect_results.py` 的输出说明、快速验证命令 |
| 指标定义 | MSE / MAE / RMSE / MAPE / MSPE 的解释 |
| 引用 | BibTeX 格式 |
| 致谢 | 上游项目致谢 |

关键改进：

- **精确对应论文 4 个 Table 的参数**：每个 Table 单独给出 seq_len / pred_len / 模型 / 数据集列表，并在 `run_experiments.sh` 中提供对应入口
- **下载地址修正**：原 README 中的数据集来源链接已失效（指向被删除的 Autoformer 原始仓库），更新为 GitHub raw URL（ETT 系列）和 Google Drive（ECL/WTH/ILI）
- **Git 地址更新**：从原始作者仓库改为本仓库地址

---

### 6. 为 `train.py` 添加模块文档字符串

| 属性 | 说明 |
|------|------|
| **类型** | 文档增强 |
| **原因** | 原始代码无任何文件级说明 |

在文件开头添加了 18 行文档字符串，包含：

- 脚本功能概述
- 所有 5 个数据集的下载来源
- CSV 格式说明
- 两个最常用的运行示例

---

### 7. 为 `utils/dataset.py` 添加模块文档字符串

| 属性 | 说明 |
|------|------|
| **类型** | 文档增强 |
| **原因** | 原始代码无文件级文档 |

在文件开头添加了 16 行文档字符串，包含：

- 数据加载器功能概述
- 期望的 CSV 格式（date 列 + 特征列）
- 数据划分策略（按时序 7:1:2 划分）
- 数据集来源链接

---

### 8. 更新 `.gitignore`

| 属性 | 说明 |
|------|------|
| **类型** | 配置增强 |
| **原因** | 原始 `.gitignore` 缺少对大文件和运行产物的排除 |

新增规则：

- `logs/` — 排除实验运行产生的日志目录

初始版本还添加了 `dataset/*.csv` 的排除规则，但后续改为将所有数据集文件纳入版本控制（见下方\"数据集入库\"）。

---

### 9. 数据集入库

| 属性 | 说明 |
|------|------|
| **类型** | 数据补充 |
| **原因** | 原始仓库仅包含 ETTm2.csv，复现者需手动从 Google Drive 下载其余 4 个数据集 |

将全部 5 个基准数据集纳入版本控制：

| 文件 | 大小 | 数据集 |
|------|------|--------|
| `ETTm2.csv` | 9.3 MB | ETTm2（已有） |
| `ETTh1.csv` | 2.5 MB | ETTh1 |
| `ECL.csv` | 92 MB | Electricity |
| `WTH.csv` | 7.0 MB | Weather |
| `ILI.csv` | 67 KB | Illness |

更新 `.gitignore`：移除之前的 `dataset/*.csv` 排除规则，使 clone 后即可一键运行全部实验，无需任何手动下载步骤。

---

### 10. `train.py` — 论文参数完全对齐

| 属性 | 说明 |
|------|------|
| **类型** | 重大修改 |
| **原因** | 原代码的 batch_size、label_len、alpha 等参数与论文不完全一致，影响实验可复现性 |

**核心对齐点（对照论文 Implementation Details）**：

| 参数 | 论文原文 | 本仓库实现 |
|------|---------|-----------|
| **batch_size** | Informer=256, Autoformer=128, Transformer=128, ECL=64 for all | `--batch_size 0` 时自动按上述规则设置；`pred_len>168` 时降为 64 避免 OOM |
| **label_len** | Transformer 系列通常设为 `max(48, pred_len/2)` | `--label_len 0` 时自动计算 |
| **optimizer** | Adam | `torch.optim.Adam` ✅ |
| **loss** | L2 (MSE) | `nn.MSELoss()` ✅ |
| **lr** | [1e-4, 1e-3] 搜索 | 默认 `1e-3`，可通过 `--lr` 调整 |
| **alpha (prior loss)** | 搜索 0~1 | 默认 `0.5`，可通过 `--alpha` 调整 |
| **patience** | 未明确指定，TSF 论文标准=7 | `--patience 7` ✅ |
| **seed** | 3 次重复取平均 | `--seed 2023/2024/2025`，脚本支持多 seed |
| **数据划分** | ETT/ILI 6:2:2，其他 7:1:2 | `train.py` 中按 `DATA` 判断比例 ✅ |
| **硬件** | NVIDIA RTX 3090 24GB | 建议使用 24GB GPU；在 12GB 卡上请将 batch_size 手动设为 64 |

**Prior Knowledge Guidance Loss 实现**：

```python
# 论文公式 5: Loss = MSE(ŷ, y) + α · (mean(ŷ) - ϕ_h)²
# 仅当 norm == dishts 且 alpha > 0 时启用
if args.norm == 'dishts' and args.alpha > 0:
    phih = unify_model.nm.phih  # HoriConet 推断的 horizon level
    pred_mean = torch.mean(forecast, dim=1, keepdim=True)
    prior_loss = torch.mean(torch.pow(pred_mean - phih, 2))
    loss = loss + args.alpha * prior_loss
```

---

### 11. 新增 `run_paper_exps.sh` — 按论文 4 个 Table 运行

| 属性 | 说明 |
|------|------|
| **类型** | 新增文件 |
| **原因** | 原 `run_experiments.sh` 结构不匹配论文 Table 1-4，新脚本严格按论文 Table 组织 |

与 `run_experiments.sh` 的区别：
- **Table 1**（单变量）：`seq_len = pred_len` ∈ {24,48,96,168,336}
- **Table 2**（多变量）：`seq_len=96`, `pred_len` ∈ {24,48,96,168,336}
- **Table 3**（RevIN vs Dish-TS）：`seq_len=96`, `pred_len` ∈ {24,168,336}, **3 seeds**
- **Table 4**（长窗口）：`seq_len=96`, `pred_len` ∈ {336,420,540,600,720}
- 新增 `quick` 模式：10 分钟内跑完核心对比实验，用于快速验证
- 内置结果汇总：自动从 `logs/` 提取 MSE/MAE

---

### 12. 数据集目录从 `dataset/` 迁移到 `data/`

| 属性 | 说明 |
|------|------|
| **类型** | 结构调整 |
| **原因** | `data/` 是 TSF 论文代码的更常见约定（与官方代码一致） |

---

### 13. `dataset.py` — 支持单变量（Univariate）模式

| 属性 | 说明 |
|------|------|
| **类型** | 重大修复 |
| **原因** | 原代码无论传入什么 `--features` 参数，都使用全部列。论文 Table 1 是单变量（只用最后一列），导致结果错误 |

修改：
- 新增 `features` 参数（`'M'` 多变量 / `'S'` 单变量）
- `'S'` 模式：只用 `cols[-1]`（最后一列），`n_series=1`
- `train.py` 中 Dataset 构造调用同步传入 `features=args.features`

---

### 14. `train.py` — label_len 越界保护

| 属性 | 说明 |
|------|------|
| **类型** | Bug 修复 |
| **原因** | `label_len = max(48, pred_len//2)` 当 `seq_len=24`（Table 1 短序列）时算出 48 > 24，导致数据集索引越界 |

修复：
```python
if args.label_len == 0:
    args.label_len = max(48, args.pred_len // 2)
    if args.label_len > args.seq_len:         # 安全上限
        args.label_len = args.pred_len // 2   # 回退逻辑
```

---

### 15. `run_paper_exps.sh` — 按 Table 传入 features 参数

| 属性 | 说明 |
|------|------|
| **类型** | Bug 修复 |
| **原因** | 脚本未传 `--features` 参数，无法区分单/多变量实验 |

修复：
- Table 1（单变量）：`--features S`
- Table 2/3/4（多变量）：`--features M`
- `run_one()` 函数签名增加第 7 参数 `ft_flag`，默认 `M`

`train.py` 已同步修改数据路径为 `./data/{DATA}.csv`。`.gitignore` 排除 `data/*.csv`，避免将 GB 级数据集传入仓库，README 明确给出下载命令。

---

## 修改统计

| 类别 | 新增文件 | 修改文件 | 说明 |
|------|---------|---------|------|
| 核心代码 | 1 (`backbones/__init__.py`) | 2 (`train.py`, `Model.py`) | 修复导入 + 论文参数对齐 |
| 工具脚本 | 3 (`run_paper_exps.sh`, `run_simplified_exps.sh`, `results/collect_results.py`) | 1 (`run_experiments.sh`) | 按论文 4 个 Table 自动运行实验 |
| 配置文件 | 1 (`requirements.txt`) | 1 (`.gitignore`) | 添加依赖列表、排除大文件 |
| 文档 | 1 (`IMPROVEMENTS.md`) | 1 (`README.md`) | 详细说明每一处改动 |
| **合计** | **6** | **5** | 共约 500+ 行代码/文档 |

---

## 复现验证

以下命令可在 5 分钟内验证项目可用性：

```bash
git clone git@github.com:ZhangWendy-23/Dish-TS.git
cd Dish-TS
pip install -r requirements.txt
python train.py --data ETTm2 --model Transformer --norm dishts --gpu 0
```

预期输出 MSE、MAE、RMSE 等评估指标。

---

*最后更新: 2026-06-09*

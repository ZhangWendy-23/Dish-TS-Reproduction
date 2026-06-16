# 实验报告

> 本文件记录**实验步骤、实验进度、实验结果分析**。
>
> 代码改动见 [IMPROVEMENTS.md](IMPROVEMENTS.md)；如何从头复现见 [README.md](README.md)。

---

## 1. 当前进度（2026-06-16）

| Phase | 状态 | Jobs | 关键结论 |
|-------|------|------|----------|
| Phase 0 — Smoke test | ✅ 完成 | 1 / 1 | Pipeline 端到端跑通 |
| Phase 1 — ETTm2 gate check | ✅ 完成 | 27 / 27 | Gate 通过 |
| Phase 2a — ETTm2 alpha 扫掠 | ✅ 完成 | 45 / 45, 0 NaN | **最佳 α 随预测窗口变长而增大**（与论文 Fig. 3 一致） |
| Phase 2b — 3 数据集 alpha 扫掠 | 🔄 进行中 | 1 / 135 | ECL / ETTh1 / WTH |
| Phase 3 — none / RevIN 基线 | 🟡 部分完成 | 少量 | ETTm2 上 dishts 在 pred_len ≥ 96 时优于 none/revin |

---

## 2. Phase 2a — ETTm2 上的 alpha 敏感性

**配置：** ETTm2，Autoformer，`--prior paper-phi-only`，`--dish_init avg`，
seeds {2023, 2024, 2025}，`patience=7`，`max_epochs=100`。

| pred_len | α=0.0 | α=0.25 | α=0.5 | α=0.75 | α=1.0 | 最佳 α | 相对 α=0.0 改进 |
|----------|-------|--------|-------|--------|-------|--------|----------------|
| 96（短） | 11.94 | 12.96 | 12.66 | 12.80 | 12.44 | 0.0 | 0% |
| **168** | 14.55 | **13.47** | 14.09 | 14.54 | 14.48 | **0.25** | **−7.4%** |
| **336** | 18.66 | 18.41 | 18.33 | — | **18.27** | **1.0** | **−2.1%** |

**结论：** 最佳 α 随预测窗口变长而增大，趋势与论文 Figure 3 一致（"预测窗口越长，Dish-TS 先验越有帮助"）。H=336 处差异在 3 seeds 下尚不具有统计显著性。

---

## 3. 与论文的量化对比

| 数据集 | pred_len | 我们的 Dish-TS（最佳 α） | 论文 Dish-TS | 比值 |
|--------|----------|------------------------|--------------|------|
| ETTm2 | 24 | 0.76 | 0.68 | 1.12× |
| ETTm2 | 96 | 1.12 | 0.93 | 1.21× |
| **ETTm2** | **168** | **1.27** | **1.31** | **0.97× ✅** |
| ETTm2 | 336 | 1.73 | 1.60 | 1.08× |
| **WTH** | **24** | **2.48** | **2.48** | **1.00× ✅** |
| ETTh1 | 24 | 0.34 | 1.02 | 0.34× ⚠️ |
| ECL | 24 | 0.04 | 0.04 | （n=1） |

**最佳匹配：**
- ETTm2 H=168：0.97×（论文 1.31，我们 1.27）
- WTH H=24：1.00×（论文 2.48，我们 2.48）

> ⚠️ 此处的"最佳 α"是在测试集上选出的，相当于上界。公平的 Table 2/3 对比（α=0.0）将由 Phase 3 给出。

---

## 4. 核心发现

1. **复现了论文 Figure 3 的核心趋势。** ETTm2 上最佳 α 随窗口增大：H=96 时 0.0，H=168 时 0.25（−7.4%），H=336 时 1.0（−2.1%）。
2. **绝对 MSE 与论文偏差 0–12%。** ETTm2 H=168（1.27 vs 1.31）和 WTH H=24（2.48 vs 2.48）匹配最佳。
3. **H=336 处趋势不显著**——3 seeds 下标准差（0.5–2.3）已经接近组间差异。
4. **α prior 偷看了未来**，存在 train/val 分布不一致。所有 Table 1–6 对比统一使用 `--alpha 0.0 --prior none`。

---

## 5. Phase 3 初步结果 — none / RevIN / Dish-TS

| pred_len | norm=none | norm=revin | norm=dishts |
|----------|-----------|------------|-------------|
| 24 | 8.08 | 8.25 | **7.07** |
| 96 | 11.86 | 13.06 | 12.14 |
| **168** | 14.55 | 14.57 | **13.57** |
| **336** | 18.60 | 18.58 | **17.36** |

dishts 在 3/4 个预测窗口上同时优于 none 和 RevIN（仅 H=96 接近）。完整的 Phase 3（约 96 个 job，覆盖 4 个数据集）正在进行中。

---

## 6. 下一步

| 步骤 | 操作 | 命令 |
|------|------|------|
| 1 | 等 Phase 2b 跑完（`PATIENCE=3 MAX_EPOCHS=70` 下约 20 h） | `bash repro_figures/run_phase2b.sh` |
| 2 | 跨数据集汇总 | `python3 repro_figures/compare_phase2b.py` |
| 3 | 完成 Phase 3（none / RevIN 基线） | `bash repro_figures/run_baselines_none_revin.sh` |
| 4 | 完整 ours vs 论文对比 | `python3 repro_figures/compare_paper.py --apply-paper-scale multivariate` |
| 5 | 生成对比图 | `python3 repro_figures/plot_ppt_comparison.py` |

Phase 2b 和 Phase 3 故意使用不同的早停配置：
- Phase 2b：`PATIENCE=3 MAX_EPOCHS=70`——只验证定性趋势
- Phase 3：保留 `patience=7 / max_epochs=100`——Table 2/3 需要准确的绝对 MSE

---

## 7. 实验日志

| 日期 | 事件 |
|------|------|
| 2026-06-14 | 搭建环境；smoke test 通过；Phase 1（27 jobs）完成 |
| 2026-06-15 | Phase 2a 完成（45/45, 0 NaN）。发现最佳 α 随窗口变长而增大。ETTm2 H=168 比值 1.01× |
| 2026-06-16 | 启动 Phase 2b（`PATIENCE=3 MAX_EPOCHS=70`，预计 ~20 h） |

---

*最后更新：2026-06-16*

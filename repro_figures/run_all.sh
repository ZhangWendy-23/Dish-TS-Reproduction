#!/usr/bin/env bash
#
# One-command launcher for the full Dish-TS reproduction battery.
#
# Run it on the 3090 box:
#     cd ~/Dish-TS
#     git pull
#     bash repro_figures/run_all.sh
#
# It runs four ablation sweeps in sequence:
#     run_table2.sh   (Table 2 / 3: Dish-TS vs baselines, alpha=0)
#     run_table4.sh   (Table 4:     long-horizon, L=96 fixed)
#     run_table5.sh   (Table 5:     lookback, H=48 fixed)
#     run_table6.sh   (Table 6:     CONET init ablation)
#     run_figure3.sh  (Figure 3:    alpha ablation, for academic discussion)
#
# Each script appends to results/figure3_runs.csv and writes logs to logs/.
# After all sweeps finish, run:
#     python3 repro_figures/compare_paper.py --apply-paper-scale multivariate
# to produce the final paper-vs-ours comparison table.

set -u
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

echo "[run_all] working dir = $PROJ_ROOT"
echo "[run_all] GPU = ${GPU:-0}"
echo

for step in run_table2 run_table4 run_table5 run_table6 run_figure3; do
    echo
    echo "====================================================="
    echo "==  STEP: $step "
    echo "====================================================="
    bash "repro_figures/${step}.sh" || {
        echo "[run_all] FAILED in step $step (non-zero exit).  Check the log above."
        exit 1
    }
done

echo
echo "[run_all] All sweeps finished.  Produce the comparison table with:"
echo "    python3 repro_figures/compare_paper.py --apply-paper-scale multivariate"

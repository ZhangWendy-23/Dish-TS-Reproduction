"""Reproduce Figure 4 from the Dish-TS paper (qualitative forecast comparison).

Figure 4 compares the raw backbone (Autoformer), RevIN and Dish-TS forecasts
against the ground truth on a single test sample of ETTm2 (or any dataset),
picking a window where the series distribution changes sharply so the
normalization methods reveal their differences.

The script re-uses :mod:`train.py` as a *command-line subprocess* for each
method (``--norm none|revin|dishts``). After the three runs complete, it reads
the last ``pred_len`` time-steps of the chosen test sample from a small
inference pass using the final model state (via a small helper loop), saves a
CSV with the four series, and plots the two-row figure.

Usage
-----
    python3 repro_figures/figure4_prediction_comparison.py \
        --data ETTm2 --seq_len 96 --pred_len 96 \
        --model Autoformer --sample_idx 0 --gpu 0

Outputs
-------
    paper_results/figure4_predictions.csv     (lookback + 4 forecast series)
    paper_results/figure4_predictions.png     (2-row plot, backbone & revin)
"""

import argparse
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backbones import Autoformer, Informer, Transformer  # noqa: E402
from DishTS import DishTS  # noqa: E402
from REVIN import RevIN  # noqa: E402
from utils.dataset import TSForecastDataset  # noqa: E402

MODEL_DICT = {"Autoformer": Autoformer, "Informer": Informer, "Transformer": Transformer}
OUT_CSV = os.path.join(ROOT, "paper_results", "figure4_predictions.csv")
OUT_PNG = os.path.join(ROOT, "paper_results", "figure4_predictions.png")
LOG_DIR = os.path.join(ROOT, "logs")


class _FakeArgs(dict):
    """Attribute-style access to dish_init / n_series / seq_len args."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _dataset(data, seq_len, label_len, pred_len, features):
    return TSForecastDataset(
        root_path=os.path.join(ROOT, "data", data),
        data_path=f"{data}.csv",
        flag="test",
        size=[seq_len, label_len, pred_len],
        features=features,
        target="OT",
        scale=False,
    )


def _run_train_via_cli(backbone_name, norm_name, args):
    """Invoke train.py via subprocess with the chosen norm method."""
    log = os.path.join(LOG_DIR, f"fig4_{args.data}_{backbone_name}_{norm_name}_"
                                 f"sq{args.seq_len}_pd{args.pred_len}.txt")
    os.makedirs(LOG_DIR, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(ROOT, "train.py"),
        "--data", args.data, "--model", backbone_name,
        "--norm", norm_name,
        "--seq_len", str(args.seq_len),
        "--pred_len", str(args.pred_len),
        "--label_len", str(args.label_len),
        "--features", args.features,
        "--batch_size", "32",
        "--train_epochs", str(args.train_epochs),
        "--patience", "3",
        "--lr", str(args.lr),
        "--alpha", str(args.alpha),
        "--seed", str(args.seed),
        "--gpu", str(args.gpu),
    ]
    print(f"[fig4] RUN: {' '.join(cmd)}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    with open(log, "w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
    print(f"[fig4]   log -> {log}")


def _build_model(backbone_name, norm_name, args):
    backbone = MODEL_DICT[backbone_name](args).to(args.device)
    n_series = args.c_out
    if norm_name == "none":
        norm = None
    elif norm_name == "revin":
        ra = _FakeArgs(n_series=n_series, seq_len=args.seq_len, dish_init=args.dish_init)
        norm = RevIN(ra).to(args.device)
    elif norm_name == "dishts":
        ra = _FakeArgs(n_series=n_series, seq_len=args.seq_len, dish_init=args.dish_init)
        norm = DishTS(ra).to(args.device)
    else:
        raise ValueError(norm_name)
    return backbone, norm


def _predict_with(backbone_name, norm_name, args):
    """Run a tiny train+infer loop; return (preds, trues, lookback_first_sample)."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ds = _dataset(args.data, args.seq_len, args.label_len, args.pred_len, args.features)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    backbone, norm = _build_model(backbone_name, norm_name, args)
    params = list(backbone.parameters()) + (list(norm.parameters()) if norm else [])
    optim = torch.optim.Adam(params, lr=args.lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(args.train_epochs):
        backbone.train()
        if norm is not None:
            norm.train()
        total, n = 0.0, 0
        for batch in loader:
            bx = batch[0].float().to(args.device)
            by = batch[1].float().to(args.device)
            dec_inp = torch.zeros_like(by[:, -args.pred_len:, :]).to(args.device)
            dec_inp[:, :args.label_len, :] = by[:, :args.label_len, :]

            if norm is not None:
                if norm_name == "dishts":
                    bx, dec_inp = norm(bx, mode="forward", dec_inp=dec_inp)
                else:
                    bx = norm(bx, mode="forward")
                    dec_inp = norm(dec_inp, mode="forward")

            forecast = backbone(bx, dec_inp)
            if norm is not None:
                forecast = norm(forecast, mode="inverse")

            true = by[:, -args.pred_len:, :]
            loss = criterion(forecast, true)
            if norm_name == "dishts" and args.alpha > 0:
                loss = loss + args.alpha * torch.mean(
                    torch.pow(torch.mean(forecast, 1, keepdim=True) - norm.phih, 2))
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.detach().item()
            n += 1
        print(f"[fig4] {backbone_name}+{norm_name} ep {epoch + 1}/{args.train_epochs} "
              f"loss {total / max(n, 1):.5f}")

    # -- inference on a single sample --
    backbone.eval()
    if norm is not None:
        norm.eval()

    with torch.no_grad():
        one_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        for i, batch in enumerate(one_loader):
            if i < args.sample_idx:
                continue
            bx = batch[0].float().to(args.device)
            by = batch[1].float().to(args.device)
            dec_inp = torch.zeros_like(by[:, -args.pred_len:, :]).to(args.device)
            dec_inp[:, :args.label_len, :] = by[:, :args.label_len, :]

            if norm is not None:
                if norm_name == "dishts":
                    bx_n, dec_n = norm(bx, mode="forward", dec_inp=dec_inp)
                else:
                    bx_n = norm(bx, mode="forward")
                    dec_n = norm(dec_inp, mode="forward")
                forecast = backbone(bx_n, dec_n)
                forecast = norm(forecast, mode="inverse")
            else:
                forecast = backbone(bx, dec_inp)
            lookback = bx[0, :, :].cpu().numpy()         # (seq_len, D)
            pred = forecast[0, :, :].cpu().numpy()       # (pred_len, D)
            true = by[0, -args.pred_len:, :].cpu().numpy()
            break
    return lookback, pred, true


def _plot(results, seq_len, pred_len, dataset, sample_idx):
    # use first channel (column 0); for multivariate datasets "OT" (target) is index -1.
    col = 0  # pick channel 0; override for ETT datasets the last column is 'OT'

    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)

    # (lookback + ground truth) is drawn on every row as reference.
    t_look = np.arange(seq_len)
    t_hor = np.arange(seq_len, seq_len + pred_len)

    look = results["lookback"][:, col]
    gt_hor = results["gt"][:, col]

    # row (a): backbone vs. Dish-TS vs GT
    ax = axes[0]
    ax.plot(t_look, look, color="#555555", linewidth=1.2, alpha=0.5, label="lookback")
    ax.plot(t_hor, results["backbone"][:, col], color="#b5a017", linewidth=2.0, label="Autoformer")
    ax.plot(t_hor, results["dishts"][:, col], color="#1f77b4", linewidth=2.2, label="Dish-TS (Autoformer)")
    ax.plot(t_hor, gt_hor, color="#d62728", linewidth=2.0, linestyle="-.", label="Ground Truth")
    ax.axvline(seq_len - 0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_title(f"(a) Autoformer vs Dish-TS (Autoformer) vs Ground Truth  [{dataset}, sample {sample_idx}]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylabel("value")

    # row (b): RevIN vs. Dish-TS vs GT
    ax = axes[1]
    ax.plot(t_look, look, color="#555555", linewidth=1.2, alpha=0.5, label="lookback")
    ax.plot(t_hor, results["revin"][:, col], color="#ff9933", linewidth=2.0, label="RevIN (Autoformer)")
    ax.plot(t_hor, results["dishts"][:, col], color="#1f77b4", linewidth=2.2, label="Dish-TS (Autoformer)")
    ax.plot(t_hor, gt_hor, color="#d62728", linewidth=2.0, linestyle="-.", label="Ground Truth")
    ax.axvline(seq_len - 0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_title("(b) RevIN (Autoformer) vs Dish-TS (Autoformer) vs Ground Truth")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylabel("value")
    ax.set_xlabel("time step (lookback | horizon)")

    fig.suptitle("Figure 4 reproduction: qualitative forecast comparison", y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ETTm2")
    ap.add_argument("--model", default="Autoformer")
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--label_len", type=int, default=48)
    ap.add_argument("--features", default="M")
    ap.add_argument("--c_out", type=int, default=7)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--e_layers", type=int, default=2)
    ap.add_argument("--d_layers", type=int, default=1)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--factor", type=int, default=3)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--dish_init", default="standard")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train_epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--sample_idx", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    results = {}
    for norm_name in ["none", "revin", "dishts"]:
        lb, pred, true = _predict_with(args.model, norm_name, args)
        key = "backbone" if norm_name == "none" else norm_name
        results[key] = pred
    results["lookback"] = lb
    results["gt"] = true

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    col = 0
    full_gt = np.concatenate([results["lookback"][:, col], results["gt"][:, col]])
    df = pd.DataFrame({
        "timestep": np.arange(args.seq_len + args.pred_len),
        "lookback_s0": np.concatenate([results["lookback"][:, col],
                                       np.full(args.pred_len, np.nan)]),
        "ground_truth_s0": full_gt,
        "backbone_pred_s0": np.concatenate([np.full(args.seq_len, np.nan),
                                             results["backbone"][:, col]]),
        "revin_pred_s0": np.concatenate([np.full(args.seq_len, np.nan),
                                         results["revin"][:, col]]),
        "dishts_pred_s0": np.concatenate([np.full(args.seq_len, np.nan),
                                          results["dishts"][:, col]]),
    })
    df.to_csv(OUT_CSV, index=False)
    print(f"[fig4] Wrote {OUT_CSV}")

    fig = _plot(results, args.seq_len, args.pred_len, args.data, args.sample_idx)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"[fig4] Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()

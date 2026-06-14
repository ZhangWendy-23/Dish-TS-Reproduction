"""
Dish-TS Training Script (AAAI 2023)
=====================================
Reproduction of "Dish-TS: A General Paradigm for Alleviating
Distribution Shift in Time Series Forecasting".

Key features:
- Dual-CONET normalization (BackCoNet + HoriCoNet) via DishTS module
- Optional prior-loss variants for horizon-level CONET (disabled by default to match the
  official Dish-TS repo, which does NOT use any prior loss):
    L = MSE                                        (prior=none, default)
    L = MSE + alpha * (phih - true_horizon_mean)^2 (prior=paper)
    L = MSE + alpha * (pred_mean - phih)^2        (prior=legacy)
- No global data normalization by default - operates on raw values

Datasets (place CSV files under ./data/):
  - ETTm2.csv  : Electricity Transformer Temperature, 15-minutely
  - ETTh1.csv  : Electricity Transformer Temperature, hourly
  - ECL.csv    : Electricity consumption, hourly (321 clients)
  - WTH.csv    : Weather, 10-minutely (21 features)
  - ILI.csv    : Influenza-like illness, weekly

CSV format: date column + feature columns (no index column)

Usage (paper settings):
  python train.py --data ETTm2 --model Autoformer --norm dishts --batch_size 128 --gpu 0
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import argparse
import copy
from utils import setup_seed
from utils.dataset import TSForecastDataset
from utils.earlystop import EarlyStopping
from utils.metric import get_metrics
from backbones import Autoformer, Informer, Transformer, NBEATS
from DishTS import DishTS
from REVIN import RevIN
from Model import Model


def update_args_from_model_params(args, n_series):
    model_params = {
        "embed_type":3,'factor':3,"output_attention":False,'d_model':512,'embed':'timeF','freq':'h',# Informer
        'dropout':0.05, 'n_heads':8,'d_ff':2048, 'moving_avg':25,'activation':'gelu','e_layers':2, # Autoformer
        'd_layers':1, 'distil':True, "enc_in":n_series,  "dec_in":n_series, 'c_out':n_series, 'n_series':n_series,
        }
    model_params.update(vars(args))
    args = argparse.Namespace(**model_params)
    return args


parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--output_file', type=str, default='forecast.csv')
# forecast — paper default: L = H (lookback == horizon), see Section 4.
# IMPORTANT: seq_len=0 is a special sentinel meaning "use pred_len" (i.e. the
# paper's "常规实验 L=H").  You can still set seq_len explicitly for ablation
# experiments like Table 4 (L fixed, H grows) and Table 5 (H fixed, L grows).
parser.add_argument('--seq_len', type=int, default=0,
                    help='Lookback length. Paper default for Tables 1-3 is L=H, '
                         'i.e. --seq_len 0 (auto-sets seq_len=pred_len).')
# label_len: paper default is 48 (decoder warm-start length). 0 means "auto"
# (clamped to min(seq_len, pred_len), with lower bound 48 — this is the
# legacy Dish-TS default; passing 0 is therefore equivalent to the paper).
parser.add_argument('--label_len', type=int, default=0,
                    help='Decoder warm-start length. Paper: 48. '
                         '0 = auto (clamped to min(seq_len, pred_len), lower bound 48).')
parser.add_argument('--pred_len', type=int, default=96)
# data
parser.add_argument('--data', type=str, default='ETTm2')
parser.add_argument('--features', type=str, default='M',
                    help='"M" = multivariate (paper default for Table 2/3), '
                         '"S" = univariate (Table 1 ablation).')
# forecast model — paper evaluates Informer, Autoformer, N-BEATS.  We default
# to Autoformer because that is the backbone used in the paper's "RevIN vs
# Dish-TS" comparative figures (Table 3).
parser.add_argument('--model', type=str, default='Autoformer',
                    help='Forecast backbone. Paper uses Autoformer (Table 2/3), Informer, and NBEATS (Tables 4/5).')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Paper settings (explicit): Informer=256, Autoformer=128, Transformer=128; ECL=64 for all models because of its 321 series; reduce to 64 when pred_len>168 to avoid OOM on 12GB GPUs. 0=use heuristic auto-selection (legacy).')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate. Paper: Adam optimizer with lr=1e-3')
parser.add_argument('--patience', type=int, default=7,
                    help='Early stopping patience. Paper default is 7.')
parser.add_argument('--train_epochs', type=int, default=100,
                    help='Maximum training epochs. Paper: 100.')
parser.add_argument('--gpu', type=int, default=0)
# shift / normalization model
parser.add_argument('--norm', type=str, default='none')  # none, revin, dishts
parser.add_argument('--affine', type=int, default=1)     # revin: use affine (1=yes, 0=no)
# CONET initialization — the paper Table 6 ablation uses names 'avg',
# 'norm' (standard normal) and 'uni' (uniform U[0,1]).  We accept those names
# as primary keys and additionally keep 'standard' and 'uniform' as backwards
# compatible aliases so older scripts still work.
parser.add_argument('--dish_init', type=str, default='avg',
                    choices=['standard', 'avg', 'uniform', 'norm', 'uni'])
# Prior loss (alpha) — disabled by default to match the OFFICIAL Dish-TS repo.
#
#   --prior none     (default): L = MSE only  -->  reproduce official Dish-TS baseline
#   --prior paper            : L = MSE + alpha * (phih - true_horizon_mean)^2
#                              (paper-style "horizon-level supervision")
#   --prior legacy           : L = MSE + alpha * (pred_mean - phih)^2
#                              (older self-consistency variant used in early
#                               reproductions; in practice this tends to PUSH
#                               forecasts toward the unreliable phih, so alpha>0
#                               usually INCREASES MSE — useful to verify effect).
parser.add_argument('--alpha', type=float, default=0.0,
                    help='Prior-knowledge guidance weight for Dish-TS. '
                         'Default 0.0 (= official repo: no prior loss).')
parser.add_argument('--prior', type=str, default='none',
                    choices=['none', 'paper', 'paper-phi-only', 'legacy'],
                    help='Which form of prior loss to use when alpha>0. '
                         "'none' disables prior loss (official Dish-TS repo behaviour); "
                         "'paper' supervises HoriCoNet with the true horizon mean "
                         "(matches the paper text verbatim); 'paper-phi-only' is a "
                         "refinement that keeps prior gradient on the HoriCoNet "
                         "channel alone (BackCoNet trained purely by MSE), which is "
                         "the implementation closest to the paper's design intent; "
                         "'legacy' pushes forecast mean towards HoriCoNet estimate "
                         "(harmful in practice, kept for reproducibility).")
# --scale enables / disables global z-score preprocessing.
#
# Paper README: "we directly take the original data for training/evaluation ...
# and do NOT use preprocessing techniques (e.g., z-score normalization, min-max
# normalization) to process time series dataset."  Therefore we default to
# scale=False; pass --scale explicitly if you want a pre-standardised comparison.
parser.add_argument('--scale', action='store_true',
                    help='Enable z-score StandardScaler on the data. '
                         'Without this flag the model operates on raw values, '
                         'matching the paper default.')
parser.add_argument('--figure3', action='store_true',
                    help='Append a row to results/figure3_runs.csv (used by '
                         'alpha sensitivity sweep scripts). Off by default so '
                         'standard Table 2/3 experiments do not pollute figure3.')
args = parser.parse_args()


# prepare parameters
setup_seed(args.seed)
DATA = args.data; GPU = args.gpu; MODEL = args.model; T=False
device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')

# === Paper-specified batch sizes (Dish-TS paper "Implementation details")
# Informer: 256, Autoformer: 128, Transformer: 128
# Special case: Electricity (ECL) -> 64 for all models (many series x long sequences)
# Long horizon (pred_len > 168): reduced to avoid OOM — safely capped at 64 for 12GB GPU
# === Paper's "lookback == horizon" convention (Tables 1-3) ===
# If the user passes --seq_len 0 we auto-set lookback = prediction length.
# This is applied BEFORE the label_len auto-clamp so label_len sees the
# real lookback value (not 0).
if args.seq_len == 0:
    args.seq_len = args.pred_len

# === Paper default: label_len = 48 (decoder warm-start length) ===
# Clamp to <= seq_len and <= pred_len to guarantee the decoder warm-start
# slice fits inside both the lookback window and the forecast window.
if args.label_len == 0:
    upper = min(args.seq_len, args.pred_len)
    args.label_len = min(upper, max(48, args.pred_len // 2))

if args.batch_size == 0:
    # legacy auto-selection — kept for backward compatibility only;
    # new runs should pass --batch_size explicitly.
    if DATA == 'ECL':
        args.batch_size = 64
    elif MODEL == 'Informer':
        args.batch_size = 256
    elif MODEL == 'NBEATS':
        args.batch_size = 1024
    else:  # Autoformer, Transformer
        args.batch_size = 128
# Safety cap for long prediction horizons on 12GB GPU
if args.pred_len > 168:
    args.batch_size = min(args.batch_size, 64)


# prepare dataset
# === Paper-specified data splits ===
# ETT datasets (ETTm2, ETTh1, ETTh2, ETTm1): train/val/test = 6:2:2
# Electricity, Weather, Illness: train/val/test = 7:1:2
# Reference: Dish-TS paper Section 5.1 "Experimental Setup"
if DATA in ('ETTm2', 'ETTh1', 'ETTh2', 'ETTm1', 'ILI'):
    val_ratio, test_ratio = 0.2, 0.2
else:
    val_ratio, test_ratio = 0.1, 0.2
train_dataset = TSForecastDataset(data_path=f'./data/{DATA}.csv', flag='train', size=(args.seq_len, args.label_len, args.pred_len), split=(val_ratio, test_ratio), features=args.features, scale_data=args.scale)
# Share the StandardScaler fitted on train across val / test so all splits
# live in the same Z-normalised space (see utils/dataset.py docstring).
# (Fitted scaler is None when --scale is not passed.)
_scaler = getattr(train_dataset, '_scaler', None)
val_dataset = TSForecastDataset(data_path=f'./data/{DATA}.csv', flag='val', size=(args.seq_len, args.label_len, args.pred_len), split=(val_ratio, test_ratio), features=args.features, scaler=_scaler)
test_dataset = TSForecastDataset(data_path=f'./data/{DATA}.csv', flag='test', size=(args.seq_len, args.label_len, args.pred_len), split=(val_ratio, test_ratio), features=args.features, scaler=_scaler)

# set forecast dataloader
# Performance: num_workers=0 avoids DataLoader multiprocessing crashes on Python 3.12
# The GPU is the bottleneck on RTX 3090, so single-process loading does not slow training
_loader_kwargs = dict(batch_size=args.batch_size, num_workers=0, pin_memory=True)
train_loader = DataLoader(dataset=train_dataset, shuffle=True,  **_loader_kwargs)
val_loader   = DataLoader(dataset=val_dataset,   shuffle=False, **_loader_kwargs)
test_loader  = DataLoader(dataset=test_dataset,  shuffle=False, **_loader_kwargs)

n_series = train_dataset.N
args = update_args_from_model_params(args, n_series)
# set forecast models
model_dict = {'Autoformer': Autoformer, 'Transformer': Transformer, 'Informer': Informer, 'NBEATS': NBEATS}
# set norm models
norm_dict = {'revin': RevIN, 'dishts': DishTS}
forecast_model = model_dict[MODEL].Model(args)
norm_model = None if args.norm == 'none' else norm_dict[args.norm](args)


unify_model = Model(args, forecast_model, norm_model).to(device)
# PyTorch 2.x JIT compilation fuses small ops and reduces kernel-launch overhead,
# giving ~1.3x-2x speedup for Autoformer/Informer at no accuracy cost.
# torch.compile is NOT supported on Python 3.12+, so guard it.
import sys as _sys
_py_major, _py_minor = _sys.version_info[:2]
_torch_major = int(torch.__version__.split('.')[0])
if (_py_major == 3 and _py_minor < 12) and (_torch_major >= 2):
    unify_model = torch.compile(unify_model)
optimizer = optim.Adam(unify_model.parameters(), lr=args.lr)
early_stopping = EarlyStopping(patience=args.patience, verbose=True, dump=False)
loss_fn = nn.MSELoss()


def add_prior_loss(loss, forecast, phih, target_horizon, alpha, prior, nm=None):
    """Optional prior term for Dish-TS.

    The OFFICIAL Dish-TS repo does NOT add a prior loss at all.  We therefore
    default to prior='none'.  The two other variants are provided for
    ablation studies (they are NOT paper defaults):

      - 'paper' : aligns the horizon-level estimate phih against the TRUE
          horizon-window mean.  This is the closest reading of "use prediction
          window mean as prior soft-label" from the paper text, and is what
          alpha-sweep figures show (monotonically better results as alpha grows).
      - 'legacy': aligns the model forecast MEAN against phih.  Empirically
          this pushes forecasts toward whatever phih predicts, which is itself
          derived from the lookback window; alpha>0 therefore usually INCREASES
          test MSE on most datasets.  Kept only for reproducibility of earlier
          self-consistency experiments.

    Parameters
    ----------
    loss : base MSE loss (already computed over the prediction horizon).
    forecast : [B, H, D] tensor of forecasts (only used by 'legacy' prior).
    phih : [B, 1, D] horizon-level tensor produced by HoriCoNet.
    target_horizon : [B, H, D] true-value tensor for the forecast window.
    alpha : weight. 0.0 disables the prior entirely regardless of `prior`.
    prior : 'none' | 'paper' | 'paper-phi-only' | 'legacy'
    nm    : the DishTS module that owns `phih`. Required by 'paper-phi-only',
            so that the prior gradient is restricted to the HORICONET channel
            (reduce_mlayer[:, :, 1]) while leaving the BACKCONET channel
            (reduce_mlayer[:, :, 0]) trained purely by the base MSE loss.

    Returns
    -------
    loss + (alpha * prior_term)

    Notes on 'paper-phi-only'
    --------------------------
    The official Dish-TS module stores the BackCoNet and HoriCoNet coefficients
    in a single trainable Parameter of shape [n_series, lookback, 2].  A plain
    `alpha * (phih - true_mean)^2` loss therefore sends gradient to both
    channels, which couples the two levels and is the likely reason our earlier
    runs saw short-window MSE go *up* as alpha grows.

    To recover the paper's intended behaviour ("HoriCoNet only is supervised by
    the horizon mean"), we compute phi_h again from a VIEW of reduce_mlayer
    where the BackCoNet column is detached (`.detach()`).  The resulting tensor
    is differentiable only through the HoriCoNet column, so the prior term
    supervises HoriCoNet only.
    """
    if prior == 'none' or alpha <= 0 or phih is None:
        return loss

    if prior == 'paper-phi-only':
        # Re-derive phi_h while keeping BackCoNet channel frozen.
        assert nm is not None, (
            "prior='paper-phi-only' requires the DishTS module passed in."
        )
        rm = nm.reduce_mlayer  # [D, L, 2]
        # Only the HoriCoNet column carries gradients for the prior term.
        # This exactly mirrors the paper's verbal description: "the horizon
        # level estimate phi_h is guided by the true horizon mean; the
        # BackCoNet level phi_l is trained only by the base MSE".
        rm_prior = rm.detach().clone()
        rm_prior[:, :, 1] = rm[:, :, 1]  # Hori channel: live gradient
        x_transpose = nm._last_batch_x.permute(2, 0, 1)  # [D, B, L]
        theta = torch.bmm(x_transpose, rm_prior).permute(1, 2, 0)
        if nm.activate:
            theta = torch.nn.functional.gelu(theta)
        phih_prior = theta[:, 1:, :]  # [B, 1, D] for each horizon dim

        true_mean = torch.mean(target_horizon, dim=1, keepdim=True)  # [B, 1, D]
        prior_term = torch.mean(torch.pow(phih_prior - true_mean, 2))
        return loss + alpha * prior_term

    if prior == 'paper':
        # L_prior = alpha * mean( (phih - mean(truth_horizon))^2 )
        true_mean = torch.mean(target_horizon, dim=1, keepdim=True)  # [B,1,D]
        prior_term = torch.mean(torch.pow(phih - true_mean, 2))
    elif prior == 'legacy':
        # Old self-consistency form: L_prior = alpha * mean( (mean(pred)-phih)^2 )
        pred_mean = torch.mean(forecast, dim=1, keepdim=True)
        prior_term = torch.mean(torch.pow(pred_mean - phih, 2))
    else:
        raise ValueError(f'Unknown --prior={prior}')
    return loss + alpha * prior_term


def get_init_batch(batch):
    batch_x, batch_y = batch
    # non_blocking=True allows H2D transfer to overlap with CPU compute
    # (requires pin_memory=True in DataLoader); ignored if device is cpu
    batch_x = batch_x.to(device, non_blocking=True).float()
    batch_y = batch_y.to(device, non_blocking=True).float()
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :])
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)
    return batch_x, batch_y, dec_inp


max_epochs = args.train_epochs
for epoch in range(max_epochs):
    # --- train -----------------------------------------------------------
    train_loss_sum = torch.zeros(1, device=device)
    train_n = 0
    unify_model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        batch_x, batch_y, dec_inp = get_init_batch(batch)
        forecast = unify_model(batch_x, dec_inp)
        target_horizon = batch_y[:, -args.pred_len:, :]
        loss = loss_fn(forecast, target_horizon)
        # Optional Dish-TS prior loss.  The phih tensor is only available when
        # norm='dishts' (it is written during the DishTS.forward 'forward' pass).
        nm = getattr(unify_model, 'nm', None)
        phih = getattr(nm, 'phih', None)
        loss = add_prior_loss(loss, forecast, phih, target_horizon,
                              alpha=args.alpha, prior=args.prior, nm=nm)
        train_loss_sum = train_loss_sum + loss.detach()
        train_n = train_n + 1
        loss.backward()
        optimizer.step()
    # --- validate --------------------------------------------------------
    val_loss_sum = torch.zeros(1, device=device)
    val_n = 0
    with torch.no_grad():
        unify_model.eval()
        for batch in val_loader:
            batch_x, batch_y, dec_inp = get_init_batch(batch)
            forecast = unify_model(batch_x, dec_inp)
            # For early-stopping we evaluate the plain MSE in the same space
            # as test metrics.  Prior loss is a training regulariser only and
            # is intentionally NOT evaluated here.
            loss = loss_fn(forecast, batch_y[:, -args.pred_len:, :])
            val_loss_sum = val_loss_sum + loss.detach()
            val_n = val_n + 1
    train_loss_avg = (train_loss_sum / train_n).item()
    val_loss_avg = (val_loss_sum / val_n).item()
    # early stop
    print('epoch:{0:}, train_loss:{1:.5f}, val_loss:{2:.5f}'.format(epoch, train_loss_avg, val_loss_avg))
    early_stopping(val_loss_avg, unify_model, epoch)
    if early_stopping.early_stop:
        print("Early stopping with best_score:{}".format(-early_stopping.best_score))
        break
    if np.isnan(val_loss_avg) or np.isnan(train_loss_avg):
        break

# test
model = early_stopping.best_model
model.eval()
preds, trues = None, None
with torch.no_grad():
    for batch in test_loader:
        batch_x, batch_y, dec_inp = get_init_batch(batch)
        forecast = model(batch_x, dec_inp)
        # concat
        pred = forecast.detach().cpu().numpy()
        true = batch_y[:, -args.pred_len:, :].detach().cpu().numpy()
        preds = pred if preds is None else np.concatenate((preds, pred), 0)
        trues = true if trues is None else np.concatenate((trues, true), 0)


mae, mse, rmse, mape, mspe = get_metrics(preds, trues)

# --- Per-column MSE for magnitude analysis ---------------------------------
# The paper reports a single MSE averaged over all predicted columns.  When
# reproducing, however, a single out-of-scale feature (e.g. an un-normalized
# sensor on ECL / WTH) can dominate the total and make the result look
# orders-of-magnitude different from the paper.  Printing each column's MSE
# lets us spot such features instantly.
per_col_mse = np.mean((preds - trues) ** 2, axis=(0, 1))  # shape (D,)
per_col_str = " ".join(f"{float(v):.3f}" for v in per_col_mse)
print(f"[train] per-column MSE: {per_col_str}")
# ---------------------------------------------------------------------------

print("=" * 80)
print(f"DATA={DATA}  MODEL={MODEL}  NORM={args.norm}  SEED={args.seed}  "
      f"seq_len={args.seq_len}  label_len={args.label_len}  pred_len={args.pred_len}  "
      f"batch_size={args.batch_size}  lr={args.lr}  "
      f"alpha={args.alpha}  prior={args.prior}  "
      f"dish_init={args.dish_init}  scale={args.scale}  patience={args.patience}")
print(f"  MSE={mse:.6f}  MAE={mae:.6f}  RMSE={rmse:.6f}  MAPE={mape:.6f}  MSPE={mspe:.6f}")
print("=" * 80)

df = pd.DataFrame([[DATA, MODEL, args.norm, args.seed,
                     args.seq_len, args.label_len, args.pred_len,
                     args.batch_size, args.alpha,
                     args.features,
                     mse, mae, rmse, mape, mspe]],
                   columns=['data', 'model', 'norm', 'seed',
                            'seq_len', 'label_len', 'pred_len',
                            'batch_size', 'alpha',
                            'features',
                            'MSE', 'MAE', 'RMSE', 'MAPE', 'MSPE'])
# defensive: make sure the column list matches the data list exactly
assert len(df.columns) == len(df.values[0]), (
    f"Column/value mismatch: {len(df.columns)} columns vs "
    f"{len(df.values[0])} values"
)
print(df.to_string(index=False))

# ---------------------------------------------------------------------------
# Extra artifacts for reproducing Figure 3 and Figure 4.
#   Figure 3: one row per run, appended to results/figure3_runs.csv
#   Figure 4: the FIRST test sample's lookback + forecast + ground-truth saved
#             to results/figures/figure4_<RUN_ID>.csv
#
# These are ONLY written when --figure3 is passed (used by alpha-sweep / Figure
# reproduction scripts).  Normal Table 2/3 experiments skip this to avoid
# polluting figure3_runs.csv.
# ---------------------------------------------------------------------------
if args.figure3:
    import os
    import datetime

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(ROOT_DIR, "results")
    FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Figure 3: cumulative CSV of (dataset, seq_len, pred_len, model, norm, alpha, MSE, ...)
    f3_csv = os.path.join(RESULTS_DIR, "figure3_runs.csv")
    # The "alpha" column only has a physical meaning under norm == 'dishts'.
    # For 'none' / 'revin' we write 0.0 so aggregate scripts (which do pivot
    # over the alpha column) don't accidentally group under 0.5 / the
    # dishts default.
    reported_alpha = args.alpha if args.norm == 'dishts' else 0.0
    f3_values = [DATA, args.seq_len, args.pred_len, MODEL, args.norm,
                 reported_alpha, args.seed, mse, mae, rmse, mape, mspe,
                 args.dish_init, args.scale, args.prior, args.features, timestamp]
    f3_columns = ["dataset", "seq_len", "pred_len", "model", "norm",
                  "alpha", "seed", "MSE", "MAE", "RMSE", "MAPE", "MSPE",
                  "dish_init", "scaled", "prior", "features", "timestamp"]
    assert len(f3_values) == len(f3_columns), (
        f"figure3_runs: {len(f3_values)} values vs {len(f3_columns)} columns"
    )
    f3_df = pd.DataFrame([f3_values], columns=f3_columns)
    if os.path.exists(f3_csv):
        # --- header check: don't silently append to a CSV produced by a
        # different train.py version (different column set → impossible to
        # compare rows).
        with open(f3_csv, "r", encoding="utf-8") as fh:
            existing_header = [c.strip() for c in fh.readline().split(",")]
        if existing_header != f3_columns:
            raise RuntimeError(
                f"Refusing to append to {f3_csv}: existing CSV uses a "
                f"different header than the current code.  Expected "
                f"{len(f3_columns)} columns {f3_columns}; got "
                f"{len(existing_header)} columns {existing_header}.  "
                f"Run repair_figure3_csv.py once to normalize the file, or "
                f"rename it out of the way."
            )
        f3_df.to_csv(f3_csv, mode="a", header=False, index=False)
    else:
        f3_df.to_csv(f3_csv, mode="w", header=True, index=False)
    print(f"[train] appended 1 row to {f3_csv}")

    # --- Figure 4: save the FIRST test sample (lookback + prediction + ground-truth)
    if preds is not None and trues is not None:
        lookback0 = None
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                lookback0 = batch[0][0:1].detach().cpu().numpy()  # (1, seq_len, D)
                break
        if lookback0 is not None:
            lb = lookback0[0]                  # (seq_len, D)
            pred0 = preds[0]                   # (pred_len, D)
            true0 = trues[0]                   # (pred_len, D)
            full_len = args.seq_len + args.pred_len
            ground_truth = np.concatenate([lb, true0], axis=0)
            pred_series = np.concatenate([np.full_like(lb, np.nan), pred0], axis=0)

            f4_filename = (f"figure4_{DATA}_{MODEL}_{args.norm}_"
                          f"sq{args.seq_len}_pd{args.pred_len}_alpha{reported_alpha}_"
                          f"seed{args.seed}.csv")
            f4_file = os.path.join(FIGURES_DIR, f4_filename)
            out_rows = []
            for t in range(full_len):
                row = {"timestep": t, "dataset": DATA, "model": MODEL,
                       "norm": args.norm, "seq_len": args.seq_len,
                       "pred_len": args.pred_len, "alpha": reported_alpha,
                       "seed": args.seed, "dish_init": args.dish_init,
                       "scaled": args.scale, "prior": args.prior}
                for c in range(preds.shape[-1]):
                    row[f"gt_{c}"] = ground_truth[t, c]
                    row[f"pred_{c}"] = pred_series[t, c]
                out_rows.append(row)
            pd.DataFrame(out_rows).to_csv(f4_file, index=False)
            print(f"[train] wrote figure4 sample to {f4_file}")



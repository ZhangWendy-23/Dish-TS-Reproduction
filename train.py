"""
Dish-TS Training Script (AAAI 2023)
=====================================
Official reproduction of "Dish-TS: A General Paradigm for Alleviating
Distribution Shift in Time Series Forecasting".

Key features:
- Dual-CONET normalization (BackCoNet + HoriCoNet) via DishTS module
- Prior knowledge guidance loss: Loss = MSE + alpha * (mean(pred) - phih)^2
- No global data normalization - operates on raw values

Datasets (place CSV files under ./data/):
  - ETTm2.csv  : Electricity Transformer Temperature, 15-minutely
  - ETTh1.csv  : Electricity Transformer Temperature, hourly
  - ECL.csv    : Electricity consumption, hourly (321 clients)
  - WTH.csv    : Weather, 10-minutely (21 features)
  - ILI.csv    : Influenza-like illness, weekly

CSV format: date column + feature columns (no index column)

Usage (paper settings):
  python train.py --data ETTm2 --model Autoformer --norm dishts --batch_size 0 --alpha 0.5 --gpu 0
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
from backbones import Autoformer, Informer, Transformer
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
# forecast — paper default: seq_len=96, label_len=max(48, pred_len//2)
# IMPORTANT: label_len is additionally clamped to <= seq_len and <= pred_len
#            to avoid negative r_begin in __getitem__ (pred_len=336 crashes otherwise).
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=0,
                    help='Paper: label_len = max(48, pred_len//2); 0=auto (clamped to seq_len, pred_len)')
parser.add_argument('--pred_len', type=int, default=96)
# data
parser.add_argument('--data', type=str, default='ETTm2')
parser.add_argument('--features', type=str, default='M')
# forecast model
parser.add_argument('--model', type=str, default='Transformer')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Paper settings (explicit): Informer=256, Autoformer=128, Transformer=128; ECL=64 for all models because of its 321 series; reduce to 64 when pred_len>168 to avoid OOM on 12GB GPUs. 0=use heuristic auto-selection (legacy).')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate. Paper: Adam optimizer with lr=1e-3')
parser.add_argument('--patience', type=int, default=15,
                    help='Early stopping patience. Paper default is 7, but Dish-TS '
                         'sometimes takes more epochs to learn the horizon-level '
                         'CONET parameters, hence a default of 15 is safer.')
parser.add_argument('--train_epochs', type=int, default=100,
                    help='Maximum training epochs. Paper: 100.')
parser.add_argument('--gpu', type=int, default=0)
# shift / normalization model
parser.add_argument('--norm', type=str, default='none')  # none, revin, dishts
parser.add_argument('--affine', type=int, default=1)     # revin: use affine (1=yes, 0=no)
parser.add_argument('--dish_init', type=str, default='standard')  # standard, avg, uniform
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Prior knowledge guidance weight for Dish-TS. Paper: searched 0 to 1. Default=0.5')
parser.add_argument('--no-scale', action='store_true',
                    help='Disable StandardScaler on input data. Keep this OFF '
                         '(scaled) to match paper MSE scale. Use --no-scale to '
                         'reproduce old raw-scale results.')
args = parser.parse_args()


# prepare parameters
setup_seed(args.seed)
DATA = args.data; GPU = args.gpu; MODEL = args.model; T=False
device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')

# === Paper-specified batch sizes (Dish-TS paper "Implementation details")
# Informer: 256, Autoformer: 128, Transformer: 128
# Special case: Electricity (ECL) -> 64 for all models (many series x long sequences)
# Long horizon (pred_len > 168): reduced to avoid OOM — safely capped at 64 for 12GB GPU
if args.label_len == 0:
    # Paper: label_len = max(48, pred_len//2)
    #
    # Additionally we MUST clamp label_len <= min(seq_len, pred_len).  Why?
    #   1) seq_y in TSForecastDataset.__getitem__ spans
    #        [s_end - label_len, s_end - label_len + label_len + pred_len]
    #      which requires s_end - label_len >= 0  →  label_len <= s_end <= seq_len.
    #   2) dec_inp in get_init_batch below slices batch_y[:, :label_len, :];
    #      label_len must therefore not exceed the batch_y lookback region,
    #      which is at most seq_len because we pass label_len as the decoder "warm start".
    #
    # Without this clamp pred_len=336 with seq_len=96 used to produce
    #    RuntimeError: stack expects each tensor to be equal size,
    #      but got [504, 7] at entry 0 and [0, 7] at entry 18
    # because __getitem__ returned a 0-length seq_y for the last rows of the dataset.
    upper = min(args.seq_len, args.pred_len)
    args.label_len = min(upper, max(48, args.pred_len // 2))

if args.batch_size == 0:
    # legacy auto-selection — kept for backward compatibility only;
    # new runs should pass --batch_size explicitly.
    if DATA == 'ECL':
        args.batch_size = 64
    elif MODEL == 'Informer':
        args.batch_size = 256
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
train_dataset = TSForecastDataset(data_path=f'./data/{DATA}.csv', flag='train', size=(args.seq_len, args.label_len, args.pred_len), split=(val_ratio, test_ratio), features=args.features, scale_data=not args.no_scale)
# Share the StandardScaler fitted on train across val / test so all splits
# live in the same Z-normalised space (see utils/dataset.py docstring).
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
model_dict = {'Autoformer': Autoformer, 'Transformer': Transformer, 'Informer': Informer}
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
    # --- train (loss accumulated on GPU, one .item() at epoch end) ---
    train_loss_sum = torch.zeros(1, device=device)
    train_n = 0
    unify_model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        batch_x, batch_y, dec_inp = get_init_batch(batch)
        forecast = unify_model(batch_x, dec_inp)
        # base loss: MSE
        loss = loss_fn(forecast, batch_y[:, -args.pred_len:, :])
        # Paper: prior knowledge guidance for Dish-TS
        # Loss = MSE + alpha * (mean(pred) - phih)^2
        if args.norm == 'dishts' and args.alpha > 0:
            phih = unify_model.nm.phih  # [B, 1, D] - learned horizon level
            pred_mean = torch.mean(forecast, dim=1, keepdim=True)  # [B, 1, D]
            prior_loss = torch.mean(torch.pow(pred_mean - phih, 2))
            loss = loss + args.alpha * prior_loss
        train_loss_sum = train_loss_sum + loss.detach()
        train_n = train_n + 1
        loss.backward()
        optimizer.step()
    # --- validate (loss accumulated on GPU, one .item() at epoch end) ---
    val_loss_sum = torch.zeros(1, device=device)
    val_n = 0
    with torch.no_grad():
        unify_model.eval()
        for batch in val_loader:
            batch_x, batch_y, dec_inp = get_init_batch(batch)
            forecast = unify_model(batch_x, dec_inp)
            loss = loss_fn(forecast, batch_y[:, -args.pred_len:, :])
            if args.norm == 'dishts' and args.alpha > 0:
                phih = unify_model.nm.phih
                pred_mean = torch.mean(forecast, dim=1, keepdim=True)
                prior_loss = torch.mean(torch.pow(pred_mean - phih, 2))
                loss = loss + args.alpha * prior_loss
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
      f"batch_size={args.batch_size}  lr={args.lr}  alpha={args.alpha}")
print(f"  MSE={mse:.6f}  MAE={mae:.6f}  RMSE={rmse:.6f}  MAPE={mape:.6f}  MSPE={mspe:.6f}")
print("=" * 80)

df = pd.DataFrame([[DATA, MODEL, args.norm, args.seed,
                     args.seq_len, args.label_len, args.pred_len,
                     args.batch_size, args.alpha,
                     mse, mae, rmse, mape, mspe]],
                   columns=['data', 'model', 'norm', 'seed',
                            'seq_len', 'label_len', 'pred_len',
                            'batch_size', 'alpha',
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
# Later we can call repro_figures/plot_figure{3,4}.py to generate PNGs directly
# from these CSVs -- no re-training needed.
# ---------------------------------------------------------------------------
import os
import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Figure 3: cumulative CSV of (dataset, seq_len, pred_len, model, norm, alpha, MSE, ...)
f3_csv = os.path.join(RESULTS_DIR, "figure3_runs.csv")
f3_values = [DATA, args.seq_len, args.pred_len, MODEL, args.norm,
             args.alpha, args.seed, mse, mae, rmse, mape, mspe, timestamp]
f3_columns = ["dataset", "seq_len", "pred_len", "model", "norm",
              "alpha", "seed", "MSE", "MAE", "RMSE", "MAPE", "MSPE", "timestamp"]
assert len(f3_values) == len(f3_columns), (
    f"figure3_runs: {len(f3_values)} values vs {len(f3_columns)} columns"
)
f3_df = pd.DataFrame([f3_values], columns=f3_columns)
if os.path.exists(f3_csv):
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
                      f"sq{args.seq_len}_pd{args.pred_len}_alpha{args.alpha}_"
                      f"seed{args.seed}.csv")
        f4_file = os.path.join(FIGURES_DIR, f4_filename)
        out_rows = []
        for t in range(full_len):
            row = {"timestep": t, "dataset": DATA, "model": MODEL,
                   "norm": args.norm, "seq_len": args.seq_len,
                   "pred_len": args.pred_len, "alpha": args.alpha,
                   "seed": args.seed}
            for c in range(preds.shape[-1]):
                row[f"gt_{c}"] = ground_truth[t, c]
                row[f"pred_{c}"] = pred_series[t, c]
            out_rows.append(row)
        pd.DataFrame(out_rows).to_csv(f4_file, index=False)
        print(f"[train] wrote figure4 sample to {f4_file}")



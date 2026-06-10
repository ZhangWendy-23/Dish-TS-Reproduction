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
# forecast — paper default: seq_len=96, label_len=pred_len//2 (min 48)
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=0,
                    help='Paper: label_len = max(48, pred_len//2); 0=auto')
parser.add_argument('--pred_len', type=int, default=96)
# data
parser.add_argument('--data', type=str, default='ETTm2')
parser.add_argument('--features', type=str, default='M')
# forecast model
parser.add_argument('--model', type=str, default='Transformer')
parser.add_argument('--batch_size', type=int, default=0,
                    help='Paper settings: Informer=256, Autoformer=128, Transformer=128; ECL=64 for all; 0=auto')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate. Paper: Adam optimizer with lr=1e-3')
parser.add_argument('--patience', type=int, default=7,
                    help='Early stopping patience. Paper: 7')
parser.add_argument('--gpu', type=int, default=0)
# shift / normalization model
parser.add_argument('--norm', type=str, default='none')  # none, revin, dishts
parser.add_argument('--affine', type=int, default=1)     # revin: use affine (1=yes, 0=no)
parser.add_argument('--dish_init', type=str, default='standard')  # standard, avg, uniform
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Prior knowledge guidance weight for Dish-TS. Paper: searched 0 to 1. Default=0.5')
args = parser.parse_args()


# prepare parameters
setup_seed(args.seed)
DATA = args.data; GPU = args.gpu; MODEL = args.model; T=False
device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')

# === Paper-specified batch sizes (Dish-TS paper "Implementation details")
# Informer: 256, Autoformer: 128, Transformer: 128
# Special case: Electricity (ECL) → 64 for all models (many series × long sequences)
# Long horizon (pred_len > 168): reduced to avoid OOM — safely capped at 64 for 12GB GPU
if args.label_len == 0:
    # Paper: label_len = max(48, pred_len//2) for standard 96-length lookback
    # But must not exceed seq_len (e.g., Table 1: seq_len=24, pred_len=24)
    args.label_len = max(48, args.pred_len // 2)
    if args.label_len > args.seq_len:
        args.label_len = args.pred_len // 2
        print(f"[INFO] label_len capped to {args.label_len} (seq_len={args.seq_len} too short for 48)")

if args.batch_size == 0:
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
train_dataset = TSForecastDataset(data_path=f'./data/{DATA}.csv', flag='train', size=(args.seq_len, args.label_len, args.pred_len), split=(val_ratio, test_ratio), features=args.features)
val_dataset = TSForecastDataset(data_path=f'./data/{DATA}.csv', flag='val', size=(args.seq_len, args.label_len, args.pred_len), split=(val_ratio, test_ratio), features=args.features)
test_dataset = TSForecastDataset(data_path=f'./data/{DATA}.csv', flag='test', size=(args.seq_len, args.label_len, args.pred_len), split=(val_ratio, test_ratio), features=args.features)

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
    batch_x,  batch_y = batch
    # non_blocking=True allows H2D transfer to overlap with CPU compute
    # (requires pin_memory=True in DataLoader)
    batch_x = batch_x.cuda(non_blocking=True).float()
    batch_y = batch_y.cuda(non_blocking=True).float()
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :])
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)
    return batch_x,  batch_y, dec_inp


max_epochs = 100
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

print("=" * 80)
print(f"DATA={DATA}  MODEL={MODEL}  NORM={args.norm}  SEED={args.seed}  "
      f"seq_len={args.seq_len}  label_len={args.label_len}  pred_len={args.pred_len}  "
      f"batch_size={args.batch_size}  lr={args.lr}  alpha={args.alpha}")
print(f"  MSE={mse:.6f}  MAE={mae:.6f}  RMSE={rmse:.6f}  MAPE={mape:.6f}  MSPE={mspe:.6f}")
print("=" * 80)

df = pd.DataFrame([[DATA, MODEL, args.norm, args.seed, args.seq_len, args.label_len,
                    args.pred_len, args.batch_size, args.alpha, mse, mae, rmse, mape, mspe]],
                  columns=['data', 'model', 'norm', 'seed', 'seq_len', 'label_len',
                           'pred_len', 'batch_size', 'alpha', 'MSE', 'MAE', 'RMSE', 'MAPE', 'MSPE'])
print(df.to_string(index=False))



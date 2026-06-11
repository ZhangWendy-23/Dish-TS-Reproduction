"""Offline smoke test: exactly mimics the first ~170 lines of train.py but
stops before the training loop.  This checks:

- imports / argparse
- TSForecastDataset & DataLoader construction
- model / norm (Dish-TS / RevIN) init on the selected device
- the two post-training DataFrame column-count invariants
- label_len clamping / batch_size auto-selection
- .to(device) instead of .cuda() so it works on CPU-only machines too

Usage:
    python3 smoke_test.py            # CPU
    python3 smoke_test.py --gpu 0    # GPU (if available)
"""
from __future__ import annotations

import argparse
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

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
        "embed_type": 3, 'factor': 3, "output_attention": False,
        'd_model': 512, 'embed': 'timeF', 'freq': 'h',
        'dropout': 0.05, 'n_heads': 8, 'd_ff': 2048, 'moving_avg': 25,
        'activation': 'gelu', 'e_layers': 2, 'd_layers': 1,
        'distil': True, "enc_in": n_series, "dec_in": n_series,
        'c_out': n_series, 'n_series': n_series,
    }
    model_params.update(vars(args))
    return argparse.Namespace(**model_params)


def build_pipeline(args):
    setup_seed(args.seed)
    DATA = args.data; GPU = args.gpu; MODEL = args.model
    device = torch.device(
        f'cuda:{GPU}' if torch.cuda.is_available() and GPU != -1 else 'cpu')
    print(f"[smoke] device = {device}")

    if args.label_len == 0:
        upper = min(args.seq_len, args.pred_len)
        args.label_len = min(upper, max(48, args.pred_len // 2))
    if args.batch_size == 0:
        if DATA == 'ECL': args.batch_size = 64
        elif MODEL == 'Informer': args.batch_size = 256
        else: args.batch_size = 128
    if args.pred_len > 168:
        args.batch_size = min(args.batch_size, 64)

    if DATA in ('ETTm2', 'ETTh1', 'ETTh2', 'ETTm1', 'ILI'):
        val_ratio, test_ratio = 0.2, 0.2
    else:
        val_ratio, test_ratio = 0.1, 0.2

    train_dataset = TSForecastDataset(
        data_path=f'./data/{DATA}.csv', flag='train',
        size=(args.seq_len, args.label_len, args.pred_len),
        split=(val_ratio, test_ratio), features=args.features)
    train_loader = DataLoader(
        train_dataset, shuffle=True,
        batch_size=args.batch_size, num_workers=0, pin_memory=True)
    print(f"[smoke] train_loader len = {len(train_loader)}, n_series = {train_dataset.N}")

    args = update_args_from_model_params(args, train_dataset.N)
    model_dict = {'Autoformer': Autoformer, 'Transformer': Transformer, 'Informer': Informer}
    norm_dict = {'revin': RevIN, 'dishts': DishTS}
    forecast_model = model_dict[MODEL].Model(args)
    norm_model = None if args.norm == 'none' else norm_dict[args.norm](args)
    unify_model = Model(args, forecast_model, norm_model).to(device)

    print(f"[smoke] unify_model param count = "
          f"{sum(p.numel() for p in unify_model.parameters()):,}")

    # --- post-training CSV invariants ------------------------------
    mse, mae, rmse, mape, mspe = 12.36, 2.11, 3.52, 948681.87, 4.85e14
    timestamp = '20260611_000000'

    # (1) summary DataFrame
    values = [DATA, MODEL, args.norm, args.seed,
              args.seq_len, args.label_len, args.pred_len,
              args.batch_size, args.alpha,
              mse, mae, rmse, mape, mspe]
    columns = ['data', 'model', 'norm', 'seed',
               'seq_len', 'label_len', 'pred_len',
               'batch_size', 'alpha',
               'MSE', 'MAE', 'RMSE', 'MAPE', 'MSPE']
    assert len(values) == len(columns), \
        f"summary: {len(values)} values vs {len(columns)} columns"
    pd.DataFrame([values], columns=columns)

    # (2) figure3_runs row
    f3_values = [DATA, args.seq_len, args.pred_len, MODEL, args.norm,
                 args.alpha, mse, mae, rmse, mape, mspe, timestamp]
    f3_columns = ["dataset", "seq_len", "pred_len", "model", "norm",
                  "alpha", "MSE", "MAE", "RMSE", "MAPE", "MSPE", "timestamp"]
    assert len(f3_values) == len(f3_columns), \
        f"figure3_runs: {len(f3_values)} values vs {len(f3_columns)} columns"
    return device, args


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='ETTh1')
    p.add_argument('--model', default='Autoformer')
    p.add_argument('--norm', default='dishts')
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--label_len', type=int, default=0)
    p.add_argument('--pred_len', type=int, default=24)
    p.add_argument('--batch_size', type=int, default=0)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--gpu', type=int, default=-1)
    p.add_argument('--features', default='M')
    p.add_argument('--affine', type=int, default=1)
    p.add_argument('--dish_init', default='standard')
    args = p.parse_args()
    device, args = build_pipeline(args)
    print(f"[smoke] final args: data={args.data} model={args.model} "
          f"norm={args.norm} seq={args.seq_len} label={args.label_len} "
          f"pred={args.pred_len} batch_size={args.batch_size} alpha={args.alpha}")
    print("SMOKE TEST PASSED")


if __name__ == '__main__':
    main()

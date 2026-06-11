"""
Time Series Forecasting Dataset Loader
======================================
Loads CSV-formatted time series data for training/validation/testing.

Expected CSV format:
  date,feature_1,feature_2,...,feature_N
  2016-07-01 00:00:00,41.13,12.48,...,38.66
  ...

The 'date' column is automatically excluded from input features.
Data is split chronologically (no shuffle) into train/val/test sets.

StandardScaler is fitted on training data only and applied to all splits,
so that metrics (MSE/MAE) match the scale reported in the original Dish-TS
and Autoformer papers.  If scikit-learn is not installed, the scaler is
skipped with a warning (metrics will be on raw-data scale).

To disable scaling (e.g. for Dish-TS raw-value experiments), pass
``scale_data=False`` to the constructor.

Dataset sources:
  - ETTm2/ETTh1: https://github.com/zhouhaoyi/ETDataset
  - ECL/WTH/ILI: https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy
"""

import pandas as pd
import numpy as np
from torch.utils.data import Dataset

try:
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    import warnings
    warnings.warn(
        "scikit-learn is not installed.  Input data will NOT be standardized "
        "to z-score space; MSE values will be on raw-data scale and cannot be "
        "directly compared with paper tables.  Install scikit-learn to fix: "
        "pip install scikit-learn>=0.24.0"
    )


class TSForecastDataset(Dataset):
    def __init__(self, data_path='ETTh1.csv', flag='train',
                 size=(96, 48, 96), split=(0.1, 0.2),
                 features='M', scaler=None, scale_data=True):
        self.seq_len, self.label_len, self.pred_len = (
            size[0], size[1], size[2])
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.ratio_vali, self.ratio_test = split[0], split[1]
        self.features = features
        self._scaler = scaler  # shared across train/val/test
        self.scale_data = (scale_data and scaler is None
                           and _HAS_SKLEARN)  # fit only if no scaler provided
        self.__read_data__(data_path)

    def __read_data__(self, data_path):
        df_raw = pd.read_csv(data_path)
        # cols: 'M'=all variables (multivariate), 'S'=last variable only (univariate)
        cols = list(df_raw.columns)
        cols.remove('date')
        if self.features == 'S':
            cols = [cols[-1]]  # paper Table 1 univariate: forecast target series only
        # get rows
        num_vali = int(len(df_raw) * self.ratio_vali)
        num_test = int(len(df_raw) * self.ratio_test)
        num_train = len(df_raw) - num_vali - num_test
        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        left_border = border1s[self.set_type]
        right_border = border2s[self.set_type]
        # get raw data
        raw_data = df_raw[cols].values.astype(np.float32)
        self.train_data = raw_data[border1s[0]:border2s[0]]  # for scaler fitting
        # StandardScaler: fit on train only, apply to all
        if self._scaler is not None:
            raw_data = self._scaler.transform(raw_data)
        elif self.scale_data:
            self._scaler = StandardScaler()
            self._scaler.fit(self.train_data)
            raw_data = self._scaler.transform(raw_data)
        self.N = raw_data.shape[1]  # number of series
        self.data_x = raw_data[left_border:right_border]
        self.data_y = raw_data[left_border:right_border]

    def __getitem__(self, index):
        s_begin = index; s_end = s_begin + self.seq_len
        # The decoder "warm start" needs the preceding label_len time steps from
        # the same row.  r_begin must therefore be >= 0 and <= s_end.  When the
        # user passes label_len > seq_len (e.g. --pred_len 336 --seq_len 96 and
        # an overly generous auto label_len), the guard below keeps indexing sane.
        r_begin = max(0, s_end - self.label_len)
        r_end = r_begin + self.label_len + self.pred_len
        if r_end > len(self.data_x):
            raise IndexError(
                f"__getitem__({index}): r_end={r_end} exceeds data_x length "
                f"{len(self.data_x)}. seq_len={self.seq_len} label_len={self.label_len} "
                f"pred_len={self.pred_len}. Reduce pred_len/label_len or use a "
                f"larger dataset."
            )
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # Defensive check: keep slices non-empty even when label_len > seq_len
        # (otherwise stack error in the DataLoader is very hard to debug).
        if seq_x.size == 0 or seq_y.size == 0:
            raise ValueError(
                f"Empty slice at index={index} (seq_len={self.seq_len} "
                f"label_len={self.label_len} pred_len={self.pred_len}). "
                f"Ensure label_len <= min(seq_len, pred_len)."
            )
        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
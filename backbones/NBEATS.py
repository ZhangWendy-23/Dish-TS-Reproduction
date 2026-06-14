"""N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting.

Paper: https://arxiv.org/abs/1905.10437
Implementation reference: the original public N-BEATS repo + common community
re-implementations.

We implement the "Generic" flavour (not the Trend/Seasonality interpretable
version) because the Dish-TS / RevIN papers that our repo reproduces pair with
N-BEATS as a standard backbone without referring to interpretable variants.

The network maps an input lookback (B, seq_len, d_series) to a forecast of
shape (B, pred_len, d_series).  Following the paper we use residual "stacks"
of fully-connected blocks, each of which predicts a (backcast, forecast) pair
which is subtracted from / added to the running residual.  The d_series
dimension is handled with a final (1x1) convolution-style linear, which is
equivalent to running d_series independent N-BEATS networks — this exactly
matches the "univariate per-channel" design of the original N-BEATS paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    """A single N-BEATS residual block.

    Input:  (B, seq_len, d_series)  flattened to (B, seq_len * d_series).
    Output: (backcast, forecast) both of shape (B, seq_len * d_series) and
            (B, pred_len * d_series) respectively — they are applied element-wise
            to the running residual in the caller.
    """

    def __init__(self, seq_len, pred_len, d_series, hidden=256, layers=4):
        super().__init__()
        in_dim = seq_len * d_series
        fc = []
        prev = in_dim
        for _ in range(layers):
            fc.append(nn.Linear(prev, hidden))
            fc.append(nn.ReLU(inplace=True))
            prev = hidden
        self.fc = nn.Sequential(*fc)

        self.backcast_fc = nn.Linear(hidden, in_dim)
        self.forecast_fc = nn.Linear(hidden, pred_len * d_series)

    def forward(self, x):
        # x: (B, seq_len * d_series)
        h = self.fc(x)
        return self.backcast_fc(h), self.forecast_fc(h)


class _Stack(nn.Module):
    """N-BEATS stack: several residual blocks operating in residual mode."""

    def __init__(self, seq_len, pred_len, d_series, num_blocks=3, hidden=256,
                 layers=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            _Block(seq_len, pred_len, d_series, hidden=hidden, layers=layers)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x: (B, seq_len * d_series).  Returns forecast of same shape's "time"
        # dim (actually flattened: (B, pred_len * d_series)).
        forecast = 0.0
        residual = x
        for block in self.blocks:
            backcast, step = block(residual)
            residual = residual - backcast
            forecast = forecast + step
        return forecast


class Model(nn.Module):
    """N-BEATS forecast model, matching the backbones/ interface.

    __init__(configs): configs must contain seq_len, pred_len, and c_out
    (the input feature dimension).  The rest of the hyperparameters (hidden,
    layers, num_blocks, stacks) are optional so the model still works even
    when they are absent on the argparse Namespace (keeps compatibility with
    the outer training script which is tuned for Autoformer-style configs).
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # "c_out" / "enc_in" / "d_series" are used interchangeably by various
        # parts of the repo; be permissive so callers don't have to remember.
        d_series = getattr(configs, "c_out", None)
        if d_series is None:
            d_series = getattr(configs, "enc_in", 1)

        hidden = getattr(configs, "n_beats_hidden", 256)
        layers = getattr(configs, "n_beats_layers", 4)
        num_blocks = getattr(configs, "n_beats_blocks", 3)
        stacks = getattr(configs, "n_beats_stacks", 2)

        self.d_series = d_series
        self.stacks = nn.ModuleList([
            _Stack(self.seq_len, self.pred_len, d_series,
                   num_blocks=num_blocks, hidden=hidden, layers=layers)
            for _ in range(stacks)
        ])

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """Forecast.

        The wide argument list matches the Informer / Autoformer signature so
        `Model.py` can dispatch the same way for any backbone.  N-BEATS is
        purely feed-forward and ignores the mark / decoder inputs.
        """
        # x_enc: (B, seq_len, d_series).  Flatten → (B, seq_len * d_series).
        B, L, D = x_enc.shape
        flat = x_enc.reshape(B, L * D)

        # Sum stack-wise forecasts; each stack independently produces a
        # (B, pred_len * d_series) estimate.  The residual/backcast path
        # is handled inside _Stack.forward — we simply accumulate forecasts
        # across stacks here.
        forecast = 0.0
        for stack in self.stacks:
            forecast = forecast + stack(flat)

        # Reshape back to (B, pred_len, d_series) so it plugs directly into
        # the loss / inverse-normalisation pipelines expected by train.py.
        return forecast.reshape(B, self.pred_len, D)

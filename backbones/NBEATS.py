"""N-BEATS: Neural Basis Expansion Analysis — Generic Architecture (paper baseline).

Original N-BEATS paper: https://arxiv.org/abs/1905.10437

Implementation aligned with the Dish-TS paper's N-BEATS backbone
settings.  Key parameters (from the official N-BEATS repo / paper Table):

    generic_architecture          = True
    num_stacks                    = 2
    num_blocks                    = 3   (blocks per stack)
    num_layers                    = 4   (FC layers per block)
    layer_width                   = 128
    expansion_coefficient_dim     = 128

The network is purely time-series — no covariates, no time embeddings,
no positional encoding.  Input shape: (B, lookback, D).  Output shape:
(B, horizon, D).

Multivariate: the N-BEATS *network* is strictly univariate (in_channels=1,
out_channels=1).  Following the Dish-TS paper ("* denotes N-BEATS
re-implemented for multivariate"), the multivariate model simply stacks
one N-BEATS net per input feature; every feature shares the same
block/stack hyper-parameters, and each feature's forecast is produced
independently.  This is *not* "flatten D features into a single
fully-connected N-BEATS" — it is D *independent* N-BEATS runs sharing
the same weights across features (so the network size does not grow with
D, matching the "layer_width=128" baseline).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#                                Block                                       #
# --------------------------------------------------------------------------- #
class _Block(nn.Module):
    """A single N-BEATS residual block.

    Input / output shapes are all (B, lookback) because the block itself
    operates on a *single feature* — the outer network handles the D-feature
    loop (see NBEATSStack).
    """

    def __init__(self, lookback: int, horizon: int, layer_width: int = 128,
                 num_layers: int = 4, expansion_coefficient_dim: int = 128):
        super().__init__()
        # 4 FC layers, each of shape (prev, layer_width)
        layers: list[nn.Module] = []
        prev = lookback
        for _ in range(num_layers):
            layers.append(nn.Linear(prev, layer_width))
            layers.append(nn.ReLU(inplace=True))
            prev = layer_width
        self.forward_fc = nn.Sequential(*layers)

        # Two "expansion coefficient" heads — shape (layer_width, theta_dim).
        # The final backcast / forecast projections *are* thetas; they are
        # linearly projected by the caller to (lookback) and (horizon) resp.
        self.backcast_coef = nn.Linear(layer_width, expansion_coefficient_dim)
        self.forecast_coef = nn.Linear(layer_width, expansion_coefficient_dim)

        # Final projections:  theta  ->  backcast (lookback) / forecast (horizon)
        self.backcast_proj = nn.Linear(expansion_coefficient_dim, lookback)
        self.forecast_proj = nn.Linear(expansion_coefficient_dim, horizon)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, lookback).  Returns (backcast, forecast) both of shape (B, lookback)/(B, horizon)."""
        h = self.forward_fc(x)
        backcast = self.backcast_proj(self.backcast_coef(h))  # (B, lookback)
        forecast = self.forecast_proj(self.forecast_coef(h))  # (B, horizon)
        return backcast, forecast


# --------------------------------------------------------------------------- #
#                                Stack                                       #
# --------------------------------------------------------------------------- #
class _Stack(nn.Module):
    """num_blocks residual blocks feeding into a common residual summation."""

    def __init__(self, lookback: int, horizon: int, num_blocks: int = 3,
                 layer_width: int = 128, num_layers: int = 4,
                 expansion_coefficient_dim: int = 128):
        super().__init__()
        self.blocks = nn.ModuleList([
            _Block(lookback, horizon,
                   layer_width=layer_width, num_layers=num_layers,
                   expansion_coefficient_dim=expansion_coefficient_dim)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, lookback).  Returns the stack forecast of shape (B, horizon)."""
        residual = x
        forecast_sum = 0.0
        for block in self.blocks:
            backcast, step = block(residual)
            residual = residual - backcast
            forecast_sum = forecast_sum + step
        return forecast_sum


# --------------------------------------------------------------------------- #
#                               N-BEATS Model                                 #
# --------------------------------------------------------------------------- #
class NBEATS(nn.Module):
    """N-BEATS — the generic architecture, D-feature aware.

    Each feature is processed independently by its own "stacked-blocks"
    network; weights are shared across features so the parameter count does
    not scale with D (this matches the Dish-TS paper's Table 2/3 N-BEATS*
    re-implementation).  A simple 1x1 conv-style linear projection is used
    per feature.
    """

    def __init__(self, lookback: int, horizon: int, num_stacks: int = 2,
                 num_blocks: int = 3, num_layers: int = 4,
                 layer_width: int = 128, expansion_coefficient_dim: int = 128):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon

        # Per-feature N-BEATS: stacks share weights across the D dim via a
        # conv1d-style reshape.  We implement it as a standard N-BEATS
        # operating on (B*D, lookback), then reshape back to (B, horizon, D).
        self.stacks = nn.ModuleList([
            _Stack(lookback, horizon, num_blocks=num_blocks,
                   layer_width=layer_width, num_layers=num_layers,
                   expansion_coefficient_dim=expansion_coefficient_dim)
            for _ in range(num_stacks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, lookback, D).  Returns (B, horizon, D)."""
        B, L, D = x.shape
        # reshape to (B*D, lookback) — same weights, per-feature processing
        flat = x.permute(0, 2, 1).reshape(B * D, L)
        forecast = 0.0
        for stack in self.stacks:
            forecast = forecast + stack(flat)
        # forecast shape: (B*D, horizon).  Fold back to (B, D, horizon) -> (B, horizon, D).
        return forecast.reshape(B, D, self.horizon).permute(0, 2, 1)


# --------------------------------------------------------------------------- #
#                                 Model wrapper                              #
# --------------------------------------------------------------------------- #
class Model(nn.Module):
    """N-BEATS wrapper matching the backbones/*.py interface used by train.py.

    __init__ takes an argparse-style config: config.seq_len, config.pred_len
    and (optionally) the N-BEATS-specific knobs below.  All fall back to the
    paper defaults so simply running

        python train.py --model NBEATS ...

    produces the paper baseline out of the box.
    """

    def __init__(self, configs):
        super().__init__()
        lookback = configs.seq_len
        horizon = configs.pred_len
        num_stacks = getattr(configs, "n_beats_stacks", 2)
        num_blocks = getattr(configs, "n_beats_blocks", 3)
        num_layers = getattr(configs, "n_beats_layers", 4)
        layer_width = getattr(configs, "n_beats_hidden", 128)
        expansion = getattr(configs, "n_beats_expansion_dim", 128)

        self.model = NBEATS(
            lookback=lookback,
            horizon=horizon,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_width=layer_width,
            expansion_coefficient_dim=expansion,
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """Matches the Informer / Autoformer signature so Model.py's
        'non-former' dispatch path (which only forwards x_enc) works."""
        return self.model(x_enc)

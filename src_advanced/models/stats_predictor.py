"""
StatsPredictor – Multi-task prediction of player statistics.

V2 Architecture: StackedResidualSE encoder + 4 DeepHead heads (96 dim).

Heads:
  Aces:  (p1_aces, p2_aces)
  DF:    (p1_df, p2_df)
  BP:    (p1_bp_saved, p1_bp_faced, p2_bp_saved, p2_bp_faced)
  Serve: (p1_1st_pct, p1_1st_win_pct, p2_1st_pct, p2_1st_win_pct)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ..config import (
    STATS_ENCODER_DIMS_V2, STATS_ENCODER_BLOCKS, STATS_ENCODER_DROPOUT_V2,
    STATS_HEAD_HIDDEN_V2, STATS_HEAD_BLOCKS, STATS_HEAD_DROPOUT_V2,
    SE_REDUCTION,
)
from .blocks import StackedResidualSE, DeepHead


class StatsDataset(Dataset):
    """Dataset for stats prediction: (X, targets_dict)."""

    ACE_COLS = ["p1_ace", "p2_ace"]
    DF_COLS = ["p1_df", "p2_df"]
    BP_COLS = ["p1_bpSaved", "p1_bpFaced", "p2_bpSaved", "p2_bpFaced"]
    SERVE_COLS = ["p1_1st_pct", "p1_1st_win_pct", "p2_1st_pct", "p2_1st_win_pct"]
    ALL_TARGETS = ACE_COLS + DF_COLS + BP_COLS + SERVE_COLS

    def __init__(self, X, target_arrays: dict, indices):
        self.X = torch.tensor(X[indices], dtype=torch.float32)
        self.targets = {}
        for name, arr in target_arrays.items():
            self.targets[name] = torch.tensor(arr[indices], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        target_dict = {name: t[idx] for name, t in self.targets.items()}
        return self.X[idx], target_dict


class StatsPredictor(nn.Module):
    """
    V2 Multi-task model: StackedResidualSE encoder + 4 DeepHead heads.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_dims: list | None = None,
        encoder_blocks: int | None = None,
        encoder_dropout: float | None = None,
        head_hidden: int | None = None,
        head_blocks: int | None = None,
        head_dropout: float | None = None,
    ):
        super().__init__()
        encoder_dims = encoder_dims or STATS_ENCODER_DIMS_V2
        encoder_blocks = encoder_blocks if encoder_blocks is not None else STATS_ENCODER_BLOCKS
        encoder_dropout = encoder_dropout if encoder_dropout is not None else STATS_ENCODER_DROPOUT_V2
        head_hidden = head_hidden if head_hidden is not None else STATS_HEAD_HIDDEN_V2
        head_blocks = head_blocks if head_blocks is not None else STATS_HEAD_BLOCKS
        head_dropout = head_dropout if head_dropout is not None else STATS_HEAD_DROPOUT_V2

        self.encoder = StackedResidualSE(
            input_dim=input_dim,
            layer_dims=encoder_dims,
            dropout=encoder_dropout,
            se_reduction=SE_REDUCTION,
            n_blocks_per_layer=encoder_blocks,
        )
        enc_out = encoder_dims[-1]

        self.ace_head = DeepHead(enc_out, 2, head_hidden, head_blocks, head_dropout)
        self.df_head = DeepHead(enc_out, 2, head_hidden, head_blocks, head_dropout)
        self.bp_head = DeepHead(enc_out, 4, head_hidden, head_blocks, head_dropout)
        self.serve_head = DeepHead(enc_out, 4, head_hidden, head_blocks, head_dropout)

    def forward(self, x):
        features = self.encoder(x)
        return {
            "aces": self.ace_head(features),
            "df": self.df_head(features),
            "bp": self.bp_head(features),
            "serve": self.serve_head(features),
        }


class StatsLoss(nn.Module):
    """Weighted sum of MSE losses for each head."""

    HEAD_WEIGHTS = {
        "aces": 1.0,
        "df": 1.0,
        "bp": 1.0,
        "serve": 1.0,
    }

    def __init__(self, weights: dict | None = None):
        super().__init__()
        self.weights = weights or self.HEAD_WEIGHTS
        self.mse = nn.MSELoss()

    def forward(self, predictions: dict, targets: dict):
        loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        head_target_map = {
            "aces": StatsDataset.ACE_COLS,
            "df": StatsDataset.DF_COLS,
            "bp": StatsDataset.BP_COLS,
            "serve": StatsDataset.SERVE_COLS,
        }

        for head_name, target_cols in head_target_map.items():
            pred = predictions[head_name]
            target_list = []
            for col in target_cols:
                if col in targets:
                    target_list.append(targets[col].unsqueeze(1))
                else:
                    target_list.append(torch.zeros_like(pred[:, :1]))

            target_tensor = torch.cat(target_list, dim=1)
            valid = ~torch.isnan(target_tensor)
            if valid.any():
                loss += self.weights[head_name] * self.mse(
                    pred[valid], target_tensor[valid]
                )

        return loss

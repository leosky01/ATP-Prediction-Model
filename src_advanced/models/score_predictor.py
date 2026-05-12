"""
ScorePredictor – Set score prediction for BO3 and BO5 matches.

V2 Architecture: StackedResidualSE encoder + DeepHead classifier.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ..config import (
    SCORE_ENCODER_DIMS, SCORE_ENCODER_BLOCKS, SCORE_ENCODER_DROPOUT,
    SCORE_HEAD_HIDDEN, SCORE_HEAD_BLOCKS, SCORE_HEAD_DROPOUT,
    SE_REDUCTION,
)
from .blocks import StackedResidualSE, DeepHead


class ScoreDataset(Dataset):
    """Dataset for score prediction: (X, y_class)."""

    def __init__(self, X, y, indices):
        self.X = torch.tensor(X[indices], dtype=torch.float32)
        self.y = torch.tensor(y[indices], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ScorePredictorBase(nn.Module):
    """V2 base class: StackedResidualSE encoder + DeepHead."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        encoder_dims: list | None = None,
        encoder_blocks: int | None = None,
        encoder_dropout: float | None = None,
        head_hidden: int | None = None,
        head_blocks: int | None = None,
        head_dropout: float | None = None,
    ):
        super().__init__()
        encoder_dims = encoder_dims or SCORE_ENCODER_DIMS
        encoder_blocks = encoder_blocks if encoder_blocks is not None else SCORE_ENCODER_BLOCKS
        encoder_dropout = encoder_dropout if encoder_dropout is not None else SCORE_ENCODER_DROPOUT
        head_hidden = head_hidden if head_hidden is not None else SCORE_HEAD_HIDDEN
        head_blocks = head_blocks if head_blocks is not None else SCORE_HEAD_BLOCKS
        head_dropout = head_dropout if head_dropout is not None else SCORE_HEAD_DROPOUT

        self.encoder = StackedResidualSE(
            input_dim=input_dim,
            layer_dims=encoder_dims,
            dropout=encoder_dropout,
            se_reduction=SE_REDUCTION,
            n_blocks_per_layer=encoder_blocks,
        )
        enc_out = encoder_dims[-1]

        self.head = DeepHead(enc_out, num_classes, head_hidden, head_blocks, head_dropout)

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)


class ScorePredictor3(ScorePredictorBase):
    """Score predictor for best-of-3: classes 0->2-0, 1->2-1."""

    def __init__(self, input_dim: int, **kwargs):
        super().__init__(input_dim, num_classes=2, **kwargs)


class ScorePredictor5(ScorePredictorBase):
    """Score predictor for best-of-5: classes 0->3-0, 1->3-1, 2->3-2."""

    def __init__(self, input_dim: int, **kwargs):
        super().__init__(input_dim, num_classes=3, **kwargs)


def compute_class_weights(labels: list | np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    labels = np.array(labels)
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(classes) * counts)
    return torch.tensor(weights, dtype=torch.float32)

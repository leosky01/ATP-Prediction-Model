#!/usr/bin/env python3
"""
ATP Advanced Prediction System – Unified Training Pipeline (V2)

V2 improvements: AdamW, CosineAnnealing + linear warmup, gradient clipping,
FocalLoss for score, label smoothing for classification, mixup augmentation.

Usage:
    python -m src_advanced.train --model games
    python -m src_advanced.train --model score
    python -m src_advanced.train --model duration
    python -m src_advanced.train --model stats
    python -m src_advanced.train --model all
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE, CAT_COLS_ADVANCED, FIGURES_ADV_DIR,
    GAMES_LR, GAMES_MAX_EPOCHS, GAMES_PATIENCE, GAMES_WEIGHT_DECAY,
    GAMES_ENCODER_DIMS, GAMES_HEAD_HIDDEN, GAMES_HEAD_BLOCKS,
    GRADIENT_CLIP_NORM, LABEL_SMOOTHING, MIXUP_ALPHA,
    MERGED_DATASET_PATH, MODELS_ADV_DIR,
    RANDOM_SEED,
    SCORE_LR, SCORE_MAX_EPOCHS, SCORE_PATIENCE, SCORE_WEIGHT_DECAY,
    SCORE_ENCODER_DIMS, SCORE_HEAD_HIDDEN, SCORE_HEAD_BLOCKS,
    SCORE_FOCAL_GAMMA,
    TRAIN_FRAC, VAL_FRAC,
    DURATION_LR, DURATION_MAX_EPOCHS, DURATION_PATIENCE, DURATION_WEIGHT_DECAY,
    DURATION_ENCODER_DIMS, DURATION_HEAD_HIDDEN, DURATION_HEAD_BLOCKS,
    STATS_LR, STATS_MAX_EPOCHS, STATS_PATIENCE, STATS_WEIGHT_DECAY,
    STATS_ENCODER_DIMS_V2, STATS_HEAD_HIDDEN_V2, STATS_HEAD_BLOCKS,
    WARMUP_EPOCHS,
)
from .data_loader import build_merged_dataset
from .feature_engineering import build_advanced_features, get_feature_columns
from .models.games_predictor import GamesDataset, GamesLoss, GamesPredictor
from .models.score_predictor import (
    ScoreDataset, ScorePredictor3, ScorePredictor5, compute_class_weights,
)
from .models.duration_predictor import DurationDataset, DurationPredictor, GaussianNLLLoss
from .models.stats_predictor import StatsDataset, StatsLoss, StatsPredictor
from .models.blocks import FocalLoss


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_splits(n: int, seed: int = RANDOM_SEED):
    """Return (train_idx, val_idx, test_idx) with a random shuffle."""
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    t0 = int(TRAIN_FRAC * n)
    t1 = int((TRAIN_FRAC + VAL_FRAC) * n)
    return idx[:t0], idx[t0:t1], idx[t1:]


def make_temporal_splits(n: int, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC):
    """Return (train_idx, val_idx, test_idx) using temporal ordering (no shuffle).
    Assumes data is sorted chronologically. No data leakage."""
    t0 = int(train_frac * n)
    t1 = int((train_frac + val_frac) * n)
    return np.arange(0, t0), np.arange(t0, t1), np.arange(t1, n)


def prepare_features(df: pd.DataFrame, train_idx: np.ndarray):
    """Prepare feature arrays with StandardScaler + OneHotEncoder fit on training."""
    num_cols, cat_cols = get_feature_columns(df)
    print(f"   Numerical features: {len(num_cols)}  |  Categorical: {len(cat_cols)}")

    # Fill NaN with training-set median (better than 0 for features with high NaN rates)
    X_num = df[num_cols].values.astype(np.float32)
    train_medians = np.nanmedian(X_num[train_idx], axis=0)
    # Fix any NaN in medians (entire column is NaN) → 0
    train_medians = np.nan_to_num(train_medians, nan=0.0)
    for j in range(X_num.shape[1]):
        mask = np.isnan(X_num[:, j])
        X_num[mask, j] = train_medians[j]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if cat_cols:
        encoder.fit(df.loc[train_idx, cat_cols].fillna("NA").astype(str))
        X_cat = encoder.transform(df[cat_cols].fillna("NA").astype(str))
    else:
        X_cat = np.zeros((len(df), 0), dtype=np.float32)

    scaler = StandardScaler()
    scaler.fit(X_num[train_idx])
    X_num = scaler.transform(X_num).astype(np.float32)

    # Store medians in scaler for evaluation-time NaN filling
    scaler.train_medians_ = train_medians
    scaler.num_cols_ = num_cols

    X = np.hstack([X_num, X_cat])
    return X, scaler, encoder, num_cols, cat_cols


def _save_artifacts(name: str, model, scaler, encoder, input_dim: int):
    """Save model weights, scaler, encoder, and input_dim."""
    MODELS_ADV_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_ADV_DIR / f"{name}_model.pt")
    joblib.dump(scaler, MODELS_ADV_DIR / f"{name}_scaler.joblib")
    joblib.dump(encoder, MODELS_ADV_DIR / f"{name}_encoder.joblib")
    joblib.dump({"input_dim": input_dim}, MODELS_ADV_DIR / f"{name}_meta.joblib")
    print(f"   Saved {name} artifacts to {MODELS_ADV_DIR}/")


def _load_or_build_dataset():
    """Load merged dataset or build from scratch."""
    if MERGED_DATASET_PATH.exists():
        print(f"Loading cached merged dataset: {MERGED_DATASET_PATH}")
        df = pd.read_csv(MERGED_DATASET_PATH, parse_dates=["Date"])
    else:
        df = build_merged_dataset()
    return df


# ── Warmup + CosineAnnealing scheduler ───────────────────────────────────────

class WarmupCosineScheduler:
    """Linear warmup for `warmup_epochs` then CosineAnnealingLR."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int):
        self.warmup_epochs = warmup_epochs
        self.cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6,
        )
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._current_epoch = 0

    def step(self):
        if self._current_epoch < self.warmup_epochs:
            # Linear warmup
            factor = (self._current_epoch + 1) / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base_lr * factor
        else:
            self.cosine.step()
        self._current_epoch += 1

    @property
    def optimizer(self):
        return self.cosine.optimizer


# ── Mixup utility ────────────────────────────────────────────────────────────

def _mixup_data(x, y, alpha: float = 0.2):
    """Apply mixup to input batch (regression target version)."""
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def _mixup_classification(x, y, alpha: float = 0.2):
    """Apply mixup to input batch, return mixed_x, y_a, y_b, lam."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


# ── Generic training loop ────────────────────────────────────────────────────

def train_generic(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    criterion: nn.Module,
    forward_fn,               # callable(model, batch) -> loss
    val_forward_fn=None,      # optional separate val forward; defaults to forward_fn
    warmup_epochs: int = WARMUP_EPOCHS,
    gradient_clip: float = GRADIENT_CLIP_NORM,
) -> dict:
    """
    Generic training loop with AdamW, cosine annealing + warmup, gradient clipping.

    Args:
        forward_fn: callable(model, batch, device, criterion) -> loss_tensor
                    batch is whatever the DataLoader yields.

    Returns:
        dict with keys: best_state_dict, best_val_loss
    """
    if val_forward_fn is None:
        val_forward_fn = forward_fn

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, max_epochs)

    best_loss, wait, state_best = float("inf"), 0, None

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        total_loss = 0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = forward_fn(model, batch, device, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Validate
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += val_forward_fn(model, batch, device, criterion).item()
                n_val += 1
        val_loss /= max(n_val, 1)
        scheduler.step()

        if val_loss < best_loss:
            best_loss, wait = val_loss, 0
            state_best = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"   Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"   Epoch {epoch}: train_loss={total_loss/n_batches:.4f}  val_loss={val_loss:.4f}")

    return {"best_state_dict": state_best, "best_val_loss": best_loss}


# ── Training functions ───────────────────────────────────────────────────────

def train_games(df: pd.DataFrame):
    """Train GamesPredictor with temporal split + pos_weight + median fill."""
    print("\n" + "=" * 60)
    print("TRAINING V2: GamesPredictor (Over/Under + Total Games)")
    print("=" * 60)

    valid = df["over_under"].notna() & df["total_games"].notna()
    df_sub = df[valid].copy().reset_index(drop=True)
    print(f"   Valid matches: {len(df_sub):,}")

    # Temporal split (no data leakage)
    train_i, val_i, test_i = make_temporal_splits(len(df_sub))
    print(f"   Split: train={len(train_i):,} val={len(val_i):,} test={len(test_i):,}")
    X, scaler, encoder, _, _ = prepare_features(df_sub, train_i)
    input_dim = X.shape[1]

    y_ou = df_sub["over_under"].values.astype(np.float32)
    y_total = df_sub["total_games"].values.astype(np.float32)

    # Compute pos_weight for class imbalance
    n_pos = y_ou[train_i].sum()
    n_neg = len(train_i) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    print(f"   O/U balance: {n_pos:.0f} Over / {n_neg:.0f} Under  pos_weight={pos_weight.item():.2f}")

    train_ds = GamesDataset(X, y_ou, y_total, train_i)
    val_ds = GamesDataset(X, y_ou, y_total, val_i)
    test_ds = GamesDataset(X, y_ou, y_total, test_i)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GamesPredictor(input_dim).to(device)
    criterion = GamesLoss()
    # Weighted BCE to handle class imbalance
    bce_weighted = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device),
        reduction="mean",
    )

    def games_forward(model, batch, device, criterion):
        xb, y_ou_b, y_tot_b = batch
        xb, y_ou_b, y_tot_b = xb.to(device), y_ou_b.to(device), y_tot_b.to(device)
        ou_logit, total_pred = model(xb)
        # Weighted BCE with label smoothing - focus on O/U
        y_smooth = y_ou_b * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        bce_loss = bce_weighted(ou_logit, y_smooth)
        valid = ~torch.isnan(y_tot_b.squeeze())
        mse_loss = torch.tensor(0.0, device=device)
        if valid.any():
            mse_loss = nn.functional.mse_loss(total_pred[valid], y_tot_b[valid])
        # Heavy weight on BCE (0.85) to focus on classification
        return 0.85 * bce_loss + 0.15 * mse_loss

    result = train_generic(
        model, train_loader, val_loader, device,
        max_epochs=GAMES_MAX_EPOCHS, patience=GAMES_PATIENCE,
        lr=5e-5, weight_decay=GAMES_WEIGHT_DECAY,
        criterion=criterion, forward_fn=games_forward,
    )
    model.load_state_dict(result["best_state_dict"])
    _save_artifacts("games", model, scaler, encoder, input_dim)

    # Quick test evaluation
    model.eval()
    all_ou_probs, all_ou_labels, all_total_preds, all_total_labels = [], [], [], []
    with torch.no_grad():
        for xb, y_ou_b, y_tot_b in test_loader:
            xb = xb.to(device)
            ou_logit, total_pred = model(xb)
            all_ou_probs.extend(torch.sigmoid(ou_logit).cpu().numpy().flatten())
            all_ou_labels.extend(y_ou_b.numpy().flatten())
            all_total_preds.extend(total_pred.cpu().numpy().flatten())
            all_total_labels.extend(y_tot_b.numpy().flatten())

    from sklearn.metrics import roc_auc_score, mean_absolute_error
    ou_probs = np.array(all_ou_probs)
    ou_labels = np.array(all_ou_labels)
    total_preds = np.array(all_total_preds)
    total_labels = np.array(all_total_labels)

    print(f"\n   Test Results:")
    print(f"   Over/Under AUC: {roc_auc_score(ou_labels, ou_probs):.4f}")
    print(f"   Over/Under Acc: {((ou_probs > 0.5).astype(int) == ou_labels).mean():.4f}")
    valid_total = ~np.isnan(total_labels)
    if valid_total.any():
        print(f"   Total Games MAE: {mean_absolute_error(total_labels[valid_total], total_preds[valid_total]):.2f}")


def train_score(df: pd.DataFrame):
    """Train ScorePredictor with FocalLoss + mixup."""
    print("\n" + "=" * 60)
    print("TRAINING V2: ScorePredictor (Set Score BO3 + BO5)")
    print("=" * 60)

    for best_of, predictor_cls, class_names, lr, patience, max_ep in [
        (3, ScorePredictor3, {"2-0": 0, "2-1": 1}, SCORE_LR, SCORE_PATIENCE, SCORE_MAX_EPOCHS),
        (5, ScorePredictor5, {"3-0": 0, "3-1": 1, "3-2": 2}, SCORE_LR, SCORE_PATIENCE, SCORE_MAX_EPOCHS),
    ]:
        print(f"\n--- Best of {best_of} ---")
        valid = (df["best_of"].fillna(3) == best_of) & df["set_score"].notna()
        df_sub = df[valid].copy().reset_index(drop=True)

        score_to_class = {s: i for i, s in enumerate(class_names.keys())}
        df_sub["_score_class"] = df_sub["set_score"].map(score_to_class)
        df_sub = df_sub[df_sub["_score_class"].notna()].copy().reset_index(drop=True)

        if len(df_sub) < 100:
            print(f"   Skipping: only {len(df_sub)} valid matches")
            continue

        print(f"   Valid matches: {len(df_sub):,}")

        train_i, val_i, test_i = make_splits(len(df_sub))
        X, scaler, encoder, _, _ = prepare_features(df_sub, train_i)
        input_dim = X.shape[1]
        y = df_sub["_score_class"].values.astype(np.int64)

        class_weights = compute_class_weights(y[train_i])

        train_ds = ScoreDataset(X, y, train_i)
        val_ds = ScoreDataset(X, y, val_i)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = predictor_cls(input_dim).to(device)
        focal_loss = FocalLoss(gamma=SCORE_FOCAL_GAMMA, weight=class_weights.to(device))

        def score_forward(model, batch, device, criterion):
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            if MIXUP_ALPHA > 0:
                mixed_x, y_a, y_b, lam = _mixup_classification(xb, yb, MIXUP_ALPHA)
                logits = model(mixed_x)
                loss = lam * F.cross_entropy(logits, y_a, weight=class_weights.to(device), label_smoothing=LABEL_SMOOTHING) \
                     + (1 - lam) * F.cross_entropy(logits, y_b, weight=class_weights.to(device), label_smoothing=LABEL_SMOOTHING)
            else:
                logits = model(xb)
                loss = focal_loss(logits, yb)
            return loss

        def score_val_forward(model, batch, device, criterion):
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            return focal_loss(logits, yb)

        result = train_generic(
            model, train_loader, val_loader, device,
            max_epochs=max_ep, patience=patience,
            lr=lr, weight_decay=SCORE_WEIGHT_DECAY,
            criterion=focal_loss,
            forward_fn=score_forward,
            val_forward_fn=score_val_forward,
        )
        model.load_state_dict(result["best_state_dict"])
        name = f"score_bo{best_of}"
        _save_artifacts(name, model, scaler, encoder, input_dim)


def train_duration(df: pd.DataFrame):
    """Train DurationPredictor with AdamW + cosine annealing."""
    print("\n" + "=" * 60)
    print("TRAINING V2: DurationPredictor (Match Duration)")
    print("=" * 60)

    valid = df["minutes"].notna()
    df_sub = df[valid].copy().reset_index(drop=True)
    print(f"   Valid matches: {len(df_sub):,}")

    train_i, val_i, test_i = make_splits(len(df_sub))
    X, scaler, encoder, _, _ = prepare_features(df_sub, train_i)
    input_dim = X.shape[1]
    y = df_sub["minutes"].values.astype(np.float32)

    train_ds = DurationDataset(X, y, train_i)
    val_ds = DurationDataset(X, y, val_i)
    test_ds = DurationDataset(X, y, test_i)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DurationPredictor(input_dim).to(device)
    criterion = GaussianNLLLoss()

    def duration_forward(model, batch, device, criterion):
        xb, yb = batch
        xb, yb = xb.to(device), yb.to(device)
        mean, log_var = model(xb)
        return criterion(mean, log_var, yb)

    result = train_generic(
        model, train_loader, val_loader, device,
        max_epochs=DURATION_MAX_EPOCHS, patience=DURATION_PATIENCE,
        lr=DURATION_LR, weight_decay=DURATION_WEIGHT_DECAY,
        criterion=criterion, forward_fn=duration_forward,
    )
    model.load_state_dict(result["best_state_dict"])
    _save_artifacts("duration", model, scaler, encoder, input_dim)

    # Quick test evaluation
    model.eval()
    all_means, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            mean, _ = model(xb)
            all_means.extend(mean.cpu().numpy().flatten())
            all_labels.extend(yb.numpy().flatten())

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    means = np.array(all_means)
    labels = np.array(all_labels)
    within_15 = np.abs(means - labels) <= 15
    print(f"\n   Test Results:")
    print(f"   RMSE: {np.sqrt(mean_squared_error(labels, means)):.2f} min")
    print(f"   MAE:  {mean_absolute_error(labels, means):.2f} min")
    print(f"   R^2:  {r2_score(labels, means):.4f}")
    print(f"   Within 15 min: {within_15.mean():.2%}")


def train_stats(df: pd.DataFrame):
    """Train StatsPredictor with AdamW + cosine annealing."""
    print("\n" + "=" * 60)
    print("TRAINING V2: StatsPredictor (Player Statistics)")
    print("=" * 60)

    target_cols = StatsDataset.ALL_TARGETS
    df_sub = df.copy()
    if "p1_1st_pct" not in df_sub.columns and "p1_1stIn" in df_sub.columns:
        df_sub["p1_1st_pct"] = df_sub["p1_1stIn"] / df_sub["p1_svpt"].replace(0, np.nan)
        df_sub["p2_1st_pct"] = df_sub["p2_1stIn"] / df_sub["p2_svpt"].replace(0, np.nan)
    if "p1_1st_win_pct" not in df_sub.columns and "p1_1stWon" in df_sub.columns:
        df_sub["p1_1st_win_pct"] = df_sub["p1_1stWon"] / df_sub["p1_1stIn"].replace(0, np.nan)
        df_sub["p2_1st_win_pct"] = df_sub["p2_1stWon"] / df_sub["p2_1stIn"].replace(0, np.nan)

    valid_mask = pd.Series(True, index=df_sub.index)
    for col in target_cols:
        if col in df_sub.columns:
            valid_mask &= df_sub[col].notna()
    df_sub = df_sub[valid_mask].reset_index(drop=True)

    if len(df_sub) < 100:
        print(f"   Skipping: only {len(df_sub)} valid matches")
        return

    print(f"   Valid matches: {len(df_sub):,}")

    train_i, val_i, test_i = make_splits(len(df_sub))
    X, scaler, encoder, _, _ = prepare_features(df_sub, train_i)
    input_dim = X.shape[1]

    target_arrays = {}
    for col in target_cols:
        if col in df_sub.columns:
            target_arrays[col] = df_sub[col].values.astype(np.float32)
        else:
            target_arrays[col] = np.full(len(df_sub), np.nan, dtype=np.float32)

    train_ds = StatsDataset(X, target_arrays, train_i)
    val_ds = StatsDataset(X, target_arrays, val_i)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StatsPredictor(input_dim).to(device)
    criterion = StatsLoss()

    def stats_forward(model, batch, device, criterion):
        xb, target_dict = batch
        xb = xb.to(device)
        target_dict = {k: v.to(device) for k, v in target_dict.items()}
        preds = model(xb)
        return criterion(preds, target_dict)

    result = train_generic(
        model, train_loader, val_loader, device,
        max_epochs=STATS_MAX_EPOCHS, patience=STATS_PATIENCE,
        lr=STATS_LR, weight_decay=STATS_WEIGHT_DECAY,
        criterion=criterion, forward_fn=stats_forward,
    )
    model.load_state_dict(result["best_state_dict"])
    _save_artifacts("stats", model, scaler, encoder, input_dim)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ATP Advanced Prediction Training (V2)")
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["games", "score", "duration", "stats", "all"],
        help="Which model(s) to train",
    )
    parser.add_argument(
        "--rebuild-data", action="store_true",
        help="Force rebuild merged dataset from scratch",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ATP ADVANCED PREDICTION SYSTEM – TRAINING V2")
    print(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    MODELS_ADV_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_ADV_DIR.mkdir(parents=True, exist_ok=True)

    # Load / build dataset
    print("\n1) Loading data...")
    if args.rebuild_data or not MERGED_DATASET_PATH.exists():
        df = build_merged_dataset(force_download=args.rebuild_data)
    else:
        df = pd.read_csv(MERGED_DATASET_PATH, parse_dates=["Date"])
        print(f"   Loaded {len(df):,} matches from cache")

    # Feature engineering
    print("\n2) Feature engineering...")
    df = build_advanced_features(df)

    # Train selected models
    print(f"\n3) Training model(s): {args.model}")

    if args.model in ("games", "all"):
        train_games(df)
    if args.model in ("score", "all"):
        train_score(df)
    if args.model in ("duration", "all"):
        train_duration(df)
    if args.model in ("stats", "all"):
        train_stats(df)

    print(f"\nDone: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)


if __name__ == "__main__":
    main()

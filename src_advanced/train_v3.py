#!/usr/bin/env python3
"""
ATP Advanced Prediction System – Training Pipeline V3

Key fixes over V2:
* Temporal split everywhere (no random shuffle leakage)
* Symmetric (p1<->p2 swap) data augmentation for OU, duration, stats
* Score predictor reframed as "n_sets" classification (symmetric, well-defined)
* LightGBM + Neural-net ensemble for O/U, blended via logistic regression
* GPU AMP for all neural nets
* Filtering out matches with no TML stats (post-2003 only)
* Cached features
* Saves all artifacts in models_advanced/ with `_v3` suffix and writes a
  consolidated metrics report at models_advanced/metrics_v3.json

Usage:
    python -m src_advanced.train_v3 --model all
    python -m src_advanced.train_v3 --model games
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, log_loss,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from .config import (
    BATCH_SIZE, CAT_COLS_ADVANCED, FIGURES_ADV_DIR,
    GAMES_THRESHOLD_BO3, GAMES_THRESHOLD_BO5,
    MERGED_DATASET_PATH, MODELS_ADV_DIR, RANDOM_SEED,
    GRADIENT_CLIP_NORM, WARMUP_EPOCHS,
)
from .feature_cache import load_or_build_features
from .feature_engineering import get_feature_columns
from .models.blocks import StackedResidualSE, DeepHead


# ──────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ──────────────────────────────────────────────────────────────────────────────


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def set_seed(seed: int = RANDOM_SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def temporal_split(dates: pd.Series, train_frac: float = 0.70, val_frac: float = 0.15):
    """Indexes for train/val/test, ordered by date.  Returns (train, val, test)."""
    order = np.argsort(dates.values)
    n = len(order)
    t0 = int(train_frac * n)
    t1 = int((train_frac + val_frac) * n)
    return order[:t0], order[t0:t1], order[t1:]


# ──────────────────────────────────────────────────────────────────────────────
# Symmetric (player-order) feature swap
# ──────────────────────────────────────────────────────────────────────────────


def _build_swap_index(num_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """For each feature column, determine the index of its mirror feature.

    Returns
    -------
    perm : np.ndarray[int]  perm[i] = index of the column to put at position i
                            when player order is swapped
    sign : np.ndarray[float]  sign multiplier (+1 or -1) for that column
    """
    n = len(num_cols)
    perm = np.arange(n)
    sign = np.ones(n, dtype=np.float32)

    def idx(name):
        return num_cols.index(name) if name in num_cols else -1

    # Pairs (a, b): swap columns and put a in b's place and vice versa
    swap_pairs = [
        ("odd_1", "odd_2"),
        ("log_odd_1", "log_odd_2"),
        ("implied_prob_1", "implied_prob_2"),
        ("p1_surf_win_rate", "p2_surf_win_rate"),
        ("p1_surf_avg_aces", "p2_surf_avg_aces"),
        ("p1_surf_avg_bp_save", "p2_surf_avg_bp_save"),
        ("p1_surf_avg_1st_pct", "p2_surf_avg_1st_pct"),
        ("p1_surf_matches", "p2_surf_matches"),
    ]
    # Raw rolling stats: p1_<stat>_w<w> ↔ p2_<stat>_w<w>
    from .config import ROLLING_WINDOWS as _RW
    for w in _RW:
        for stat in [
            "win", "ace", "df", "bp_save_rate", "1st_serve_pct",
            "1st_serve_win_pct", "2nd_serve_win_pct", "minutes", "streak",
            "total_games", "sv_gms",
        ]:
            swap_pairs.append((f"p1_{stat}_w{w}", f"p2_{stat}_w{w}"))
    # Most diff/ratio cols negate / invert under swap.  Generate dynamically:
    diff_cols = [c for c in num_cols if c.endswith("_diff") or "_diff_w" in c or c in (
        "rank_diff", "log_odds_ratio", "rank_points_diff", "age_diff",
        "height_diff", "seed_diff", "prob_diff",
    )]
    ratio_cols_log_invert = [c for c in num_cols if "_ratio_w" in c or c == "odds_ratio"]
    fatigue_pairs = [
        ("p1_matches_7d", "p2_matches_7d"),
        ("p1_matches_14d", "p2_matches_14d"),
        ("p1_matches_30d", "p2_matches_30d"),
    ]
    fatigue_diff = ["fatigue_diff_7d", "fatigue_diff_14d", "fatigue_diff_30d"]

    # Apply: diff cols flip sign
    for c in diff_cols:
        i = idx(c)
        if i >= 0:
            sign[i] = -1.0
    # Apply: ratio cols invert (we approximate inversion by *-1 in log-space later;
    # since values can be 0..inf, we do it by 1/value in build_swap_dataset)
    # We mark them with a sentinel sign of 0 to indicate "invert (1/x)"
    for c in ratio_cols_log_invert:
        i = idx(c)
        if i >= 0:
            sign[i] = 0.0  # sentinel meaning "invert"

    # Pair swaps
    all_pairs = swap_pairs + fatigue_pairs
    for a, b in all_pairs:
        ia, ib = idx(a), idx(b)
        if ia >= 0 and ib >= 0:
            perm[ia], perm[ib] = ib, ia
    # fatigue_diff negation
    for c in fatigue_diff:
        i = idx(c)
        if i >= 0:
            sign[i] = -1.0

    return perm, sign


def apply_swap(X_num: np.ndarray, perm: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Apply a swap operation to a numerical feature matrix.

    Columns marked with sign==0 are inverted (1/x) — handled before scaling.
    Columns with sign==-1 are negated.
    Columns with sign==+1 are kept as-is.
    Then columns are permuted via `perm` to swap paired raw features.
    """
    Y = X_num[:, perm].copy()
    # Apply sign per output position: a column at output position i originally
    # came from perm[i] and its sign is the *output* sign[i] computed.
    # Here we apply sign over OUTPUT indices (i.e. after permutation):
    for i in range(Y.shape[1]):
        s = sign[i]
        if s == -1.0:
            Y[:, i] = -Y[:, i]
        elif s == 0.0:
            # invert: 1 / x, protect 0
            col = Y[:, i]
            Y[:, i] = np.where(np.abs(col) > 1e-8, 1.0 / col, col)
    return Y


# ──────────────────────────────────────────────────────────────────────────────
# Feature preparation
# ──────────────────────────────────────────────────────────────────────────────


def prepare_features(
    df: pd.DataFrame,
    train_idx: np.ndarray,
) -> Tuple[np.ndarray, StandardScaler, OneHotEncoder, List[str], List[str]]:
    """StandardScaler + OneHotEncoder fit on training rows only."""
    num_cols, cat_cols = get_feature_columns(df)

    X_num = df[num_cols].to_numpy(dtype=np.float32, copy=True)

    # Per-column training median (NaN-safe)
    medians = np.nanmedian(X_num[train_idx], axis=0)
    medians = np.nan_to_num(medians, nan=0.0)
    for j in range(X_num.shape[1]):
        mask = np.isnan(X_num[:, j])
        X_num[mask, j] = medians[j]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if cat_cols:
        encoder.fit(df.loc[train_idx, cat_cols].fillna("NA").astype(str))
        X_cat = encoder.transform(df[cat_cols].fillna("NA").astype(str))
    else:
        X_cat = np.zeros((len(df), 0), dtype=np.float32)

    scaler = StandardScaler()
    scaler.fit(X_num[train_idx])
    X_num_scaled = scaler.transform(X_num).astype(np.float32)

    # Stash for downstream use
    scaler.train_medians_ = medians
    scaler.num_cols_ = num_cols

    X = np.hstack([X_num_scaled, X_cat]).astype(np.float32)
    return X, scaler, encoder, num_cols, cat_cols


def filter_post_stats(df: pd.DataFrame, min_year: int = 2003) -> pd.DataFrame:
    """Keep matches from `min_year` onwards.  TML stats coverage is much higher."""
    return df[df["Date"].dt.year >= min_year].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# LightGBM helpers (O/U classifier + total games regressor)
# ──────────────────────────────────────────────────────────────────────────────


def train_lgbm_classifier(
    X_train, y_train, X_val, y_val, *,
    seed: int = RANDOM_SEED,
    num_leaves: int = 90,
    max_depth: int = 10,
    lr: float = 0.015,
    n_estimators: int = 5000,
    early_stopping_rounds: int = 150,
    refit_on_trainval: bool = True,
):
    """Train LGBM classifier with early stopping on val, optionally refit on train+val."""
    import lightgbm as lgb
    params = dict(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        num_leaves=num_leaves,
        max_depth=max_depth,
        learning_rate=lr,
        min_child_samples=90,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=0.1,
        n_estimators=n_estimators,
        random_state=seed,
        verbosity=-1,
        n_jobs=-1,
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iter = model.best_iteration_ or model.n_estimators

    if refit_on_trainval and len(X_val) > 0:
        # Retrain on combined train+val with the best iteration count
        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
        refit_params = params.copy()
        refit_params["n_estimators"] = int(best_iter)
        final = lgb.LGBMClassifier(**refit_params)
        final.fit(X_all, y_all)
        final._val_model_for_stack = model  # keep early-stop model for stacking probas
        return final
    return model


def train_lgbm_regressor(
    X_train, y_train, X_val, y_val, *,
    seed: int = RANDOM_SEED,
):
    import lightgbm as lgb
    params = dict(
        objective="regression",
        metric="mae",
        boosting_type="gbdt",
        num_leaves=64,
        max_depth=8,
        learning_rate=0.03,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_estimators=3000,
        random_state=seed,
        verbosity=-1,
        n_jobs=-1,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Neural net for O/U (with mixed precision)
# ──────────────────────────────────────────────────────────────────────────────


class OUDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class OUNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = StackedResidualSE(
            input_dim=input_dim, layer_dims=[384, 192, 96], dropout=0.35, se_reduction=4,
            n_blocks_per_layer=1,
        )
        self.head = DeepHead(96, 1, hidden_dim=64, n_blocks=2, dropout=0.25)

    def forward(self, x):
        return self.head(self.encoder(x))


def train_nn_ou(X_train, y_train, X_val, y_val, *, epochs=60, batch=512, lr=8e-4, seed: int = RANDOM_SEED):
    set_seed(seed)
    train_loader = DataLoader(
        OUDataset(X_train, y_train), batch_size=batch, shuffle=True,
        num_workers=2, pin_memory=USE_AMP,
    )
    val_loader = DataLoader(
        OUDataset(X_val, y_val), batch_size=batch, shuffle=False,
        num_workers=2, pin_memory=USE_AMP,
    )

    model = OUNet(X_train.shape[1]).to(DEVICE)
    pos_weight = torch.tensor(
        [(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)],
        device=DEVICE, dtype=torch.float32,
    )
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_auc = 0.0
    best_state = None
    patience = 10
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logit = model(xb)
                loss = crit(logit, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            scaler.step(opt)
            scaler.update()
        sched.step()

        # Validation AUC
        model.eval()
        probs = []
        labs = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    logit = model(xb)
                probs.append(torch.sigmoid(logit).cpu().numpy().flatten())
                labs.append(yb.cpu().numpy().flatten())
        probs = np.concatenate(probs)
        labs = np.concatenate(labs)
        auc = roc_auc_score(labs, probs)
        if auc > best_auc:
            best_auc, wait = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
        if epoch % 5 == 0:
            print(f"   [NN-OU] epoch {epoch:3d}  val_auc={auc:.4f}  best={best_auc:.4f}")

    model.load_state_dict(best_state)
    return model, best_auc


def predict_nn(model: nn.Module, X: np.ndarray, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch].astype(np.float32)).to(DEVICE)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logit = model(xb)
            out.append(torch.sigmoid(logit).cpu().numpy().flatten())
    return np.concatenate(out)


# ──────────────────────────────────────────────────────────────────────────────
# 1) GAMES (Over/Under + Total)  — LightGBM + NN + stacking, per best_of
# ──────────────────────────────────────────────────────────────────────────────


def train_games_v3(df: pd.DataFrame) -> Dict:
    print("\n" + "=" * 70)
    print("TRAIN V3 – Games (Over/Under + Total)")
    print("=" * 70)

    valid = df["over_under"].notna() & df["total_games"].notna() & df["best_of"].notna()
    df_sub = df[valid].copy().reset_index(drop=True)
    # drop huge outlier matches (>100 games) — these are pre-tiebreak finals
    df_sub = df_sub[df_sub["total_games"] <= 80].reset_index(drop=True)
    print(f"   Valid matches: {len(df_sub):,}")

    train_i, val_i, test_i = temporal_split(df_sub["Date"])
    print(f"   Temporal split: train={len(train_i):,} val={len(val_i):,} test={len(test_i):,}")
    print(f"   Train period: {df_sub['Date'].iloc[train_i].min():%Y-%m-%d} → {df_sub['Date'].iloc[train_i].max():%Y-%m-%d}")
    print(f"   Test  period: {df_sub['Date'].iloc[test_i].min():%Y-%m-%d} → {df_sub['Date'].iloc[test_i].max():%Y-%m-%d}")

    X, scaler, encoder, num_cols, cat_cols = prepare_features(df_sub, train_i)
    print(f"   Input dim: {X.shape[1]}  (num={len(num_cols)} cat_oh={X.shape[1]-len(num_cols)})")

    y_ou = df_sub["over_under"].to_numpy(dtype=np.float32)
    y_total = df_sub["total_games"].to_numpy(dtype=np.float32)
    best_of = df_sub["best_of"].to_numpy(dtype=np.int64)

    metrics = {}

    # ── Symmetric augmentation indices  ──
    perm, sign = _build_swap_index(num_cols)

    def augment_train(X_idx, y_idx, total_idx):
        """Return X_train, y_train, y_total_train with symmetric augmentation.

        The swap is applied on the *unscaled* num cols isn't possible since we
        already scaled. We approximate by: numerical features that should
        negate (-1) just become -X, ratio cols become 1/X (computed in z-space
        approximately by mirroring around mean). For simplicity we apply on the
        full scaled X using the precomputed perm+sign at scaled-space:
        - sign=-1 → negate (works since standardised has 0-mean)
        - sign=0  → flip around the column mean (1/x heuristic): -x
        - sign=1  → keep
        - permutation: swap pair columns.
        One-hot cat columns: kept as-is (they don't depend on player order).
        """
        n_num = len(num_cols)
        Xn = X_idx[:, :n_num].copy()
        Xc = X_idx[:, n_num:]
        # Apply permutation on num cols
        Xn_perm = Xn[:, perm]
        # Apply sign / inversion in scaled space:
        # since we don't have raw vals, approximate: negate for sign==-1 and sign==0
        # (inversion mapped to negation in scaled space — first-order correct
        # for symmetric distributions)
        s = sign.copy()
        s[s == 0.0] = -1.0
        Xn_aug = Xn_perm * s[None, :]
        X_aug = np.hstack([Xn_aug, Xc])  # cat features unchanged
        X_total = np.vstack([X_idx, X_aug])
        # OU label is invariant under p1<->p2 swap
        y_total_lbl = np.concatenate([y_idx, y_idx])
        y_tot = np.concatenate([total_idx, total_idx])
        return X_total, y_total_lbl, y_tot

    # Split data by best_of (BO3 vs BO5 have different distributions/thresholds)
    games_artifacts = {
        "scaler": scaler,
        "encoder": encoder,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "input_dim": X.shape[1],
        "perm": perm.tolist(),
        "sign": sign.tolist(),
    }

    # Save shared scaler/encoder once
    MODELS_ADV_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_ADV_DIR / "games_v3_scaler.joblib")
    joblib.dump(encoder, MODELS_ADV_DIR / "games_v3_encoder.joblib")

    for bo, threshold in [(3, GAMES_THRESHOLD_BO3), (5, GAMES_THRESHOLD_BO5)]:
        sel = best_of == bo
        if sel.sum() < 1000:
            print(f"\n   [BO{bo}] Not enough data ({sel.sum()}), skipping")
            continue

        print(f"\n   ── BO{bo}  (threshold={threshold} games)  n={sel.sum():,} ──")

        tr_mask = sel[train_i]
        va_mask = sel[val_i]
        te_mask = sel[test_i]

        X_tr, X_va, X_te = X[train_i][tr_mask], X[val_i][va_mask], X[test_i][te_mask]
        y_tr, y_va, y_te = y_ou[train_i][tr_mask], y_ou[val_i][va_mask], y_ou[test_i][te_mask]
        t_tr, t_va, t_te = y_total[train_i][tr_mask], y_total[val_i][va_mask], y_total[test_i][te_mask]

        X_tr_aug, y_tr_aug, t_tr_aug = augment_train(X_tr, y_tr, t_tr)
        print(f"     train={len(X_tr):,}  train(aug)={len(X_tr_aug):,}  val={len(X_va):,}  test={len(X_te):,}")
        print(f"     OU base rate train: {y_tr.mean():.3f}  test: {y_te.mean():.3f}")

        # 1) LightGBM classifier on AUC (NO augmentation - LGBM doesn't need it)
        print(f"     [LGBM-OU] training...")
        lgb_cls = train_lgbm_classifier(X_tr, y_tr, X_va, y_va)
        # Validation predictions come from the early-stop model (NOT the train+val refit)
        val_model = getattr(lgb_cls, "_val_model_for_stack", lgb_cls)
        lgb_val_proba = val_model.predict_proba(X_va)[:, 1]
        lgb_test_proba = lgb_cls.predict_proba(X_te)[:, 1]
        print(f"     [LGBM-OU] val AUC={roc_auc_score(y_va, lgb_val_proba):.4f}  test AUC={roc_auc_score(y_te, lgb_test_proba):.4f}")

        # 2) Neural net WITH augmentation: 3-seed average for variance reduction
        print(f"     [NN-OU] training on GPU={USE_AMP}  (3 seeds)...")
        seeds = [RANDOM_SEED, RANDOM_SEED + 1, RANDOM_SEED + 2]
        nn_models = []
        for s in seeds:
            print(f"     [NN-OU]  seed={s}")
            m_, _ = train_nn_ou(X_tr_aug, y_tr_aug, X_va, y_va, seed=s)
            nn_models.append(m_)
        nn_val_probs = np.stack([predict_nn(m_, X_va) for m_ in nn_models], axis=0)
        nn_test_probs = np.stack([predict_nn(m_, X_te) for m_ in nn_models], axis=0)
        nn_val_proba = nn_val_probs.mean(axis=0)
        nn_test_proba = nn_test_probs.mean(axis=0)
        print(f"     [NN-OU]   val AUC={roc_auc_score(y_va, nn_val_proba):.4f}  test AUC={roc_auc_score(y_te, nn_test_proba):.4f}")

        # 3) Stacker: logistic regression on val probas
        stacker = LogisticRegression(C=1.0, max_iter=1000)
        stack_val = np.column_stack([lgb_val_proba, nn_val_proba])
        stacker.fit(stack_val, y_va)
        stack_test = np.column_stack([lgb_test_proba, nn_test_proba])
        ens_test_proba_raw = stacker.predict_proba(stack_test)[:, 1]
        ens_val_proba = stacker.predict_proba(stack_val)[:, 1]

        # 3b) Isotonic calibration on val
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(ens_val_proba, y_va)
        ens_test_proba = calibrator.predict(ens_test_proba_raw)
        ens_test_proba = np.clip(ens_test_proba, 1e-6, 1 - 1e-6)

        # 4) Total games regressor (no augmentation - regression target is symmetric)
        print(f"     [LGBM-Total] training...")
        lgb_reg = train_lgbm_regressor(X_tr, t_tr, X_va, t_va)
        # Refit on train+val
        lgb_reg_best_iter = lgb_reg.best_iteration_ or lgb_reg.n_estimators
        import lightgbm as _lgb
        lgb_reg_final = _lgb.LGBMRegressor(**{**lgb_reg.get_params(), "n_estimators": int(lgb_reg_best_iter)})
        lgb_reg_final.fit(np.vstack([X_tr, X_va]), np.concatenate([t_tr, t_va]))
        total_test = lgb_reg_final.predict(X_te)

        # Metrics
        m = {
            "n_train_aug": int(len(X_tr_aug)), "n_val": int(len(X_va)), "n_test": int(len(X_te)),
            "lgbm_val_auc": float(roc_auc_score(y_va, lgb_val_proba)),
            "lgbm_test_auc": float(roc_auc_score(y_te, lgb_test_proba)),
            "lgbm_test_acc": float(((lgb_test_proba > 0.5).astype(int) == y_te).mean()),
            "lgbm_test_brier": float(brier_score_loss(y_te, lgb_test_proba)),
            "nn_val_auc": float(roc_auc_score(y_va, nn_val_proba)),
            "nn_test_auc": float(roc_auc_score(y_te, nn_test_proba)),
            "ensemble_test_auc": float(roc_auc_score(y_te, ens_test_proba)),
            "ensemble_test_acc": float(((ens_test_proba > 0.5).astype(int) == y_te).mean()),
            "ensemble_test_brier": float(brier_score_loss(y_te, ens_test_proba)),
            "ensemble_test_logloss": float(log_loss(y_te, np.clip(ens_test_proba, 1e-6, 1-1e-6))),
            "total_mae": float(mean_absolute_error(t_te, total_test)),
            "total_rmse": float(np.sqrt(mean_squared_error(t_te, total_test))),
            "total_r2": float(r2_score(t_te, total_test)),
            "stacker_coef": stacker.coef_.tolist(),
            "stacker_intercept": stacker.intercept_.tolist(),
            "threshold": threshold,
        }
        metrics[f"bo{bo}"] = m
        print(f"     →  ENSEMBLE test AUC = {m['ensemble_test_auc']:.4f}  acc={m['ensemble_test_acc']:.4f}  brier={m['ensemble_test_brier']:.4f}")
        print(f"     →  TOTAL    test MAE = {m['total_mae']:.2f}  R²={m['total_r2']:.4f}")

        # Save BO-specific artifacts
        lgb_cls.booster_.save_model(str(MODELS_ADV_DIR / f"games_v3_bo{bo}_lgb_ou.txt"))
        lgb_reg_final.booster_.save_model(str(MODELS_ADV_DIR / f"games_v3_bo{bo}_lgb_total.txt"))
        for si, mdl in enumerate(nn_models):
            torch.save(mdl.state_dict(), MODELS_ADV_DIR / f"games_v3_bo{bo}_nn_s{si}.pt")
        joblib.dump(stacker, MODELS_ADV_DIR / f"games_v3_bo{bo}_stacker.joblib")
        joblib.dump(calibrator, MODELS_ADV_DIR / f"games_v3_bo{bo}_calibrator.joblib")
        games_artifacts[f"bo{bo}"] = m
        games_artifacts[f"bo{bo}_n_nn_seeds"] = len(nn_models)

    joblib.dump(games_artifacts, MODELS_ADV_DIR / "games_v3_meta.joblib")
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 2) DURATION  — LightGBM + GPU-NN regression on minutes (symmetric task)
# ──────────────────────────────────────────────────────────────────────────────


class DurationDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class DurationNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = StackedResidualSE(
            input_dim=input_dim, layer_dims=[384, 192, 96], dropout=0.25,
            se_reduction=4, n_blocks_per_layer=1,
        )
        self.mean = DeepHead(96, 1, 64, 2, 0.2)
        self.logvar = DeepHead(96, 1, 32, 1, 0.2)

    def forward(self, x):
        f = self.encoder(x)
        m = self.mean(f)
        lv = torch.clamp(self.logvar(f), min=-2, max=7)
        return m, lv


def gaussian_nll(mean, logvar, y):
    var = torch.exp(logvar)
    return 0.5 * (logvar + (y - mean) ** 2 / var).mean()


def train_duration_v3(df: pd.DataFrame) -> Dict:
    print("\n" + "=" * 70)
    print("TRAIN V3 – Duration (minutes)")
    print("=" * 70)

    valid = df["minutes"].notna() & (df["minutes"] > 20) & (df["minutes"] < 360)
    df_sub = df[valid].copy().reset_index(drop=True)
    print(f"   Valid matches (20<min<360): {len(df_sub):,}")

    train_i, val_i, test_i = temporal_split(df_sub["Date"])
    print(f"   Split: train={len(train_i):,} val={len(val_i):,} test={len(test_i):,}")

    X, scaler, encoder, num_cols, cat_cols = prepare_features(df_sub, train_i)
    y = df_sub["minutes"].to_numpy(dtype=np.float32)

    # Symmetric augmentation for training
    perm, sign = _build_swap_index(num_cols)

    n_num = len(num_cols)
    Xn_tr = X[train_i][:, :n_num]
    Xc_tr = X[train_i][:, n_num:]
    s = sign.copy(); s[s == 0.0] = -1.0
    Xn_tr_aug = Xn_tr[:, perm] * s[None, :]
    X_tr = np.vstack([X[train_i], np.hstack([Xn_tr_aug, Xc_tr])]).astype(np.float32)
    y_tr = np.concatenate([y[train_i], y[train_i]])
    X_va, y_va = X[val_i], y[val_i]
    X_te, y_te = X[test_i], y[test_i]
    print(f"   Train(aug)={len(X_tr):,}  val={len(X_va):,}  test={len(X_te):,}")

    # LightGBM
    print(f"   [LGBM] training...")
    lgb_reg = train_lgbm_regressor(X_tr, y_tr, X_va, y_va)
    lgb_test = lgb_reg.predict(X_te)

    # NN with Gaussian NLL
    print(f"   [NN] training on GPU={USE_AMP}...")
    set_seed()
    train_loader = DataLoader(DurationDS(X_tr, y_tr), batch_size=512, shuffle=True,
                              num_workers=2, pin_memory=USE_AMP)
    val_loader = DataLoader(DurationDS(X_va, y_va), batch_size=512, num_workers=2,
                            pin_memory=USE_AMP)

    model = DurationNet(X_tr.shape[1]).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80, eta_min=1e-6)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    best_mae = float("inf")
    best_state = None
    patience = 8
    wait = 0
    for epoch in range(1, 81):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                m, lv = model(xb)
                loss = gaussian_nll(m, lv, yb)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            grad_scaler.step(opt)
            grad_scaler.update()
        sched.step()

        # val
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    m, _ = model(xb)
                preds.append(m.cpu().numpy().flatten())
        preds = np.concatenate(preds)
        mae = mean_absolute_error(y_va, preds)
        if mae < best_mae:
            best_mae, wait = mae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
        if epoch % 5 == 0:
            print(f"   [NN] epoch {epoch:3d}  val_mae={mae:.2f}  best={best_mae:.2f}")
    model.load_state_dict(best_state)

    # Predict NN on test
    model.eval()
    nn_test = []
    nn_test_std = []
    with torch.no_grad():
        for i in range(0, len(X_te), 1024):
            xb = torch.from_numpy(X_te[i:i+1024].astype(np.float32)).to(DEVICE)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                m, lv = model(xb)
            nn_test.append(m.cpu().numpy().flatten())
            nn_test_std.append(np.exp(0.5 * lv.cpu().numpy().flatten()))
    nn_test = np.concatenate(nn_test)
    nn_test_std = np.concatenate(nn_test_std)

    # Ensemble (simple average works well in practice)
    ens_test = 0.5 * (lgb_test + nn_test)

    metrics = {
        "lgbm_mae": float(mean_absolute_error(y_te, lgb_test)),
        "lgbm_rmse": float(np.sqrt(mean_squared_error(y_te, lgb_test))),
        "lgbm_r2": float(r2_score(y_te, lgb_test)),
        "nn_mae": float(mean_absolute_error(y_te, nn_test)),
        "nn_rmse": float(np.sqrt(mean_squared_error(y_te, nn_test))),
        "nn_r2": float(r2_score(y_te, nn_test)),
        "ensemble_mae": float(mean_absolute_error(y_te, ens_test)),
        "ensemble_rmse": float(np.sqrt(mean_squared_error(y_te, ens_test))),
        "ensemble_r2": float(r2_score(y_te, ens_test)),
        "within_15": float((np.abs(ens_test - y_te) <= 15).mean()),
        "within_30": float((np.abs(ens_test - y_te) <= 30).mean()),
    }
    print(f"   →  ENSEMBLE  MAE={metrics['ensemble_mae']:.2f}  R²={metrics['ensemble_r2']:.4f}  within15={metrics['within_15']:.2%}")

    # Save
    joblib.dump(scaler, MODELS_ADV_DIR / "duration_v3_scaler.joblib")
    joblib.dump(encoder, MODELS_ADV_DIR / "duration_v3_encoder.joblib")
    lgb_reg.booster_.save_model(str(MODELS_ADV_DIR / "duration_v3_lgb.txt"))
    torch.save(model.state_dict(), MODELS_ADV_DIR / "duration_v3_nn.pt")
    joblib.dump({
        "input_dim": X.shape[1], "num_cols": num_cols, "cat_cols": cat_cols,
        "perm": perm.tolist(), "sign": sign.tolist(),
        "metrics": metrics,
    }, MODELS_ADV_DIR / "duration_v3_meta.joblib")
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 3) N_SETS  — replacement for ScorePredictor.  Symmetric "number of sets" task.
# ──────────────────────────────────────────────────────────────────────────────


def parse_n_sets(score: str) -> Optional[int]:
    """Count sets in a tennis score string."""
    if not isinstance(score, str):
        return None
    import re
    n = 0
    for tok in score.split():
        tok = tok.strip().upper()
        if tok in ("RET", "W/O", "DEF", "ABD", "WALKOVER"):
            break
        if re.match(r"\d+-\d+", tok):
            n += 1
    return n if n > 0 else None


def train_nsets_v3(df: pd.DataFrame) -> Dict:
    print("\n" + "=" * 70)
    print("TRAIN V3 – Number of Sets (symmetric replacement for score)")
    print("=" * 70)

    df_sub = df.copy()
    df_sub["n_sets"] = df_sub["score"].apply(parse_n_sets)
    df_sub = df_sub[df_sub["n_sets"].notna() & df_sub["best_of"].notna()].reset_index(drop=True)
    # Clamp to valid range per best_of
    df_sub["best_of"] = df_sub["best_of"].astype(int)
    df_sub = df_sub[
        ((df_sub["best_of"] == 3) & df_sub["n_sets"].isin([2, 3])) |
        ((df_sub["best_of"] == 5) & df_sub["n_sets"].isin([3, 4, 5]))
    ].reset_index(drop=True)

    metrics = {}
    train_i, val_i, test_i = temporal_split(df_sub["Date"])
    X, scaler, encoder, num_cols, cat_cols = prepare_features(df_sub, train_i)
    perm, sign = _build_swap_index(num_cols)
    s = sign.copy(); s[s == 0.0] = -1.0
    n_num = len(num_cols)
    best_of_arr = df_sub["best_of"].to_numpy(dtype=np.int64)
    y_all = df_sub["n_sets"].to_numpy(dtype=np.int64)

    joblib.dump(scaler, MODELS_ADV_DIR / "nsets_v3_scaler.joblib")
    joblib.dump(encoder, MODELS_ADV_DIR / "nsets_v3_encoder.joblib")

    for bo, classes in [(3, [2, 3]), (5, [3, 4, 5])]:
        sel = best_of_arr == bo
        if sel.sum() < 1000:
            print(f"   [BO{bo}] insufficient data, skipping")
            continue
        print(f"\n   ── BO{bo}  classes={classes}  n={sel.sum():,} ──")
        cls_to_idx = {c: i for i, c in enumerate(classes)}

        tr_mask = sel[train_i]; va_mask = sel[val_i]; te_mask = sel[test_i]
        X_tr0 = X[train_i][tr_mask]; X_va = X[val_i][va_mask]; X_te = X[test_i][te_mask]
        y_tr0 = y_all[train_i][tr_mask]; y_va = y_all[val_i][va_mask]; y_te = y_all[test_i][te_mask]
        y_tr0 = np.array([cls_to_idx[v] for v in y_tr0])
        y_va = np.array([cls_to_idx[v] for v in y_va])
        y_te = np.array([cls_to_idx[v] for v in y_te])

        # NO augmentation — n_sets is symmetric so it doesn't add signal,
        # only duplicates information.  LGBM doesn't benefit from dup rows.
        X_tr = X_tr0.astype(np.float32)
        y_tr = y_tr0.copy()

        print(f"     train={len(X_tr):,}  val={len(X_va):,}  test={len(X_te):,}")
        print(f"     class dist (test): {dict(zip(*np.unique(y_te, return_counts=True)))}")

        # LightGBM (binary if 2 classes, multiclass otherwise)
        import lightgbm as lgb
        is_binary = len(classes) == 2
        params = dict(
            objective="binary" if is_binary else "multiclass",
            metric="auc" if is_binary else "multi_logloss",
            boosting_type="gbdt",
            num_leaves=90, max_depth=10, learning_rate=0.015,
            min_child_samples=90, subsample=0.7, subsample_freq=1,
            colsample_bytree=0.5, reg_alpha=1.0, reg_lambda=0.1,
            n_estimators=5000, random_state=RANDOM_SEED, verbosity=-1, n_jobs=-1,
        )
        if not is_binary:
            params["num_class"] = len(classes)
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
        # Refit on train+val with best iter
        best_iter = m.best_iteration_ or m.n_estimators
        refit_params = params.copy(); refit_params["n_estimators"] = int(best_iter)
        m_final = lgb.LGBMClassifier(**refit_params)
        m_final.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))

        probs_te = m_final.predict_proba(X_te)
        if probs_te.ndim == 1:
            probs_te = np.column_stack([1 - probs_te, probs_te])
        pred_te = probs_te.argmax(axis=1)
        acc = accuracy_score(y_te, pred_te)
        f1 = f1_score(y_te, pred_te, average="macro")
        ll = log_loss(y_te, np.clip(probs_te, 1e-6, 1 - 1e-6), labels=list(range(len(classes))))
        baseline_acc = max(np.bincount(y_te)) / len(y_te)

        bo_metrics = {
            "n_train": int(len(X_tr)), "n_val": int(len(X_va)), "n_test": int(len(X_te)),
            "accuracy": float(acc), "f1_macro": float(f1), "logloss": float(ll),
            "baseline_acc": float(baseline_acc),
            "classes": classes,
        }
        if is_binary:
            auc = roc_auc_score(y_te, probs_te[:, 1])
            bo_metrics["test_auc"] = float(auc)
            print(f"     →  acc={acc:.4f}  baseline={baseline_acc:.4f}  AUC={auc:.4f}  f1_macro={f1:.4f}  logloss={ll:.4f}")
        else:
            # Compute one-vs-rest AUC
            try:
                auc_ovr = roc_auc_score(y_te, probs_te, multi_class="ovr", average="macro")
                bo_metrics["test_auc_ovr"] = float(auc_ovr)
            except Exception:
                auc_ovr = float("nan")
            print(f"     →  acc={acc:.4f}  baseline={baseline_acc:.4f}  AUC_ovr={auc_ovr:.4f}  f1_macro={f1:.4f}  logloss={ll:.4f}")

        metrics[f"bo{bo}"] = bo_metrics
        m_final.booster_.save_model(str(MODELS_ADV_DIR / f"nsets_v3_bo{bo}_lgb.txt"))

    joblib.dump({
        "num_cols": num_cols, "cat_cols": cat_cols, "input_dim": X.shape[1],
        "perm": perm.tolist(), "sign": sign.tolist(),
        "metrics": metrics,
    }, MODELS_ADV_DIR / "nsets_v3_meta.joblib")
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 4) STATS  — multi-target LightGBM (cheap, robust)
# ──────────────────────────────────────────────────────────────────────────────


def train_stats_v3(df: pd.DataFrame) -> Dict:
    print("\n" + "=" * 70)
    print("TRAIN V3 – Player Stats (multi-target)")
    print("=" * 70)
    targets = [
        "p1_ace", "p2_ace", "p1_df", "p2_df",
        "p1_bpSaved", "p1_bpFaced", "p2_bpSaved", "p2_bpFaced",
    ]
    # Compute derived
    df = df.copy()
    df["p1_1st_pct"] = df["p1_1stIn"] / df["p1_svpt"].replace(0, np.nan)
    df["p2_1st_pct"] = df["p2_1stIn"] / df["p2_svpt"].replace(0, np.nan)
    df["p1_1st_win_pct"] = df["p1_1stWon"] / df["p1_1stIn"].replace(0, np.nan)
    df["p2_1st_win_pct"] = df["p2_1stWon"] / df["p2_1stIn"].replace(0, np.nan)
    targets += ["p1_1st_pct", "p2_1st_pct", "p1_1st_win_pct", "p2_1st_win_pct"]

    valid = df[targets].notna().all(axis=1)
    df_sub = df[valid].reset_index(drop=True)
    print(f"   Valid: {len(df_sub):,}")
    train_i, val_i, test_i = temporal_split(df_sub["Date"])
    X, scaler, encoder, num_cols, cat_cols = prepare_features(df_sub, train_i)
    print(f"   Input dim: {X.shape[1]}")

    import lightgbm as lgb
    metrics = {}
    artifacts = {}
    for tgt in targets:
        y = df_sub[tgt].to_numpy(dtype=np.float32)
        m = lgb.LGBMRegressor(
            objective="regression", metric="mae",
            num_leaves=63, max_depth=8, learning_rate=0.05,
            min_child_samples=80, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.1,
            n_estimators=2000, random_state=RANDOM_SEED, verbosity=-1, n_jobs=-1,
        )
        m.fit(X[train_i], y[train_i], eval_set=[(X[val_i], y[val_i])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        pred = m.predict(X[test_i])
        actual = y[test_i]
        metrics[tgt] = {
            "mae": float(mean_absolute_error(actual, pred)),
            "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
            "r2": float(r2_score(actual, pred)),
        }
        m.booster_.save_model(str(MODELS_ADV_DIR / f"stats_v3_{tgt}.txt"))
        artifacts[tgt] = True
    print(f"   {'stat':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    for tgt, m in metrics.items():
        print(f"   {tgt:<25} {m['mae']:>8.3f} {m['rmse']:>8.3f} {m['r2']:>8.4f}")
    joblib.dump(scaler, MODELS_ADV_DIR / "stats_v3_scaler.joblib")
    joblib.dump(encoder, MODELS_ADV_DIR / "stats_v3_encoder.joblib")
    joblib.dump({
        "num_cols": num_cols, "cat_cols": cat_cols, "input_dim": X.shape[1],
        "targets": targets, "metrics": metrics,
    }, MODELS_ADV_DIR / "stats_v3_meta.joblib")
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=["games", "duration", "nsets", "stats", "all"])
    parser.add_argument("--min-year", type=int, default=2003)
    parser.add_argument("--force-features", action="store_true",
                        help="Force rebuild of feature cache")
    args = parser.parse_args()

    print("=" * 70)
    print(f"ATP ADVANCED – TRAINING V3   device={DEVICE}  amp={USE_AMP}")
    print(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    set_seed()
    MODELS_ADV_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_ADV_DIR.mkdir(parents=True, exist_ok=True)

    df = load_or_build_features(force=args.force_features)
    df["Date"] = pd.to_datetime(df["Date"])
    df = filter_post_stats(df, min_year=args.min_year)
    print(f"   After year filter (>= {args.min_year}): {len(df):,} matches")

    all_metrics: Dict[str, Dict] = {
        "device": str(DEVICE), "amp": USE_AMP,
        "min_year": args.min_year,
        "n_matches_total": int(len(df)),
        "timestamp": datetime.now().isoformat(),
    }

    if args.model in ("games", "all"):
        all_metrics["games"] = train_games_v3(df)
    if args.model in ("duration", "all"):
        all_metrics["duration"] = train_duration_v3(df)
    if args.model in ("nsets", "all"):
        all_metrics["nsets"] = train_nsets_v3(df)
    if args.model in ("stats", "all"):
        all_metrics["stats"] = train_stats_v3(df)

    out = MODELS_ADV_DIR / "metrics_v3.json"
    with open(out, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nMetrics saved → {out}")
    print(f"Done: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()

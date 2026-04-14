#!/usr/bin/env python3
"""
ATP Tennis Match Predictor – Training & Evaluation

Usage:
    python -m src.train
"""

from __future__ import annotations

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader

from .config import (BATCH_SIZE, DATE_END, DATE_START, LR, MAX_EPOCHS,
                     MODEL_PATH, PATIENCE, POS_WEIGHT_OFFSET, RANDOM_SEED,
                     SCHEDULER_FACTOR, SCHEDULER_MODE, SCHEDULER_PATIENCE,
                     TRAIN_FRAC, VAL_FRAC, WEIGHT_DECAY,
                     DATASET_PATH, SCALER_PATH, ENCODER_PATH, FIGURES_DIR,
                     NUM_COLS, CAT_COLS)
from .features import build_features, upset_signal_score
from .model import EnhancedMLP, TennisDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_splits(n: int, seed: int = RANDOM_SEED):
    """Return (train_idx, val_idx, test_idx) with a random shuffle."""
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    t0 = int(TRAIN_FRAC * n)
    t1 = int((TRAIN_FRAC + VAL_FRAC) * n)
    return idx[:t0], idx[t0:t1], idx[t1:]


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(out).detach().cpu().numpy())
        all_labels.extend(yb.cpu().numpy())
    return total_loss / len(loader), roc_auc_score(all_labels, all_preds)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            total_loss += criterion(out, yb).item()
            all_preds.extend(torch.sigmoid(out).cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    preds = np.array(all_preds).flatten()
    labels = np.array(all_labels).flatten()
    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, preds > 0.5)
    return total_loss / len(loader), auc, acc, preds, labels


def feature_importance(model, X_val, feature_names, device, n_samples=1000):
    """Permutation-based feature importance."""
    model.eval()
    X_tensor = torch.tensor(X_val[:n_samples], dtype=torch.float32).to(device)
    baseline = torch.sigmoid(model(X_tensor)).mean().item()
    importances = []
    for i, name in enumerate(feature_names):
        X_perm = X_val[:n_samples].copy()
        X_perm[:, i] = np.random.permutation(X_perm[:, i])
        X_perm_t = torch.tensor(X_perm, dtype=torch.float32).to(device)
        perm_pred = torch.sigmoid(model(X_perm_t)).mean().item()
        importances.append((name, abs(baseline - perm_pred)))
    return sorted(importances, key=lambda x: x[1], reverse=True)


# ── Betting strategy ROI ─────────────────────────────────────────────────────

def calc_roi(df_sub, bet_on_model=True):
    """Return (roi, accuracy, n_bets) for a subset."""
    valid = df_sub[(df_sub["Odd_1"] > 0) & (df_sub["Odd_2"] > 0)].copy()
    if len(valid) == 0:
        return 0.0, 0.0, 0
    total_stake = len(valid)
    total_returns = 0.0
    wins = 0
    for _, row in valid.iterrows():
        bet_p1 = (row["pred_prob"] > 0.5) if bet_on_model else (row["Odd_1"] < row["Odd_2"])
        p1_won = row["p1_win"] == 1
        if bet_p1:
            if p1_won:
                total_returns += row["Odd_1"]; wins += 1
        else:
            if not p1_won:
                total_returns += row["Odd_2"]; wins += 1
    roi = ((total_returns - total_stake) / total_stake) * 100 if total_stake else 0
    return roi, wins / len(valid), len(valid)


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("ATP TENNIS MATCH PREDICTOR – TRAINING & ANALYSIS")
    print("=" * 80)
    print(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("1) Loading data...")
    df = pd.read_csv(DATASET_PATH, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    start, end = pd.to_datetime(DATE_START), pd.to_datetime(DATE_END)
    df = df[(df["Date"] >= start) & (df["Date"] <= end)].copy().sort_values("Date")
    print(f"   Matches loaded: {len(df):,}  ({df['Date'].min().date()} → {df['Date'].max().date()})")

    # ── 2. Feature engineering ───────────────────────────────────────────
    print("\n2) Feature engineering...")
    df = build_features(df)
    df["p1_win"] = (df["Winner"] == df["Player_1"]).astype(int)

    # ── 3. Split ─────────────────────────────────────────────────────────
    print("\n3) Splitting data...")
    train_i, val_i, test_i = make_splits(len(df))
    print(f"   Train: {len(train_i):,}  |  Val: {len(val_i):,}  |  Test: {len(test_i):,}")

    # ── 4. Prepare arrays ────────────────────────────────────────────────
    print("\n4) Preparing arrays...")
    X_num = df[NUM_COLS].fillna(0).values

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = encoder.fit_transform(df[CAT_COLS].fillna("NA"))

    scaler = StandardScaler()
    scaler.fit(X_num[train_i])
    X_num = scaler.transform(X_num)

    X = np.hstack([X_num, X_cat])
    y = df["p1_win"].fillna(0).astype(int).values
    print(f"   Num features: {len(NUM_COLS)}  |  Cat (one-hot): {X_cat.shape[1]}  |  Total: {X.shape[1]}")

    # ── 5. DataLoaders ───────────────────────────────────────────────────
    train_loader = DataLoader(TennisDataset(X, y, train_i), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TennisDataset(X, y, val_i), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TennisDataset(X, y, test_i), batch_size=BATCH_SIZE)

    # ── 6. Model & training ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n5) Training on {device}...")
    model = EnhancedMLP(X.shape[1]).to(device)

    pos = np.sum(y[train_i])
    neg = len(y[train_i]) - pos
    pw = (neg / pos + POS_WEIGHT_OFFSET) if pos > 0 else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=SCHEDULER_MODE, patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR,
    )

    best_auc, wait, state_best = 0.0, 0, None
    train_losses, val_losses, train_aucs, val_aucs = [], [], [], []

    for epoch in range(1, MAX_EPOCHS + 1):
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, device)
        vl, va, vacc, _, _ = evaluate(model, val_loader, criterion, device)
        train_losses.append(tl); val_losses.append(vl)
        train_aucs.append(ta); val_aucs.append(va)
        scheduler.step(va)

        if va > best_auc:
            best_auc, wait = va, 0
            state_best = model.state_dict()
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"   Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"   Epoch {epoch}: Train AUC={ta:.4f}  Val AUC={va:.4f}  Val Acc={vacc:.4f}")

    model.load_state_dict(state_best)
    torch.save(state_best, MODEL_PATH)

    # ── 7. Test evaluation ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("6) TEST SET EVALUATION")
    print("=" * 80)

    _, test_auc, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    pred_bin = (test_preds > 0.5).astype(int)
    lab_int = test_labels.astype(int)

    print(f"   AUC-ROC:    {test_auc:.4f}")
    print(f"   Accuracy:   {test_acc:.4f}")
    print(f"   Precision:  {precision_score(lab_int, pred_bin):.4f}")
    print(f"   Recall:     {recall_score(lab_int, pred_bin):.4f}")
    print(f"   F1:         {f1_score(lab_int, pred_bin):.4f}")
    print(f"   Brier:      {brier_score_loss(lab_int, test_preds):.4f}")
    print(f"   Log-Loss:   {log_loss(lab_int, test_preds):.4f}")

    cm = confusion_matrix(lab_int, pred_bin)
    print(f"   Confusion:  TN={cm[0,0]:,} FP={cm[0,1]:,} FN={cm[1,0]:,} TP={cm[1,1]:,}")

    # ── 8. Feature importance ────────────────────────────────────────────
    print("\n7) Feature importance (permutation)...")
    feat_names = list(NUM_COLS) + list(encoder.get_feature_names_out(CAT_COLS))
    fi = feature_importance(model, X[val_i], feat_names, device)
    for name, imp in fi[:10]:
        print(f"   {name:30s} {imp:.6f}")

    # ── 9. Surface analysis ──────────────────────────────────────────────
    print("\n8) Accuracy by surface...")
    df_test = df.iloc[test_i].copy()
    df_test["pred_prob"] = test_preds
    df_test["pred_class"] = pred_bin
    df_test["correct"] = (df_test["pred_class"] == df_test["p1_win"]).astype(int)
    df_test["min_odds"] = np.minimum(df_test["Odd_1"], df_test["Odd_2"])

    for surf in df_test["Surface"].unique():
        sub = df_test[df_test["Surface"] == surf]
        if len(sub) >= 50:
            print(f"   {surf:10s}  n={len(sub):5,}  acc={sub['correct'].mean():.2%}")

    # ── 10. Player analysis ──────────────────────────────────────────────
    print("\n9) Player predictability (top 10 easiest, min 20 matches)...")
    from collections import defaultdict
    pstats = defaultdict(lambda: {"m": 0, "c": 0})
    for _, row in df_test.iterrows():
        for p in (row["Player_1"], row["Player_2"]):
            pstats[p]["m"] += 1
            pstats[p]["c"] += row["correct"]

    player_rows = [{"Player": p, "Matches": s["m"], "Accuracy": s["c"] / s["m"]}
                   for p, s in pstats.items() if s["m"] >= 20 and str(p).strip()]
    player_df = pd.DataFrame(player_rows).sort_values("Accuracy", ascending=False)
    print(player_df.head(10).to_string(index=False))

    # ── 11. Betting strategies ───────────────────────────────────────────
    print("\n10) Betting strategy ROI...")
    df_test["confidence"] = np.abs(df_test["pred_prob"] - 0.5) * 2
    df_test["upset_signal"] = (
        ((df_test["pred_prob"] > 0.5) != (df_test["Odd_1"] < df_test["Odd_2"]))
    )
    df_test["upset_score"] = df_test.apply(upset_signal_score, axis=1)

    strategies = []
    for conf in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        sub = df_test[df_test["confidence"] >= conf]
        roi, acc, n = calc_roi(sub)
        strategies.append({"Strategy": f"Confidence >= {conf:.0%}", "Bets": n, "Accuracy": acc, "ROI": roi})

    upset_only = df_test[df_test["upset_signal"]]
    roi, acc, n = calc_roi(upset_only)
    strategies.append({"Strategy": "Upset signals only", "Bets": n, "Accuracy": acc, "ROI": roi})

    strat_df = pd.DataFrame(strategies).sort_values("ROI", ascending=False)
    print(strat_df.to_string(index=False))

    # ── 12. Year-by-year validation ──────────────────────────────────────
    print("\n11) Year-by-year validation...")
    df_test["Year"] = df_test["Date"].dt.year
    for year in sorted(df_test["Year"].unique()):
        sub = df_test[df_test["Year"] == year]
        if len(sub) >= 50:
            roi, acc, n = calc_roi(sub)
            print(f"   {year}: {n:5,} bets  acc={acc:.2%}  roi={roi:+.1f}%")

    # ── 13. Save artifacts ───────────────────────────────────────────────
    print("\n12) Saving artifacts...")
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"   {SCALER_PATH}")
    print(f"   {ENCODER_PATH}")
    print(f"   {MODEL_PATH}")

    # ── 14. Plots ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(train_losses, label="Train")
    axes[0, 0].plot(val_losses, label="Val")
    axes[0, 0].set_title("Loss"); axes[0, 0].legend()

    axes[0, 1].plot(train_aucs, label="Train")
    axes[0, 1].plot(val_aucs, label="Val")
    axes[0, 1].set_title("AUC"); axes[0, 1].legend()

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0])
    axes[1, 0].set_title("Confusion Matrix")

    frac_pos, mean_pred = calibration_curve(test_labels, test_preds, n_bins=10)
    axes[1, 1].plot([0, 1], [0, 1], "k--")
    axes[1, 1].plot(mean_pred, frac_pos, "s-")
    axes[1, 1].set_title("Calibration Curve")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "training_analysis.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"   {fig_path}")

    print(f"\nDone: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 80)


if __name__ == "__main__":
    main()

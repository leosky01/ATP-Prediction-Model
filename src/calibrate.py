#!/usr/bin/env python3
"""
ATP Tennis Match Predictor – Calibration & Balanced Optimisation

Applies isotonic regression calibration (separate for favourites and underdogs),
computes dynamic prediction thresholds, and evaluates balanced vs conservative
strategies.

Usage:
    python -m src.calibrate
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from .config import (CAT_COLS, DATE_END, DATE_START, DATASET_PATH,
                     ENCODER_PATH, FIGURES_DIR, MODEL_PATH, NUM_COLS,
                     OPTIMIZATION_CONFIG_PATH, SCALER_PATH)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Upset signal ─────────────────────────────────────────────────────────────

def upset_signal_score(row) -> float:
    s = 0.0
    fd = row.get("win_rate_diff", 0)
    if fd > 0.2:   s += 0.3
    elif fd > 0.1: s += 0.15
    if row.get("p1_streak", 0) > 3: s += 0.2
    if row.get("h2h_win_rate", 0.5) > 0.6 and row.get("h2h_matches", 0) >= 3: s += 0.25
    r1, r2 = max(row.get("Rank_1", 100), 1), max(row.get("Rank_2", 100), 1)
    o1, o2 = max(row.get("Odd_1", 1.5), 1.01), max(row.get("Odd_2", 1.5), 1.01)
    if (o2 / o1) < (r2 / r1) * 0.8: s += 0.15
    return min(s, 1.0)


# ── Dynamic threshold ────────────────────────────────────────────────────────

def dynamic_threshold(rank_diff: float, surface: str,
                      segnali_upset: float,
                      upset_bias_factor: float = 0.05) -> float:
    """Threshold varies by ranking gap and upset signals (0.22 – 0.55)."""
    base = 0.38
    if abs(rank_diff) > 50:
        base = 0.30
    elif abs(rank_diff) > 20:
        base = 0.35
    else:
        base = 0.42

    surface_adj = {"Clay": 0.03, "Grass": -0.02}.get(surface, 0)
    upset_adj = segnali_upset * upset_bias_factor
    return float(np.clip(base + surface_adj + upset_adj, 0.22, 0.55))


# ── Main calibration pipeline ───────────────────────────────────────────────

def run_calibration():
    print("=" * 80)
    print("CALIBRATION & BALANCED OPTIMISATION")
    print("=" * 80)
    print(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data & model ────────────────────────────────────────────────
    print("Loading data and model...")
    df = pd.read_csv(DATASET_PATH, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[(df["Date"] >= DATE_START) & (df["Date"] <= DATE_END)].sort_values("Date")

    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    n_num = len(NUM_COLS)
    n_cat = sum(len(c) for c in encoder.categories_)
    model_dim = n_num + n_cat
    model = __import__("src.model", fromlist=["EnhancedMLP"]).EnhancedMLP(model_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    # ── Compute raw predictions on full dataset ──────────────────────────
    print("Computing raw predictions...")
    from .features import build_features
    df = build_features(df)
    df["p1_win"] = (df["Winner"] == df["Player_1"]).astype(int)

    X_num = scaler.transform(df[NUM_COLS].fillna(0).values)
    X_cat = encoder.transform(df[CAT_COLS].fillna("NA").values)
    X = np.hstack([X_num, X_cat])
    X_t = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_t).numpy().flatten()
    raw_probs = 1 / (1 + np.exp(-logits))  # sigmoid

    df["raw_prob"] = raw_probs

    # ── Isotonic calibration (separate for favourites / underdogs) ───────
    print("Isotonic calibration...")
    df["is_fav_p1"] = df["Odd_1"] < df["Odd_2"]
    df["segnali_upset"] = df.apply(upset_signal_score, axis=1)

    fav_mask = df["is_fav_p1"]
    und_mask = ~fav_mask

    iso_fav = IsotonicRegression(out_of_bounds="clip")
    iso_und = IsotonicRegression(out_of_bounds="clip")

    df.loc[fav_mask, "cal_prob"] = iso_fav.fit_transform(
        df.loc[fav_mask, "raw_prob"], df.loc[fav_mask, "p1_win"])
    df.loc[und_mask, "cal_prob"] = iso_und.fit_transform(
        df.loc[und_mask, "raw_prob"], df.loc[und_mask, "p1_win"])

    # ── Strategy evaluation ──────────────────────────────────────────────
    print("\nStrategy comparison...")

    # Baseline: raw prob > 0.5
    baseline_pred = (df["raw_prob"] > 0.5).astype(int)
    baseline_acc = accuracy_score(df["p1_win"], baseline_pred)

    # Conservative: calibrated > 0.55
    cons_pred = (df["cal_prob"] > 0.55).astype(int)
    cons_acc = accuracy_score(df["p1_win"], cons_pred)

    # Balanced: dynamic threshold
    df["threshold"] = df.apply(
        lambda r: dynamic_threshold(r["rank_diff"], r["Surface"], r["segnali_upset"]), axis=1)
    bal_pred = (df["cal_prob"] > df["threshold"]).astype(int)
    bal_acc = accuracy_score(df["p1_win"], bal_pred)

    # Upset metrics
    def upset_metrics(preds, labels, favs):
        fp = ((preds == 1) & (labels == 0) & favs).sum()
        upset_recall = ((preds == 1) & (labels == 1) & (~favs)).sum() / max((~favs & (labels == 1)).sum(), 1)
        return int(fp), float(upset_recall)

    fp_b, ur_b = upset_metrics(baseline_pred.values, df["p1_win"].values, df["is_fav_p1"].values)
    fp_c, ur_c = upset_metrics(cons_pred.values, df["p1_win"].values, df["is_fav_p1"].values)
    fp_bl, ur_bl = upset_metrics(bal_pred.values, df["p1_win"].values, df["is_fav_p1"].values)

    print(f"  Baseline:     acc={baseline_acc:.2%}  FP={fp_b:,}  upset_recall={ur_b:.2%}")
    print(f"  Conservative: acc={cons_acc:.2%}  FP={fp_c:,}  upset_recall={ur_c:.2%}")
    print(f"  Balanced:     acc={bal_acc:.2%}  FP={fp_bl:,}  upset_recall={ur_bl:.2%}")

    # ── Strong-signal subset ─────────────────────────────────────────────
    strong = df[df["segnali_upset"] > 0.4]
    if len(strong) >= 50:
        strong_acc = accuracy_score(strong["p1_win"], (strong["cal_prob"] > 0.5).astype(int))
        print(f"  Strong upset signal (>0.4): n={len(strong):,}  acc={strong_acc:.2%}")

    # ── Save config ──────────────────────────────────────────────────────
    config = {
        "iso_fav": iso_fav,
        "iso_und": iso_und,
        "baseline_acc": baseline_acc,
        "conservative_acc": cons_acc,
        "balanced_acc": bal_acc,
        "date": str(datetime.now()),
    }
    joblib.dump(config, OPTIMIZATION_CONFIG_PATH)
    print(f"\nSaved config → {OPTIMIZATION_CONFIG_PATH}")

    # ── Plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Accuracy comparison
    names = ["Baseline", "Conservative", "Balanced"]
    accs = [baseline_acc, cons_acc, bal_acc]
    axes[0, 0].bar(names, accs, color=["steelblue", "orange", "green"])
    axes[0, 0].set_title("Accuracy Comparison")
    axes[0, 0].set_ylim(0.5, max(accs) + 0.05)

    # False positives
    axes[0, 1].bar(names, [fp_b, fp_c, fp_bl], color=["steelblue", "orange", "green"])
    axes[0, 1].set_title("False Positives")

    # Upset recall
    axes[0, 2].bar(names, [ur_b, ur_c, ur_bl], color=["steelblue", "orange", "green"])
    axes[0, 2].set_title("Upset Recall")

    # Distribution pre/post calibration
    axes[1, 0].hist(df["raw_prob"], bins=50, alpha=0.5, label="Raw")
    axes[1, 0].hist(df["cal_prob"], bins=50, alpha=0.5, label="Calibrated")
    axes[1, 0].set_title("Probability Distribution")
    axes[1, 0].legend()

    # Upset signal distribution
    axes[1, 1].hist(df["segnali_upset"], bins=30, edgecolor="black")
    axes[1, 1].axvline(0.4, color="red", linestyle="--", label="threshold=0.4")
    axes[1, 1].set_title("Upset Signal Distribution")
    axes[1, 1].legend()

    # Summary table
    axes[1, 2].axis("off")
    table_data = [[n, f"{a:.2%}"] for n, a in zip(names, accs)]
    axes[1, 2].table(cellText=table_data, colLabels=["Strategy", "Accuracy"],
                     loc="center", cellLoc="center")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "calibration_analysis.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved figure → {fig_path}")

    print(f"\nDone: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    run_calibration()

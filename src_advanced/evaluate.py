#!/usr/bin/env python3
"""
ATP Advanced Prediction System – Evaluation & Metrics

Usage:
    python -m src_advanced.evaluate --model games
    python -m src_advanced.evaluate --model all
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE, MERGED_DATASET_PATH, MODELS_ADV_DIR,
)
from .feature_engineering import build_advanced_features, get_feature_columns
from .models.games_predictor import GamesPredictor
from .models.score_predictor import ScorePredictor3, ScorePredictor5
from .models.duration_predictor import DurationPredictor
from .models.stats_predictor import StatsPredictor, StatsDataset


# ── Generic loader ───────────────────────────────────────────────────────────

def _load_artifacts(name: str, device: torch.device):
    """Load model, scaler, encoder for a given model name."""
    meta = joblib.load(MODELS_ADV_DIR / f"{name}_meta.joblib")
    input_dim = meta["input_dim"]
    scaler = joblib.load(MODELS_ADV_DIR / f"{name}_scaler.joblib")
    encoder = joblib.load(MODELS_ADV_DIR / f"{name}_encoder.joblib")

    model_map = {
        "games": GamesPredictor,
        "score_bo3": ScorePredictor3,
        "score_bo5": ScorePredictor5,
        "duration": DurationPredictor,
        "stats": StatsPredictor,
    }
    model_cls = model_map[name]
    model = model_cls(input_dim).to(device)

    state = torch.load(MODELS_ADV_DIR / f"{name}_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, scaler, encoder, input_dim


def _prepare_eval_features(df: pd.DataFrame, scaler, encoder):
    """Apply saved scaler and encoder to DataFrame."""
    num_cols, cat_cols = get_feature_columns(df)
    X_num = df[num_cols].values.astype(np.float32)

    # Use training medians if available, else fillna(0)
    if hasattr(scaler, 'train_medians_'):
        for j in range(X_num.shape[1]):
            mask = np.isnan(X_num[:, j])
            if mask.any():
                X_num[mask, j] = scaler.train_medians_[j]
    else:
        X_num = np.nan_to_num(X_num, nan=0.0)

    X_num = scaler.transform(X_num).astype(np.float32)

    if cat_cols and encoder.categories_:
        X_cat = encoder.transform(df[cat_cols].fillna("NA").astype(str))
    else:
        X_cat = np.zeros((len(df), 0), dtype=np.float32)

    return np.hstack([X_num, X_cat])


# ── Evaluation functions ────────────────────────────────────────────────────

def evaluate_games(df: pd.DataFrame):
    """Evaluate GamesPredictor: AUC-ROC, Accuracy, Brier Score."""
    print("\n" + "=" * 60)
    print("EVALUATING: GamesPredictor")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler, encoder, input_dim = _load_artifacts("games", device)

    valid = df["over_under"].notna() & df["total_games"].notna()
    df_sub = df[valid].copy().reset_index(drop=True)
    if len(df_sub) == 0:
        print("   No valid data for evaluation")
        return

    X = _prepare_eval_features(df_sub, scaler, encoder)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    all_ou_probs, all_total_preds = [], []
    with torch.no_grad():
        for i in range(0, len(X_t), BATCH_SIZE):
            batch = X_t[i:i + BATCH_SIZE]
            ou_logit, total_pred = model(batch)
            all_ou_probs.extend(torch.sigmoid(ou_logit).cpu().numpy().flatten())
            all_total_preds.extend(total_pred.cpu().numpy().flatten())

    ou_probs = np.array(all_ou_probs)
    total_preds = np.array(all_total_preds)
    ou_labels = df_sub["over_under"].values
    total_labels = df_sub["total_games"].values

    # Metrics
    pred_bin = (ou_probs > 0.5).astype(int)
    print(f"\n   Over/Under Classification:")
    print(f"   AUC-ROC:     {roc_auc_score(ou_labels, ou_probs):.4f}")
    print(f"   Accuracy:    {accuracy_score(ou_labels, pred_bin):.4f}")
    print(f"   Brier Score: {brier_score_loss(ou_labels, ou_probs):.4f}")
    print(f"   F1 Score:    {f1_score(ou_labels, pred_bin):.4f}")

    cm = confusion_matrix(ou_labels, pred_bin)
    print(f"   Confusion:   TN={cm[0,0]:,} FP={cm[0,1]:,} FN={cm[1,0]:,} TP={cm[1,1]:,}")

    valid_total = ~np.isnan(total_labels)
    if valid_total.any():
        print(f"\n   Total Games Regression:")
        print(f"   MAE:  {mean_absolute_error(total_labels[valid_total], total_preds[valid_total]):.2f}")
        print(f"   RMSE: {np.sqrt(mean_squared_error(total_labels[valid_total], total_preds[valid_total])):.2f}")
        print(f"   R^2:  {r2_score(total_labels[valid_total], total_preds[valid_total]):.4f}")


def evaluate_score(df: pd.DataFrame):
    """Evaluate ScorePredictor: Accuracy, F1 macro, Confusion Matrix."""
    print("\n" + "=" * 60)
    print("EVALUATING: ScorePredictor")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for best_of, name, class_names in [
        (3, "score_bo3", ["2-0", "2-1"]),
        (5, "score_bo5", ["3-0", "3-1", "3-2"]),
    ]:
        model_path = MODELS_ADV_DIR / f"{name}_model.pt"
        if not model_path.exists():
            print(f"\n   {name}: model not found, skipping")
            continue

        print(f"\n--- Best of {best_of} ---")
        model, scaler, encoder, input_dim = _load_artifacts(name, device)

        score_to_class = {s: i for i, s in enumerate(class_names)}
        valid = (df["best_of"].fillna(3) == best_of) & df["set_score"].notna()
        df_sub = df[valid].copy().reset_index(drop=True)
        df_sub["_class"] = df_sub["set_score"].map(score_to_class)
        df_sub = df_sub[df_sub["_class"].notna()].reset_index(drop=True)

        if len(df_sub) == 0:
            print("   No valid data")
            continue

        X = _prepare_eval_features(df_sub, scaler, encoder)
        X_t = torch.tensor(X, dtype=torch.float32).to(device)

        all_preds = []
        with torch.no_grad():
            for i in range(0, len(X_t), BATCH_SIZE):
                batch = X_t[i:i + BATCH_SIZE]
                logits = model(batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)

        preds = np.array(all_preds)
        labels = df_sub["_class"].values.astype(int)

        print(f"   Accuracy: {accuracy_score(labels, preds):.4f}")
        print(f"   F1 Macro: {f1_score(labels, preds, average='macro'):.4f}")
        cm = confusion_matrix(labels, preds)
        print(f"   Confusion Matrix:")
        print(f"   {cm}")


def evaluate_duration(df: pd.DataFrame):
    """Evaluate DurationPredictor: RMSE, MAE, R^2, within-15-min accuracy."""
    print("\n" + "=" * 60)
    print("EVALUATING: DurationPredictor")
    print("=" * 60)

    model_path = MODELS_ADV_DIR / "duration_model.pt"
    if not model_path.exists():
        print("   Model not found, skipping")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler, encoder, input_dim = _load_artifacts("duration", device)

    valid = df["minutes"].notna()
    df_sub = df[valid].copy().reset_index(drop=True)
    if len(df_sub) == 0:
        print("   No valid data")
        return

    X = _prepare_eval_features(df_sub, scaler, encoder)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    all_means, all_logvars = [], []
    with torch.no_grad():
        for i in range(0, len(X_t), BATCH_SIZE):
            batch = X_t[i:i + BATCH_SIZE]
            mean, log_var = model(batch)
            all_means.extend(mean.cpu().numpy().flatten())
            all_logvars.extend(log_var.cpu().numpy().flatten())

    means = np.array(all_means)
    logvars = np.array(all_logvars)
    stds = np.exp(0.5 * logvars)
    labels = df_sub["minutes"].values

    print(f"\n   RMSE:          {np.sqrt(mean_squared_error(labels, means)):.2f} min")
    print(f"   MAE:           {mean_absolute_error(labels, means):.2f} min")
    print(f"   R^2:           {r2_score(labels, means):.4f}")
    print(f"   Within 15 min: {(np.abs(means - labels) <= 15).mean():.2%}")
    print(f"   Within 30 min: {(np.abs(means - labels) <= 30).mean():.2%}")

    # Coverage of 95% confidence interval
    ci_lower = means - 1.96 * stds
    ci_upper = means + 1.96 * stds
    coverage = ((labels >= ci_lower) & (labels <= ci_upper)).mean()
    print(f"   95% CI coverage: {coverage:.2%}")
    print(f"   Mean predicted std: {stds.mean():.2f} min")


def evaluate_stats(df: pd.DataFrame):
    """Evaluate StatsPredictor: MAE, RMSE, R^2 per statistic."""
    print("\n" + "=" * 60)
    print("EVALUATING: StatsPredictor")
    print("=" * 60)

    model_path = MODELS_ADV_DIR / "stats_model.pt"
    if not model_path.exists():
        print("   Model not found, skipping")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler, encoder, input_dim = _load_artifacts("stats", device)

    # Prepare derived columns
    df_sub = df.copy()
    if "p1_1st_pct" not in df_sub.columns and "p1_1stIn" in df_sub.columns:
        df_sub["p1_1st_pct"] = df_sub["p1_1stIn"] / df_sub["p1_svpt"].replace(0, np.nan)
        df_sub["p2_1st_pct"] = df_sub["p2_1stIn"] / df_sub["p2_svpt"].replace(0, np.nan)
    if "p1_1st_win_pct" not in df_sub.columns and "p1_1stWon" in df_sub.columns:
        df_sub["p1_1st_win_pct"] = df_sub["p1_1stWon"] / df_sub["p1_1stIn"].replace(0, np.nan)
        df_sub["p2_1st_win_pct"] = df_sub["p2_1stWon"] / df_sub["p2_1stIn"].replace(0, np.nan)

    target_groups = {
        "aces": StatsDataset.ACE_COLS,
        "df": StatsDataset.DF_COLS,
        "bp": StatsDataset.BP_COLS,
        "serve": StatsDataset.SERVE_COLS,
    }

    X = _prepare_eval_features(df_sub, scaler, encoder)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    all_preds = {}
    with torch.no_grad():
        for i in range(0, len(X_t), BATCH_SIZE):
            batch = X_t[i:i + BATCH_SIZE]
            preds = model(batch)
            for head_name, pred_tensor in preds.items():
                if head_name not in all_preds:
                    all_preds[head_name] = []
                all_preds[head_name].append(pred_tensor.cpu().numpy())

    for head_name, pred_arrays in all_preds.items():
        all_preds[head_name] = np.concatenate(pred_arrays, axis=0)

    print(f"\n   {'Stat':<25} {'MAE':>8} {'RMSE':>8} {'R^2':>8}")
    print(f"   {'-'*25} {'-'*8} {'-'*8} {'-'*8}")

    for head_name, cols in target_groups.items():
        if head_name not in all_preds:
            continue
        pred_matrix = all_preds[head_name]
        for j, col in enumerate(cols):
            if col not in df_sub.columns:
                continue
            actual = df_sub[col].values
            predicted = pred_matrix[:, j]
            valid = ~np.isnan(actual)
            if valid.sum() < 10:
                continue
            mae = mean_absolute_error(actual[valid], predicted[valid])
            rmse = np.sqrt(mean_squared_error(actual[valid], predicted[valid]))
            r2 = r2_score(actual[valid], predicted[valid])
            print(f"   {col:<25} {mae:>8.3f} {rmse:>8.3f} {r2:>8.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ATP Advanced Evaluation")
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["games", "score", "duration", "stats", "all"],
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ATP ADVANCED PREDICTION SYSTEM – EVALUATION")
    print(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    if not MERGED_DATASET_PATH.exists():
        print("ERROR: Merged dataset not found. Run training first.")
        return

    df = pd.read_csv(MERGED_DATASET_PATH, parse_dates=["Date"])
    print(f"   Loaded {len(df):,} matches")

    df = build_advanced_features(df)

    if args.model in ("games", "all"):
        evaluate_games(df)
    if args.model in ("score", "all"):
        evaluate_score(df)
    if args.model in ("duration", "all"):
        evaluate_duration(df)
    if args.model in ("stats", "all"):
        evaluate_stats(df)

    print(f"\nDone: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()

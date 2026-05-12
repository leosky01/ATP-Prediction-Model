#!/usr/bin/env python3
"""
ATP Upset Detector — ML model to predict match upsets.

Replaces rule-based upset detection with a LightGBM model that learns
which patterns foreshadow an upset, using the 188 V3 features + 11
new upset-specific features.

Target: is_upset = 1 if underdog (by odds) wins the match.
Models are trained per best_of (BO3 / BO5) with class-imbalance handling.

Usage:
    python -m src_advanced.upset_detector --bo both
    python -m src_advanced.upset_detector --bo 5

Programmatically:
    from src_advanced.upset_detector import UpsetPredictor
    pred = UpsetPredictor()
    result = pred.predict(features_dict, best_of=3)

Simple API (from notebook):
    from src_advanced.upset_detector import UpsetLivePredictor
    pred = UpsetLivePredictor()
    result = pred.predici_upset(
        "Sinner J.", "Alcaraz C.", rank1=1, rank2=2,
        odds1=1.90, odds2=1.95, surface="Clay",
        tournament="Slam", date_str="2026-05-15", best_of=5,
    )
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from difflib import get_close_matches

from .config import CAT_COLS_ADVANCED, MODELS_ADV_DIR, RANDOM_SEED
from .feature_cache import CACHE_PATH, load_or_build_features
from .feature_engineering import get_feature_columns
from .train_v3 import (
    _build_swap_index,
    apply_swap,
    filter_post_stats,
    prepare_features,
    temporal_split,
)

# ── 11 upset-specific feature names ──────────────────────────────────────────

UPSET_FEATURES = [
    "market_rank_disagreement",
    "underdog_form_trend",
    "favorite_fatigue_7d",
    "favorite_fatigue_14d",
    "surface_expertise_gap",
    "underdog_ace_edge",
    "upset_h2h_signal",
    "underdog_avg_match_length",
    "odds_sensitivity",
    "rank_odds_divergence",
    "underdog_streak",
]


# ──────────────────────────────────────────────────────────────────────────────
# 1) Target definition
# ──────────────────────────────────────────────────────────────────────────────


def define_upset_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add `is_upset` and `_p1_is_underdog` columns.

    Underdog = player with higher odds (Odd_1 > Odd_2 → P1 is underdog).
    is_upset = 1 if the underdog wins.
    """
    df = df.copy()
    odds1 = pd.to_numeric(df["Odd_1"], errors="coerce")
    odds2 = pd.to_numeric(df["Odd_2"], errors="coerce")
    # Replace invalid odds (≤0) with NaN
    odds1 = odds1.where(odds1 > 0, np.nan)
    odds2 = odds2.where(odds2 > 0, np.nan)

    p1_won = (df["Winner"] == df["Player_1"]).astype(float)

    # P1 is underdog if their odds are higher
    df["_p1_is_underdog"] = (odds1 > odds2).astype(float)
    # Equal odds: mark as False (no clear underdog)
    df.loc[odds1 == odds2, "_p1_is_underdog"] = np.nan

    # is_upset: underdog wins
    df["is_upset"] = np.where(
        df["_p1_is_underdog"] == 1, p1_won,
        np.where(df["_p1_is_underdog"] == 0, 1 - p1_won, np.nan),
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2) Perspective normalization
# ──────────────────────────────────────────────────────────────────────────────


def normalize_to_underdog_perspective(
    df: pd.DataFrame, num_cols: List[str],
) -> pd.DataFrame:
    """Normalize features to underdog perspective.

    If P1 is underdog → keep as-is.
    If P2 is underdog → swap/negate features so that column i represents
    the underdog's perspective in every row.
    """
    df = df.copy()

    swap_mask = df["_p1_is_underdog"] == 0  # P2 is underdog → need swap
    keep_mask = df["_p1_is_underdog"] == 1  # P1 is underdog → keep

    # Build swap index for num_cols
    perm, sign = _build_swap_index(num_cols)

    X_num = df[num_cols].to_numpy(dtype=np.float32).copy()

    # Apply swap only to rows where P2 is underdog
    swap_idx = df.index[swap_mask].to_numpy()
    if len(swap_idx) > 0:
        X_swap = X_num[swap_mask.values if isinstance(swap_mask, pd.Series) else swap_mask]
        X_swapped = apply_swap(X_swap, perm, sign)
        X_num[swap_mask.values if isinstance(swap_mask, pd.Series) else swap_mask] = X_swapped

    # Write back
    for j, col in enumerate(num_cols):
        df[col] = X_num[:, j]

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3) Upset-specific features (11)
# ──────────────────────────────────────────────────────────────────────────────


def compute_upset_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 11 upset-specific features.

    These are already underdog-relative by construction (computed after
    perspective normalization), so no additional swap is needed.
    """
    df = df.copy()

    odds1 = pd.to_numeric(df["Odd_1"], errors="coerce").where(
        pd.to_numeric(df["Odd_1"], errors="coerce") > 0, np.nan
    )
    odds2 = pd.to_numeric(df["Odd_2"], errors="coerce").where(
        pd.to_numeric(df["Odd_2"], errors="coerce") > 0, np.nan
    )
    rank1 = pd.to_numeric(df["Rank_1"], errors="coerce")
    rank2 = pd.to_numeric(df["Rank_2"], errors="coerce")

    # After normalization, rank_diff = underdog_rank - favorite_rank (positive = gap)
    # We need to detect who is the underdog BEFORE normalization for some raw features.
    # But we call this after normalization, so features are from underdog perspective.
    # Use rank_diff directly (positive = underdog has worse rank = expected)
    rank_diff = pd.to_numeric(df.get("rank_diff", np.nan), errors="coerce")

    # 1) market_rank_disagreement: odds and ranking disagree
    # Odds favorite ≠ rank favorite → abs(sign(odds_diff) - sign(rank_diff))
    odds_diff = odds1 - odds2
    rank_diff_raw = rank1 - rank2
    odds_sign = np.sign(odds_diff)
    rank_sign = np.sign(rank_diff_raw)
    df["market_rank_disagreement"] = (odds_sign != rank_sign).astype(float)

    # 2) underdog_form_trend: recent improvement (w10 vs w50 win rate)
    # After normalization, win_diff_w10 > 0 means underdog has been better recently
    w10 = df.get("win_diff_w10", pd.Series(0, index=df.index))
    w50 = df.get("win_diff_w50", pd.Series(0, index=df.index))
    df["underdog_form_trend"] = pd.to_numeric(w10, errors="coerce") - pd.to_numeric(w50, errors="coerce")

    # 3-4) favorite_fatigue: fatigue of the favorite
    # After normalization, fatigue_diff_7d = underdog_fatigue - favorite_fatigue
    # Negative means favorite more tired. We want favorite fatigue directly.
    fat7 = pd.to_numeric(df.get("fatigue_diff_7d", 0), errors="coerce")
    fat14 = pd.to_numeric(df.get("fatigue_diff_14d", 0), errors="coerce")
    df["favorite_fatigue_7d"] = -fat7  # negate: positive = favorite more tired
    df["favorite_fatigue_14d"] = -fat14

    # 5) surface_expertise_gap: underdog's surface advantage
    # After normalization, surf_win_rate_diff = underdog_surf_wr - favorite_surf_wr
    df["surface_expertise_gap"] = pd.to_numeric(
        df.get("surf_win_rate_diff", 0), errors="coerce"
    )

    # 6) underdog_ace_edge: underdog's serving advantage
    # After normalization, ace_diff_w10 = underdog_ace - favorite_ace
    df["underdog_ace_edge"] = pd.to_numeric(
        df.get("ace_diff_w10", 0), errors="coerce"
    )

    # 7) upset_h2h_signal: underdog's H2H advantage
    # After normalization, h2h_win_rate = P1's win rate → if P1 is underdog, this is underdog H2H
    df["upset_h2h_signal"] = pd.to_numeric(
        df.get("h2h_win_rate", 0.5), errors="coerce"
    )

    # 8) underdog_avg_match_length: resilience proxy
    # After normalization, minutes_diff_w10 = underdog_avg_min - favorite_avg_min
    # Positive means underdog plays longer matches
    df["underdog_avg_match_length"] = pd.to_numeric(
        df.get("minutes_diff_w10", 0), errors="coerce"
    )

    # 9) odds_sensitivity: how competitive the market thinks the match is
    min_odds = np.minimum(odds1, odds2)
    max_odds = np.maximum(odds1, odds2)
    df["odds_sensitivity"] = min_odds / max_odds.clip(lower=0.01)

    # 10) rank_odds_divergence: big rank gap but close odds → "smart money"
    rank_gap = (rank1 - rank2).abs()
    odds_gap = (odds1 - odds2).abs()
    df["rank_odds_divergence"] = (
        np.log1p(rank_gap.clip(upper=500)) - np.log1p(odds_gap.clip(upper=20))
    )

    # 11) underdog_streak: momentum of the underdog
    # After normalization, streak_diff_w10 = underdog_streak - favorite_streak
    df["underdog_streak"] = pd.to_numeric(
        df.get("streak_diff_w10", 0), errors="coerce"
    )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4) LightGBM training
# ──────────────────────────────────────────────────────────────────────────────


def train_upset_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int = RANDOM_SEED,
    num_leaves: int = 70,
    max_depth: int = 8,
    lr: float = 0.02,
    n_estimators: int = 5000,
    early_stopping_rounds: int = 100,
    refit_on_trainval: bool = True,
):
    """Train LightGBM classifier with scale_pos_weight for class imbalance."""
    import lightgbm as lgb

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = dict(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        num_leaves=num_leaves,
        max_depth=max_depth,
        learning_rate=lr,
        min_child_samples=70,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_alpha=1.5,
        reg_lambda=0.5,
        scale_pos_weight=scale_pos_weight,
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
        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
        refit_params = params.copy()
        refit_params["n_estimators"] = int(best_iter)
        final = lgb.LGBMClassifier(**refit_params)
        final.fit(X_all, y_all)
        final._val_model_for_stack = model
        return final
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 5) Evaluation
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_upset_model(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    prefix: str = "",
) -> Dict:
    """Evaluate upset model: AUC-PR, AUC-ROC, precision@k, F1, Brier."""
    y_true = y_true.astype(float)
    y_proba = np.clip(y_proba.astype(float), 1e-6, 1 - 1e-6)

    metrics = {}
    metrics["auc_pr"] = float(average_precision_score(y_true, y_proba))
    metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba))
    metrics["brier"] = float(brier_score_loss(y_true, y_proba))

    # Precision@k (top 50, 100, 200)
    sorted_idx = np.argsort(-y_proba)
    for k in [50, 100, 200]:
        if k > len(y_true):
            continue
        top_k_true = y_true[sorted_idx[:k]]
        metrics[f"precision_at_{k}"] = float(top_k_true.mean())

    # Optimal F1 threshold
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.3, 0.7, 0.01):
        y_pred = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    metrics["f1_optimal"] = float(best_f1)
    metrics["f1_optimal_threshold"] = float(best_thr)

    # At 0.5 threshold
    y_pred_05 = (y_proba >= 0.5).astype(int)
    metrics["precision_05"] = float(precision_score(y_true, y_pred_05, zero_division=0))
    metrics["recall_05"] = float(recall_score(y_true, y_pred_05, zero_division=0))
    metrics["f1_05"] = float(f1_score(y_true, y_pred_05, zero_division=0))
    metrics["accuracy_05"] = float(accuracy_score(y_true, y_pred_05))

    # Upset rate (baseline)
    metrics["upset_rate"] = float(y_true.mean())
    metrics["n_total"] = int(len(y_true))
    metrics["n_upset"] = int(y_true.sum())

    # Print
    tag = f"[{prefix}] " if prefix else ""
    print(f"   {tag}AUC-PR={metrics['auc_pr']:.4f}  AUC-ROC={metrics['auc_roc']:.4f}  Brier={metrics['brier']:.4f}")
    print(f"   {tag}P@100={metrics.get('precision_at_100', 0):.4f}  F1(opt)={metrics['f1_optimal']:.4f}@{metrics['f1_optimal_threshold']:.2f}")
    print(f"   {tag}Upset rate={metrics['upset_rate']:.3f}  ({metrics['n_upset']}/{metrics['n_total']})")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 6) Segment analysis
# ──────────────────────────────────────────────────────────────────────────────


def segment_analysis_upset(
    df_test: pd.DataFrame,
    y_proba: np.ndarray,
    y_true: np.ndarray,
    prefix: str = "",
) -> Dict:
    """Analyze upset model performance per surface, round, tournament level, etc."""
    segments = {}
    y_proba = np.clip(y_proba, 1e-6, 1 - 1e-6)

    def _safe_auc_pr(y_t, y_p):
        try:
            return float(average_precision_score(y_t, y_p))
        except Exception:
            return float("nan")

    # By Surface
    for surf in df_test["Surface"].dropna().unique():
        mask = df_test["Surface"] == surf
        if mask.sum() < 30:
            continue
        auc_pr = _safe_auc_pr(y_true[mask], y_proba[mask])
        acc = float(accuracy_score(y_true[mask], (y_proba[mask] > 0.5).astype(int)))
        upset_rate = float(y_true[mask].mean())
        segments[f"surface_{surf}"] = {
            "auc_pr": auc_pr, "accuracy": acc,
            "n": int(mask.sum()), "upset_rate": upset_rate,
        }

    # By Round
    round_col = "Round" if "Round" in df_test.columns else "round"
    if round_col in df_test.columns:
        for rnd in df_test[round_col].dropna().unique():
            mask = df_test[round_col] == rnd
            if mask.sum() < 30:
                continue
            auc_pr = _safe_auc_pr(y_true[mask], y_proba[mask])
            acc = float(accuracy_score(y_true[mask], (y_proba[mask] > 0.5).astype(int)))
            segments[f"round_{rnd}"] = {
                "auc_pr": auc_pr, "accuracy": acc,
                "n": int(mask.sum()), "upset_rate": float(y_true[mask].mean()),
            }

    # By Tournament Level
    if "tourney_level" in df_test.columns:
        for level in df_test["tourney_level"].dropna().unique():
            mask = df_test["tourney_level"] == level
            if mask.sum() < 30:
                continue
            auc_pr = _safe_auc_pr(y_true[mask], y_proba[mask])
            acc = float(accuracy_score(y_true[mask], (y_proba[mask] > 0.5).astype(int)))
            segments[f"level_{level}"] = {
                "auc_pr": auc_pr, "accuracy": acc,
                "n": int(mask.sum()), "upset_rate": float(y_true[mask].mean()),
            }

    # By favorite strength (implied probability of favorite)
    if "favorite_strength" in df_test.columns:
        fs = df_test["favorite_strength"].values
        bins = [
            (0.45, 0.55, "competitive"),
            (0.55, 0.70, "medium_fav"),
            (0.70, 1.01, "strong_fav"),
        ]
        for lo, hi, label in bins:
            mask = (fs >= lo) & (fs < hi) & ~np.isnan(fs)
            if mask.sum() < 30:
                continue
            auc_pr = _safe_auc_pr(y_true[mask], y_proba[mask])
            acc = float(accuracy_score(y_true[mask], (y_proba[mask] > 0.5).astype(int)))
            segments[f"fav_{label}"] = {
                "auc_pr": auc_pr, "accuracy": acc,
                "n": int(mask.sum()), "upset_rate": float(y_true[mask].mean()),
            }

    # By confidence bucket
    for lo, hi, label in [(0.3, 0.4, "low"), (0.4, 0.5, "moderate"), (0.5, 0.6, "high"), (0.6, 1.0, "very_high")]:
        mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() < 20:
            continue
        upset_rate = float(y_true[mask].mean())
        segments[f"confidence_{label}"] = {
            "upset_rate": upset_rate, "n": int(mask.sum()),
            "mean_predicted": float(y_proba[mask].mean()),
        }

    # Print top segments
    tag = f"[{prefix}] " if prefix else ""
    sorted_segs = sorted(
        [(k, v) for k, v in segments.items() if "auc_pr" in v and v["n"] >= 30],
        key=lambda x: x[1].get("auc_pr", 0), reverse=True,
    )
    if sorted_segs:
        print(f"\n   {tag}Top segments:")
        for name, m in sorted_segs[:8]:
            print(f"     {name:<35s} AUC-PR={m['auc_pr']:.4f}  Acc={m['accuracy']:.4f}  N={m['n']:,}  upset={m['upset_rate']:.3f}")

    return segments


# ──────────────────────────────────────────────────────────────────────────────
# 7) Betting simulation
# ──────────────────────────────────────────────────────────────────────────────


def betting_simulation(
    y_proba: np.ndarray,
    y_true: np.ndarray,
    underdog_odds: np.ndarray,
    prefix: str = "",
) -> Dict:
    """Simulate ROI if betting on predicted upsets at various confidence thresholds.

    Parameters
    ----------
    y_proba : predicted P(upset)
    y_true : actual is_upset (0/1)
    underdog_odds : decimal odds of the underdog for each match
    """
    results = {}
    y_proba = np.clip(y_proba, 1e-6, 1 - 1e-6)

    # Remove rows with missing odds
    valid = np.isfinite(underdog_odds) & (underdog_odds > 1.0)
    if valid.sum() < 50:
        print(f"   {prefix}Not enough valid odds for betting simulation")
        return {"note": "insufficient odds data"}

    yp = y_proba[valid]
    yt = y_true[valid]
    odds = underdog_odds[valid]

    for threshold in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
        bet_mask = yp >= threshold
        n_bets = bet_mask.sum()
        if n_bets < 5:
            continue

        bet_results = yt[bet_mask]
        bet_odds = odds[bet_mask]

        # ROI: profit per unit staked
        winnings = bet_results * (bet_odds - 1)  # net profit when won
        losses = (1 - bet_results) * 1.0  # lost stake
        total_profit = winnings.sum() - losses.sum()
        total_staked = float(n_bets)
        roi = total_profit / total_staked if total_staked > 0 else 0

        win_rate = bet_results.mean()
        avg_odds = bet_odds.mean()

        key = f"threshold_{threshold:.2f}"
        results[key] = {
            "n_bets": int(n_bets),
            "win_rate": float(win_rate),
            "avg_odds": float(avg_odds),
            "roi": float(roi),
            "total_profit": float(total_profit),
        }

    # Print summary
    if results:
        tag = f"[{prefix}] " if prefix else ""
        print(f"\n   {tag}Betting simulation (bet on predicted upsets):")
        print(f"     {'threshold':>10s}  {'n_bets':>7s}  {'win_rate':>9s}  {'avg_odds':>9s}  {'ROI':>8s}")
        for key, r in results.items():
            print(f"     {key[-5:]:>10s}  {r['n_bets']:>7d}  {r['win_rate']:>9.3f}  {r['avg_odds']:>9.2f}  {r['roi']:>+8.3f}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 8) Rule-based comparison
# ──────────────────────────────────────────────────────────────────────────────


def compare_with_rule_based(
    df_test: pd.DataFrame,
    y_proba: np.ndarray,
    y_true: np.ndarray,
    prefix: str = "",
) -> Dict:
    """Compare ML model with a simple rule-based upset detector."""
    # Rule-based: flag as upset if (a) odds gap > 2.0 AND (b) rank gap > 30
    # This is a simplified version of the notebook's calcola_segnali_upset
    odds1 = pd.to_numeric(df_test.get("Odd_1", np.nan), errors="coerce")
    odds2 = pd.to_numeric(df_test.get("Odd_2", np.nan), errors="coerce")
    rank1 = pd.to_numeric(df_test.get("Rank_1", np.nan), errors="coerce")
    rank2 = pd.to_numeric(df_test.get("Rank_2", np.nan), errors="coerce")

    odds_gap = (odds1 - odds2).abs()
    rank_gap = (rank1 - rank2).abs()
    min_odds = np.minimum(odds1, odds2)

    rule_signals = (
        (odds_gap > 2.0) &
        (rank_gap > 30) &
        (min_odds < 1.5)
    ).astype(int).values

    # Remove NaN rows for fair comparison
    valid = np.isfinite(rule_signals.astype(float))
    if valid.sum() < 50:
        return {"note": "insufficient data for rule-based comparison"}

    yt = y_true[valid]
    yp = y_proba[valid]
    rs = rule_signals[valid]

    results = {}

    # Rule-based metrics
    if rs.sum() > 0:
        rule_precision = float(yt[rs == 1].mean())
        rule_recall = float(yt[rs == 1].sum() / max(yt.sum(), 1))
        results["rule_based"] = {
            "n_flagged": int(rs.sum()),
            "precision": rule_precision,
            "recall": rule_recall,
        }
    else:
        results["rule_based"] = {"n_flagged": 0, "precision": 0, "recall": 0}

    # ML model metrics at same number of predictions (top-k)
    n_rule = int(rs.sum())
    if n_rule > 0:
        ml_top_k = np.argsort(-yp)[:n_rule]
        ml_precision_at_k = float(yt[ml_top_k].mean())
        results["ml_at_same_k"] = {
            "n_flagged": n_rule,
            "precision": ml_precision_at_k,
        }

    # Print
    tag = f"[{prefix}] " if prefix else ""
    if results.get("rule_based", {}).get("n_flagged", 0) > 0:
        rb = results["rule_based"]
        ml = results.get("ml_at_same_k", {})
        print(f"\n   {tag}Rule-based vs ML comparison:")
        print(f"     Rule-based: flagged={rb['n_flagged']:,}  precision={rb['precision']:.3f}  recall={rb['recall']:.3f}")
        if ml:
            print(f"     ML (top-{n_rule}):  precision={ml['precision']:.3f}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 9) Main pipeline
# ──────────────────────────────────────────────────────────────────────────────


def train_upset_detector(df: pd.DataFrame, bo_filter: str = "both") -> Dict:
    """Full upset detector training pipeline.

    Parameters
    ----------
    df : DataFrame with V3 features (from feature cache)
    bo_filter : '3', '5', or 'both'
    """
    print("\n" + "=" * 70)
    print("UPSET DETECTOR — Training")
    print("=" * 70)

    # Define target
    print("\n1) Defining upset target...")
    df = define_upset_target(df)

    # Filter rows with valid target
    valid = df["is_upset"].notna() & df["_p1_is_underdog"].notna() & df["best_of"].notna()
    # Also require valid odds
    valid &= pd.to_numeric(df["Odd_1"], errors="coerce").notna()
    valid &= pd.to_numeric(df["Odd_2"], errors="coerce").notna()
    df_sub = df[valid].copy().reset_index(drop=True)
    print(f"   Valid matches: {len(df_sub):,}")

    upset_rate_overall = df_sub["is_upset"].mean()
    print(f"   Overall upset rate: {upset_rate_overall:.3f}")

    # Compute upset features (before normalization, uses raw odds/rank)
    print("\n2) Computing 11 upset-specific features...")
    df_sub = compute_upset_features(df_sub)

    # Add upset features to num_cols
    num_cols_base, cat_cols = get_feature_columns(df_sub)
    num_cols = num_cols_base + [c for c in UPSET_FEATURES if c in df_sub.columns]

    # Normalize to underdog perspective
    print("\n3) Normalizing to underdog perspective...")
    df_sub = normalize_to_underdog_perspective(df_sub, num_cols_base)

    # Temporal split
    print("\n4) Temporal split...")
    train_i, val_i, test_i = temporal_split(df_sub["Date"])
    print(f"   Train={len(train_i):,}  Val={len(val_i):,}  Test={len(test_i):,}")
    print(f"   Train: {df_sub['Date'].iloc[train_i].min():%Y-%m-%d} → {df_sub['Date'].iloc[train_i].max():%Y-%m-%d}")
    print(f"   Test:  {df_sub['Date'].iloc[test_i].min():%Y-%m-%d} → {df_sub['Date'].iloc[test_i].max():%Y-%m-%d}")

    # Prepare features (scaler + encoder fit on train only)
    print("\n5) Preparing features...")
    X, scaler, encoder, fitted_num_cols, fitted_cat_cols = prepare_features(df_sub, train_i)

    # The prepare_features only uses the base num_cols from get_feature_columns.
    # We need to also include the 11 upset features.
    # Check if they're already in fitted_num_cols (they might be if they were
    # created by compute_context_features or similar).
    missing = [c for c in UPSET_FEATURES if c not in fitted_num_cols]
    if missing:
        # Check which exist in df_sub
        existing_missing = [c for c in missing if c in df_sub.columns]
        if existing_missing:
            # Re-prepare with the full feature set
            # We'll manually add these columns to X
            upset_data = df_sub[existing_missing].to_numpy(dtype=np.float32)
            # Impute with training medians
            medians = np.nanmedian(upset_data[train_i], axis=0)
            medians = np.nan_to_num(medians, nan=0.0)
            for j in range(upset_data.shape[1]):
                mask = np.isnan(upset_data[:, j])
                upset_data[mask, j] = medians[j]
            # Scale
            upset_scaler = StandardScaler()
            upset_scaler.fit(upset_data[train_i])
            upset_scaled = upset_scaler.transform(upset_data).astype(np.float32)
            X = np.hstack([X, upset_scaled]).astype(np.float32)
            fitted_num_cols = fitted_num_cols + existing_missing
            # Store extra info for inference
            scaler.upset_features_ = existing_missing
            scaler.upset_medians_ = medians
            scaler.upset_scaler_ = upset_scaler

    y_all = df_sub["is_upset"].to_numpy(dtype=np.float32)
    best_of = df_sub["best_of"].to_numpy(dtype=np.int64)

    print(f"   Input dim: {X.shape[1]}  (num={len(fitted_num_cols)} + cat_oh={X.shape[1]-len(fitted_num_cols)})")

    # Train per BO
    all_metrics = {}
    all_artifacts = {
        "scaler": scaler,
        "encoder": encoder,
        "num_cols": fitted_num_cols,
        "cat_cols": fitted_cat_cols,
        "input_dim": X.shape[1],
        "upset_features": UPSET_FEATURES,
    }

    MODELS_ADV_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_ADV_DIR / "upset_scaler.joblib")
    joblib.dump(encoder, MODELS_ADV_DIR / "upset_encoder.joblib")

    bo_list = [3, 5] if bo_filter == "both" else [int(bo_filter)]

    for bo in bo_list:
        sel = best_of == bo
        if sel.sum() < 500:
            print(f"\n   [BO{bo}] Not enough data ({sel.sum()}), skipping")
            continue

        print(f"\n{'─'*70}")
        print(f"   BO{bo}  n={sel.sum():,}")
        print(f"{'─'*70}")

        tr_mask = sel[train_i]
        va_mask = sel[val_i]
        te_mask = sel[test_i]

        X_tr = X[train_i][tr_mask]
        X_va = X[val_i][va_mask]
        X_te = X[test_i][te_mask]
        y_tr = y_all[train_i][tr_mask]
        y_va = y_all[val_i][va_mask]
        y_te = y_all[test_i][te_mask]

        upset_rate = y_tr.mean()
        print(f"   Train upset rate: {upset_rate:.3f}  (scale_pos_weight={1/upset_rate - 1:.2f})")
        print(f"   train={len(X_tr):,}  val={len(X_va):,}  test={len(X_te):,}")

        # Train LightGBM
        print(f"\n   [LGBM] Training...")
        lgb_model = train_upset_lgbm(X_tr, y_tr, X_va, y_va)
        lgb_val_proba = lgb_model.predict_proba(X_va)[:, 1]
        lgb_test_proba = lgb_model.predict_proba(X_te)[:, 1]

        # Isotonic calibration on validation
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(lgb_val_proba, y_va)
        test_proba = calibrator.predict(lgb_test_proba)
        test_proba = np.clip(test_proba, 1e-6, 1 - 1e-6)

        # Get underdog odds for betting simulation
        df_test_bo = df_sub.iloc[test_i][te_mask].copy()
        odds1 = pd.to_numeric(df_test_bo["Odd_1"], errors="coerce")
        odds2 = pd.to_numeric(df_test_bo["Odd_2"], errors="coerce")
        p1_is_ud = df_test_bo["_p1_is_underdog"].values
        underdog_odds = np.where(p1_is_ud == 1, odds1, odds2).astype(float)

        # Evaluate
        print(f"\n   ── Evaluation ──")
        bo_metrics = evaluate_upset_model(y_te, test_proba, prefix=f"BO{bo}")

        # Segment analysis
        print(f"\n   ── Segment Analysis ──")
        bo_metrics["segments"] = segment_analysis_upset(
            df_test_bo.reset_index(drop=True), test_proba, y_te, prefix=f"BO{bo}",
        )

        # Betting simulation
        print(f"\n   ── Betting Simulation ──")
        bo_metrics["betting"] = betting_simulation(
            test_proba, y_te, underdog_odds, prefix=f"BO{bo}",
        )

        # Rule-based comparison
        print(f"\n   ── Rule-based Comparison ──")
        bo_metrics["rule_comparison"] = compare_with_rule_based(
            df_test_bo.reset_index(drop=True), test_proba, y_te, prefix=f"BO{bo}",
        )

        all_metrics[f"bo{bo}"] = bo_metrics

        # Save artifacts
        lgb_model.booster_.save_model(str(MODELS_ADV_DIR / f"upset_lgb_bo{bo}.txt"))
        joblib.dump(calibrator, MODELS_ADV_DIR / f"upset_calibrator_bo{bo}.joblib")
        all_artifacts[f"bo{bo}_metrics"] = bo_metrics

    # Save meta
    joblib.dump(all_artifacts, MODELS_ADV_DIR / "upset_meta.joblib")

    # Save metrics JSON
    metrics_out = {
        "timestamp": datetime.now().isoformat(),
        "n_matches_total": int(len(df_sub)),
        "upset_rate_overall": float(upset_rate_overall),
        "models": all_metrics,
    }
    with open(MODELS_ADV_DIR / "upset_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"\nMetrics saved → {MODELS_ADV_DIR / 'upset_metrics.json'}")

    return all_metrics


# ──────────────────────────────────────────────────────────────────────────────
# 10) Inference class
# ──────────────────────────────────────────────────────────────────────────────


class UpsetPredictor:
    """Load trained upset models and predict on new matches."""

    def __init__(self, models_dir: Path = MODELS_ADV_DIR):
        self.models_dir = models_dir
        self.scaler = self._maybe_load(models_dir / "upset_scaler.joblib")
        self.encoder = self._maybe_load(models_dir / "upset_encoder.joblib")
        self.meta = self._maybe_load(models_dir / "upset_meta.joblib") or {}

        self.num_cols = self.meta.get("num_cols", [])
        self.cat_cols = self.meta.get("cat_cols", list(CAT_COLS_ADVANCED))
        self.upset_features = self.meta.get("upset_features", UPSET_FEATURES)

        import lightgbm as lgb
        self.models: Dict[int, dict] = {}
        for bo in (3, 5):
            model_path = models_dir / f"upset_lgb_bo{bo}.txt"
            if not model_path.exists():
                continue
            calibrator_path = models_dir / f"upset_calibrator_bo{bo}.joblib"
            self.models[bo] = {
                "lgb": lgb.Booster(model_file=str(model_path)),
                "calibrator": self._maybe_load(calibrator_path),
            }

        loaded_bo = list(self.models.keys())
        print(f"[UpsetPredictor] loaded: BO={loaded_bo}")

    @staticmethod
    def _maybe_load(path: Path):
        if path.exists():
            try:
                return joblib.load(path)
            except Exception as e:
                print(f"   warn: failed loading {path.name}: {e}")
        return None

    def _build_X(self, features: dict) -> np.ndarray:
        """Build feature vector from a features dict."""
        num_vals = np.array(
            [float(features.get(c, np.nan)) for c in self.num_cols],
            dtype=np.float32,
        )
        # Fill NaN with training medians
        if self.scaler is not None and hasattr(self.scaler, "train_medians_"):
            medians = self.scaler.train_medians_
            mask = np.isnan(num_vals)
            if len(medians) == len(num_vals):
                num_vals[mask] = medians[mask]
            else:
                num_vals = np.nan_to_num(num_vals, nan=0.0)
        else:
            num_vals = np.nan_to_num(num_vals, nan=0.0)

        X_num = self.scaler.transform(num_vals.reshape(1, -1)).astype(
            np.float32
        ) if self.scaler else num_vals.reshape(1, -1)

        # Handle upset-specific extra features
        if hasattr(self.scaler, "upset_features_") and self.scaler.upset_features_:
            upset_vals = np.array(
                [float(features.get(c, np.nan)) for c in self.scaler.upset_features_],
                dtype=np.float32,
            )
            mask = np.isnan(upset_vals)
            upset_vals[mask] = self.scaler.upset_medians_[mask]
            upset_scaled = self.scaler.upset_scaler_.transform(
                upset_vals.reshape(1, -1)
            ).astype(np.float32)
            X_num = np.hstack([X_num, upset_scaled]).astype(np.float32)

        # Categorical
        if (
            self.encoder is not None
            and hasattr(self.encoder, "feature_names_in_")
            and len(self.encoder.feature_names_in_) > 0
        ):
            cat_names = list(self.encoder.feature_names_in_)
            cat_df = pd.DataFrame(
                [[features.get(c, "NA") for c in cat_names]],
                columns=cat_names,
            )
            X_cat = self.encoder.transform(cat_df.astype(str)).astype(np.float32)
        else:
            X_cat = np.zeros((1, 0), dtype=np.float32)

        return np.hstack([X_num, X_cat]).astype(np.float32)

    def predict(self, features: dict, best_of: int = 3) -> dict:
        """Predict upset probability for a match.

        Parameters
        ----------
        features : dict of feature values (must include all V3 + upset features)
        best_of : 3 or 5

        Returns
        -------
        dict with upset_probability, is_predicted_upset, confidence
        """
        models = self.models.get(best_of)
        if models is None:
            return {
                "upset_probability": 0.0,
                "is_predicted_upset": False,
                "confidence": "no_model",
            }

        X = self._build_X(features)

        # LGBM prediction
        raw_p = float(models["lgb"].predict(X)[0])

        # Calibrate
        calibrator = models["calibrator"]
        if calibrator is not None:
            p = float(np.clip(calibrator.predict([raw_p])[0], 1e-6, 1 - 1e-6))
        else:
            p = float(np.clip(raw_p, 1e-6, 1 - 1e-6))

        # Confidence bucket
        if p >= 0.6:
            confidence = "high"
        elif p >= 0.45:
            confidence = "moderate"
        else:
            confidence = "low"

        return {
            "upset_probability": round(p, 4),
            "is_predicted_upset": p >= 0.5,
            "confidence": confidence,
            "_raw_probability": round(raw_p, 4),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 11) Live predictor — simple API for notebook usage
# ──────────────────────────────────────────────────────────────────────────────


class UpsetLivePredictor:
    """High-level predictor that assembles all V3 + upset features from the
    feature cache, so the user only needs to provide basic match info.

    Usage::

        pred = UpsetLivePredictor()
        result = pred.predici_upset(
            "Sinner J.", "Alcaraz C.", rank1=1, rank2=2,
            odds1=1.90, odds2=1.95, surface="Clay",
            tournament="Slam", date_str="2026-05-15", best_of=5,
        )
    """

    _BASE_STATS = [
        "win", "ace", "df", "bp_save_rate",
        "1st_serve_pct", "1st_serve_win_pct", "2nd_serve_win_pct",
        "minutes", "streak", "total_games", "sv_gms",
    ]
    _SURF_STATS = [
        "surf_win_rate", "surf_avg_aces", "surf_avg_bp_save",
        "surf_avg_1st_pct", "surf_matches",
    ]
    _WINDOWS = [10, 20, 50]

    def __init__(self, models_dir: Path = MODELS_ADV_DIR):
        # 1) Load feature cache
        print("[UpsetLivePredictor] Loading feature cache...")
        self.cache = pd.read_parquet(CACHE_PATH)
        self.cache["Date"] = pd.to_datetime(self.cache["Date"])

        # 2) Build player → last-row-index lookup
        print("[UpsetLivePredictor] Building player lookup...")
        self._player_last_idx: Dict[str, int] = {}
        all_players = set(self.cache["Player_1"].dropna()) | set(
            self.cache["Player_2"].dropna()
        )
        for p in all_players:
            mask = (self.cache["Player_1"] == p) | (self.cache["Player_2"] == p)
            idx = self.cache.loc[mask, "Date"].idxmax()
            self._player_last_idx[p] = idx

        # 3) Load trained upset model artifacts
        self._base_predictor = UpsetPredictor(models_dir=models_dir)

        # 4) Extract ordered feature lists from trained model meta
        meta = joblib.load(models_dir / "upset_meta.joblib")
        self._num_cols: List[str] = meta["num_cols"]
        self._cat_cols: List[str] = meta.get("cat_cols", list(CAT_COLS_ADVANCED))
        self._upset_feat_names: List[str] = meta.get("upset_features", UPSET_FEATURES)

        # 5) Build swap index for perspective normalization
        perm, sign = _build_swap_index(self._num_cols[:188])
        self._swap_perm = perm
        self._swap_sign = sign

        # 6) Precompute player date series for fatigue (as int64 ns for fast comparison)
        print("[UpsetLivePredictor] Building player date index...")
        self._player_dates: Dict[str, np.ndarray] = {}
        for p in all_players:
            dates = self.cache.loc[
                (self.cache["Player_1"] == p) | (self.cache["Player_2"] == p), "Date"
            ].values
            # Convert to int64 ns timestamps for fast searchsorted
            self._player_dates[p] = np.sort(dates.astype("int64"))

        print(f"[UpsetLivePredictor] Ready — {len(all_players)} players indexed.")

    # ── name resolution ────────────────────────────────────────────────────

    def _resolve_player(self, name: str) -> str:
        """Fuzzy-match a player name against the cache."""
        if name in self._player_last_idx:
            return name
        candidates = list(self._player_last_idx.keys())
        matches = get_close_matches(name, candidates, n=1, cutoff=0.6)
        if matches:
            print(f"   [resolve] '{name}' → '{matches[0]}' (fuzzy)")
            return matches[0]
        raise ValueError(
            f"Player '{name}' not found in cache. "
            f"Closest: {get_close_matches(name, candidates, n=3, cutoff=0.4)}"
        )

    # ── per-player stats from cache ────────────────────────────────────────

    def _get_player_stats(self, player: str) -> Dict[str, float]:
        """Extract rolling, surface, bio stats from the player's last match."""
        idx = self._player_last_idx.get(player)
        if idx is None:
            return {}
        row = self.cache.loc[idx]
        is_p1 = row["Player_1"] == player
        prefix = "p1_" if is_p1 else "p2_"

        stats: Dict[str, float] = {}
        # Rolling stats (11 stats × 3 windows = 33 raw features)
        for w in self._WINDOWS:
            for stat in self._BASE_STATS:
                col = f"{prefix}{stat}_w{w}"
                val = row.get(col, np.nan)
                stats[f"{stat}_w{w}"] = float(val) if pd.notna(val) else np.nan

        # Surface stats
        for s in self._SURF_STATS:
            col = f"{prefix}{s}"
            val = row.get(col, np.nan)
            stats[s] = float(val) if pd.notna(val) else np.nan

        # Bio
        for bio in ["age", "ht", "hand", "seed"]:
            col = f"{prefix}{bio}"
            val = row.get(col, np.nan)
            stats[bio] = val if pd.notna(val) else None

        return stats

    # ── feature computation helpers ────────────────────────────────────────

    def _compute_context(
        self, rank1, rank2, odds1, odds2, best_of, tournament,
    ) -> Dict[str, float]:
        """Match context features from user inputs."""
        eps = 1e-8
        return {
            "rank_diff": rank1 - rank2,
            "rank_diff_abs": abs(rank1 - rank2),
            "log_odds_ratio": np.log(odds1 / max(odds2, eps)),
            "odds_ratio": odds1 / max(odds2, eps),
            "odd_1": odds1,
            "odd_2": odds2,
            "log_odd_1": np.log(max(odds1, eps)),
            "log_odd_2": np.log(max(odds2, eps)),
            "implied_prob_1": 1.0 / max(odds1, eps),
            "implied_prob_2": 1.0 / max(odds2, eps),
            "prob_diff": 1.0 / max(odds1, eps) - 1.0 / max(odds2, eps),
            "min_odds": min(odds1, odds2),
            "max_odds": max(odds1, odds2),
            "favorite_strength": 1.0 / min(odds1, odds2),
            "match_competitiveness": min(1 / odds1, 1 / odds2) / max(1 / odds1, 1 / odds2),
            "best_of": best_of,
            "rank_points_diff": 0.0,
            "age_diff": 0.0,
            "height_diff": 0.0,
            "seed_diff": 0.0,
            "draw_size": self._infer_draw_size(tournament),
        }

    @staticmethod
    def _infer_draw_size(tournament: str) -> float:
        t = tournament.lower().strip()
        if "slam" in t or "g" == t:
            return 128.0
        if "masters" in t or "m1000" in t or "m" == t:
            return 96.0
        if "500" in t or "500" == t:
            return 48.0
        if "250" in t or "250" == t:
            return 32.0
        if "final" in t or "f" == t:
            return 8.0
        return 32.0

    def _compute_diff_ratio(
        self, p1_stats: Dict, p2_stats: Dict,
    ) -> Dict[str, float]:
        """Compute diff and ratio features from per-player rolling stats."""
        eps = 1e-8
        feats: Dict[str, float] = {}
        for w in self._WINDOWS:
            for stat in self._BASE_STATS:
                key = f"{stat}_w{w}"
                v1 = p1_stats.get(key, np.nan)
                v2 = p2_stats.get(key, np.nan)
                if pd.notna(v1) and pd.notna(v2):
                    feats[f"{stat}_diff_w{w}"] = v1 - v2
                    feats[f"{stat}_ratio_w{w}"] = (v1 + eps) / (v2 + eps)
                else:
                    feats[f"{stat}_diff_w{w}"] = np.nan
                    feats[f"{stat}_ratio_w{w}"] = np.nan
        return feats

    def _compute_raw_rolling(
        self, p1_stats: Dict, p2_stats: Dict,
    ) -> Dict[str, float]:
        """Raw per-player rolling features (p1_<stat>_w<W>, p2_<stat>_w<W>)."""
        feats: Dict[str, float] = {}
        for w in self._WINDOWS:
            for stat in self._BASE_STATS:
                key = f"{stat}_w{w}"
                feats[f"p1_{key}"] = p1_stats.get(key, np.nan)
                feats[f"p2_{key}"] = p2_stats.get(key, np.nan)
        return feats

    def _compute_surface(
        self, p1_stats: Dict, p2_stats: Dict,
    ) -> Dict[str, float]:
        """Surface diff features + raw per-player surface stats."""
        feats: Dict[str, float] = {}
        surf_keys = ["surf_win_rate", "surf_avg_aces", "surf_avg_bp_save",
                      "surf_avg_1st_pct", "surf_matches"]
        for s in surf_keys:
            v1 = p1_stats.get(s, 0.0) or 0.0
            v2 = p2_stats.get(s, 0.0) or 0.0
            feats[f"{s}_diff"] = v1 - v2
            feats[f"p1_{s}"] = v1
            feats[f"p2_{s}"] = v2
        return feats

    def _compute_h2h(
        self, p1: str, p2: str, surface: str,
    ) -> Dict[str, float]:
        """H2H features by scanning the cache."""
        mask = (
            ((self.cache["Player_1"] == p1) & (self.cache["Player_2"] == p2))
            | ((self.cache["Player_1"] == p2) & (self.cache["Player_2"] == p1))
        )
        h2h = self.cache.loc[mask]
        n = len(h2h)
        if n == 0:
            return {
                "h2h_win_rate": 0.5, "h2h_matches": 0,
                "h2h_avg_total_games": np.nan, "h2h_avg_minutes": np.nan,
                "h2h_surface_win_rate": 0.5,
            }
        p1_wins = ((h2h["Winner"] == p1)).sum()
        tg = pd.to_numeric(h2h.get("total_games", np.nan), errors="coerce")
        mins = pd.to_numeric(h2h.get("minutes", np.nan), errors="coerce")
        # Surface-specific H2H
        surf_h2h = h2h[h2h["Surface"].str.lower() == surface.lower()]
        surf_n = len(surf_h2h)
        surf_p1_wins = (surf_h2h["Winner"] == p1).sum() if surf_n > 0 else 0
        return {
            "h2h_win_rate": float(p1_wins / n),
            "h2h_matches": float(n),
            "h2h_avg_total_games": float(tg.mean()) if tg.notna().any() else np.nan,
            "h2h_avg_minutes": float(mins.mean()) if mins.notna().any() else np.nan,
            "h2h_surface_win_rate": float(surf_p1_wins / surf_n) if surf_n > 0 else 0.5,
        }

    def _compute_fatigue(
        self, p1: str, p2: str, target_date: pd.Timestamp,
    ) -> Dict[str, float]:
        """Fatigue features: recent match count per player."""
        feats: Dict[str, float] = {}
        target_ns = np.int64(target_date.value)  # ns timestamp
        for player, prefix in [(p1, "p1"), (p2, "p2")]:
            dates_ns = self._player_dates.get(player, np.array([], dtype=np.int64))
            for w in [7, 14, 30]:
                cutoff_ns = np.int64(
                    (target_date - pd.Timedelta(days=w)).value
                )
                # Count dates in (cutoff, target) interval
                recent = dates_ns[(dates_ns > cutoff_ns) & (dates_ns < target_ns)]
                feats[f"{prefix}_matches_{w}d"] = float(len(recent))
        for w in [7, 14, 30]:
            feats[f"fatigue_diff_{w}d"] = (
                feats[f"p1_matches_{w}d"] - feats[f"p2_matches_{w}d"]
            )
        return feats

    def _compute_tournament(self, tournament: str) -> Dict[str, float]:
        """Tournament features from cache averages."""
        # Map user-friendly names to tourney_level codes
        level_map = {
            "slam": "G", "g": "G", "grand slam": "G",
            "masters": "M", "masters 1000": "M", "m1000": "M", "m": "M",
            "500": "500", "atp500": "500",
            "250": "250", "atp250": "250",
            "finals": "F", "f": "F", "tour finals": "F",
            "next gen": "A", "a": "A",
        }
        level = level_map.get(tournament.lower().strip(), None)
        if level and "tourney_level" in self.cache.columns:
            similar = self.cache[self.cache["tourney_level"] == level]
            if len(similar) > 10:
                tg = pd.to_numeric(similar["total_games"], errors="coerce")
                return {
                    "tourney_avg_games": float(tg.mean()),
                    "tourney_std_games": float(tg.std()),
                    "tourney_game_trend": 0.0,
                }
        return {"tourney_avg_games": np.nan, "tourney_std_games": np.nan,
                "tourney_game_trend": 0.0}

    def _compute_bio(self, p1_stats: Dict, p2_stats: Dict) -> Dict[str, float]:
        """Bio features from player stats."""
        # Hand match
        h1, h2 = p1_stats.get("hand"), p2_stats.get("hand")
        hand_match = 1.0 if (h1 and h2 and h1 == h2) else 0.5
        # Height diff
        ht1 = float(p1_stats["ht"]) if p1_stats.get("ht") is not None else 0.0
        ht2 = float(p2_stats["ht"]) if p2_stats.get("ht") is not None else 0.0
        # Age diff
        age1 = float(p1_stats["age"]) if p1_stats.get("age") is not None else 0.0
        age2 = float(p2_stats["age"]) if p2_stats.get("age") is not None else 0.0
        return {
            "hand_match": hand_match,
            "backhand_match": 0.0,
            "bmi_diff": 0.0,
            "height_diff": ht1 - ht2,
            "age_diff": age1 - age2,
        }

    def _compute_upset_features_single(
        self,
        features: Dict[str, float],
        rank1: float, rank2: float,
        odds1: float, odds2: float,
    ) -> Dict[str, float]:
        """Compute the 11 upset-specific features for a single match dict.
        These are computed AFTER diff/ratio features are available."""
        eps = 1e-8

        # 1) market_rank_disagreement
        odds_sign = np.sign(odds1 - odds2)
        rank_sign = np.sign(rank1 - rank2)
        market_rank_disagreement = 1.0 if odds_sign != rank_sign else 0.0

        # 2) underdog_form_trend: w10 - w50 win rate diff
        w10 = features.get("win_diff_w10", 0.0) or 0.0
        w50 = features.get("win_diff_w50", 0.0) or 0.0
        underdog_form_trend = w10 - w50

        # 3-4) favorite_fatigue (negated fatigue_diff: positive = favorite more tired)
        fat7 = features.get("fatigue_diff_7d", 0.0) or 0.0
        fat14 = features.get("fatigue_diff_14d", 0.0) or 0.0

        # 5) surface_expertise_gap
        surface_expertise_gap = features.get("surf_win_rate_diff", 0.0) or 0.0

        # 6) underdog_ace_edge
        underdog_ace_edge = features.get("ace_diff_w10", 0.0) or 0.0

        # 7) upset_h2h_signal
        upset_h2h_signal = features.get("h2h_win_rate", 0.5) or 0.5

        # 8) underdog_avg_match_length
        underdog_avg_match_length = features.get("minutes_diff_w10", 0.0) or 0.0

        # 9) odds_sensitivity
        min_o = min(odds1, odds2)
        max_o = max(odds1, odds2)
        odds_sensitivity = min_o / max(max_o, eps)

        # 10) rank_odds_divergence
        rank_gap = abs(rank1 - rank2)
        odds_gap = abs(odds1 - odds2)
        rank_odds_divergence = (
            np.log1p(min(rank_gap, 500)) - np.log1p(min(odds_gap, 20))
        )

        # 11) underdog_streak
        underdog_streak = features.get("streak_diff_w10", 0.0) or 0.0

        return {
            "market_rank_disagreement": market_rank_disagreement,
            "underdog_form_trend": underdog_form_trend,
            "favorite_fatigue_7d": -fat7,
            "favorite_fatigue_14d": -fat14,
            "surface_expertise_gap": surface_expertise_gap,
            "underdog_ace_edge": underdog_ace_edge,
            "upset_h2h_signal": upset_h2h_signal,
            "underdog_avg_match_length": underdog_avg_match_length,
            "odds_sensitivity": odds_sensitivity,
            "rank_odds_divergence": rank_odds_divergence,
            "underdog_streak": underdog_streak,
        }

    def _normalize_perspective(
        self, features: Dict[str, float], p1_is_underdog: bool,
    ) -> Dict[str, float]:
        """If P1 is NOT the underdog, swap features to underdog perspective."""
        if p1_is_underdog:
            return features
        # Need to swap: apply the swap logic to the 188 base num features
        # Build a vector of the first 188 features, swap, then rebuild dict
        base_cols = self._num_cols[:188]
        vec = np.array([features.get(c, 0.0) for c in base_cols], dtype=np.float32)
        vec = vec.reshape(1, -1)
        vec_swapped = apply_swap(vec, self._swap_perm, self._swap_sign)
        swapped_features = dict(features)  # copy
        for j, col in enumerate(base_cols):
            swapped_features[col] = float(vec_swapped[0, j])
        return swapped_features

    # ── main entry point ───────────────────────────────────────────────────

    def predici_upset(
        self,
        player1: str,
        player2: str,
        rank1: float,
        rank2: float,
        odds1: float,
        odds2: float,
        surface: str = "Hard",
        tournament: str = "ATP500",
        date_str: str = "2026-06-01",
        best_of: int = 3,
    ) -> dict:
        """Predict upset probability for a match.

        Parameters
        ----------
        player1, player2 : player names (fuzzy-matched against cache)
        rank1, rank2 : ATP rankings
        odds1, odds2 : decimal odds
        surface : "Hard", "Clay", "Grass", or "Carpet"
        tournament : "Slam", "Masters 1000", "ATP500", "ATP250", "Finals"
        date_str : YYYY-MM-DD (used for fatigue and feature freshness)
        best_of : 3 or 5
        """
        # 1. Resolve names
        p1 = self._resolve_player(player1)
        p2 = self._resolve_player(player2)

        target_date = pd.Timestamp(date_str)

        # 2. Per-player stats from last cache match
        p1_stats = self._get_player_stats(p1)
        p2_stats = self._get_player_stats(p2)

        # 3. Context features
        features = self._compute_context(rank1, rank2, odds1, odds2, best_of, tournament)

        # 4. Bio features (may update age_diff, height_diff)
        bio = self._compute_bio(p1_stats, p2_stats)
        features["age_diff"] = bio["age_diff"]
        features["height_diff"] = bio["height_diff"]
        features["hand_match"] = bio["hand_match"]

        # 5. Diff/ratio features
        features.update(self._compute_diff_ratio(p1_stats, p2_stats))

        # 6. Raw rolling features
        features.update(self._compute_raw_rolling(p1_stats, p2_stats))

        # 7. Surface features
        features.update(self._compute_surface(p1_stats, p2_stats))

        # 8. H2H features
        features.update(self._compute_h2h(p1, p2, surface))

        # 9. Tournament features
        features.update(self._compute_tournament(tournament))

        # 10. Fatigue features
        features.update(self._compute_fatigue(p1, p2, target_date))

        # 11. Compute the 11 upset features (before perspective normalization,
        #     but using the raw diff features which are P1-P2 perspective)
        upset_feats = self._compute_upset_features_single(
            features, rank1, rank2, odds1, odds2,
        )
        features.update(upset_feats)

        # 12. Determine underdog and normalize perspective
        p1_is_underdog = odds1 > odds2
        features = self._normalize_perspective(features, p1_is_underdog)

        # After normalization, re-compute upset features that depend on
        # the normalized perspective (form trend, fatigue, ace edge, etc.)
        # Actually: the upset features use diff features which have been swapped.
        # Re-compute them from the normalized features.
        upset_feats_norm = self._compute_upset_features_single(
            features, rank1, rank2, odds1, odds2,
        )
        features.update(upset_feats_norm)

        # 13. Build categorical values
        features["Surface"] = surface
        features["tourney_level"] = self._map_tournament_level(tournament)
        features["indoor"] = "0"  # outdoor by default

        # 14. Predict — build feature vector manually
        bp = self._base_predictor
        models = bp.models.get(best_of)
        if models is None:
            return {
                "upset_probability": 0.0,
                "is_predicted_upset": False,
                "confidence": "no_model",
                "underdog": p1 if p1_is_underdog else p2,
                "favorite": p2 if p1_is_underdog else p1,
                "error": f"No model for BO{best_of}",
            }

        # Build base 188 numerical features
        base_num_cols = bp.meta.get("num_cols", self._num_cols)[:188]
        num_vals = np.array(
            [float(features.get(c, np.nan)) for c in base_num_cols],
            dtype=np.float32,
        )
        # Impute NaN with training medians
        medians = bp.scaler.train_medians_
        mask = np.isnan(num_vals)
        if len(medians) == len(num_vals):
            num_vals[mask] = medians[mask]
        else:
            num_vals = np.nan_to_num(num_vals, nan=0.0)
        X_num = bp.scaler.transform(num_vals.reshape(1, -1)).astype(np.float32)

        # Upset-specific features (11)
        upset_feat_names = bp.scaler.upset_features_
        upset_vals = np.array(
            [float(features.get(c, np.nan)) for c in upset_feat_names],
            dtype=np.float32,
        )
        mask_u = np.isnan(upset_vals)
        upset_vals[mask_u] = bp.scaler.upset_medians_[mask_u]
        upset_scaled = bp.scaler.upset_scaler_.transform(
            upset_vals.reshape(1, -1)
        ).astype(np.float32)
        X_num = np.hstack([X_num, upset_scaled]).astype(np.float32)

        # Categorical one-hot
        if (bp.encoder is not None
                and hasattr(bp.encoder, "feature_names_in_")
                and len(bp.encoder.feature_names_in_) > 0):
            cat_names = list(bp.encoder.feature_names_in_)
            cat_df = pd.DataFrame(
                [[features.get(c, "NA") for c in cat_names]],
                columns=cat_names,
            )
            X_cat = bp.encoder.transform(cat_df.astype(str)).astype(np.float32)
        else:
            X_cat = np.zeros((1, 0), dtype=np.float32)

        X = np.hstack([X_num, X_cat]).astype(np.float32)

        # LGBM prediction
        raw_p = float(models["lgb"].predict(X)[0])

        # Calibrate
        calibrator = models["calibrator"]
        if calibrator is not None:
            p = float(np.clip(calibrator.predict([raw_p])[0], 1e-6, 1 - 1e-6))
        else:
            p = float(np.clip(raw_p, 1e-6, 1 - 1e-6))

        # Confidence bucket
        if p >= 0.6:
            confidence = "high"
        elif p >= 0.45:
            confidence = "moderate"
        else:
            confidence = "low"

        result = {
            "upset_probability": round(p, 4),
            "is_predicted_upset": p >= 0.5,
            "confidence": confidence,
            "_raw_probability": round(raw_p, 4),
        }

        # 15. Build rich output
        underdog = p1 if p1_is_underdog else p2
        favorite = p2 if p1_is_underdog else p1

        result["underdog"] = underdog
        result["favorite"] = favorite
        result["underdog_odds"] = max(odds1, odds2)
        result["favorite_odds"] = min(odds1, odds2)
        result["player1"] = p1
        result["player2"] = p2
        result["best_of"] = best_of
        result["surface"] = surface
        result["analysis"] = {
            "market_rank_disagreement": float(features.get("market_rank_disagreement") or 0),
            "underdog_form_trend": float(features.get("underdog_form_trend") or 0),
            "favorite_fatigue_7d": float(features.get("favorite_fatigue_7d") or 0),
            "surface_expertise_gap": float(features.get("surface_expertise_gap") or 0),
            "underdog_ace_edge": float(features.get("underdog_ace_edge") or 0),
            "h2h_signal": float(features.get("upset_h2h_signal") or 0.5),
            "odds_sensitivity": float(features.get("odds_sensitivity") or 0),
        }
        return result

    @staticmethod
    def _map_tournament_level(tournament: str) -> str:
        t = tournament.lower().strip()
        if "slam" in t or t == "g":
            return "G"
        if "masters" in t or t in ("m1000", "m"):
            return "M"
        if "500" in t or t == "500":
            return "500"
        if "250" in t or t == "250":
            return "250"
        if "final" in t or t == "f":
            return "F"
        if "next" in t or t == "a":
            return "A"
        return "250"  # default

    # ── historical analysis (backtesting) ──────────────────────────────────

    def _find_historical_match(
        self, player1: str, player2: str, date_str: str,
        surface: Optional[str] = None,
    ) -> Optional[pd.Series]:
        """Find a specific match in the feature cache."""
        p1 = self._resolve_player(player1)
        p2 = self._resolve_player(player2)
        target_date = pd.Timestamp(date_str)

        # Match by players (in either order)
        mask = (
            ((self.cache["Player_1"] == p1) & (self.cache["Player_2"] == p2))
            | ((self.cache["Player_1"] == p2) & (self.cache["Player_2"] == p1))
        )
        candidates = self.cache.loc[mask]
        if candidates.empty:
            return None

        # Match by date (exact or closest)
        candidates = candidates.copy()
        candidates["_date_diff"] = (candidates["Date"] - target_date).abs()
        best_idx = candidates["_date_diff"].idxmin()
        row = candidates.loc[best_idx]

        # If surface given, prefer same surface if multiple on same date
        if surface is not None and len(candidates) > 1:
            same_surf = candidates[candidates["Surface"].str.lower() == surface.lower()]
            if not same_surf.empty:
                best_idx = same_surf["_date_diff"].idxmin()
                row = same_surf.loc[best_idx]

        return row

    def _predict_from_row(self, row: pd.Series) -> dict:
        """Predict upset from a cache row (all 188 features already computed)."""
        bp = self._base_predictor

        # Extract base 188 numerical features directly from the row
        base_num_cols = bp.meta.get("num_cols", self._num_cols)[:188]
        num_vals = np.array(
            [float(row.get(c, np.nan)) if pd.notna(row.get(c, np.nan)) else np.nan
             for c in base_num_cols],
            dtype=np.float32,
        )
        # Impute NaN with training medians
        medians = bp.scaler.train_medians_
        mask = np.isnan(num_vals)
        if len(medians) == len(num_vals):
            num_vals[mask] = medians[mask]
        else:
            num_vals = np.nan_to_num(num_vals, nan=0.0)

        # Get raw odds/rank before normalization
        odds1 = float(row.get("Odd_1", np.nan))
        odds2 = float(row.get("Odd_2", np.nan))
        rank1 = float(row.get("Rank_1", np.nan))
        rank2 = float(row.get("Rank_2", np.nan))
        p1_name = str(row.get("Player_1", ""))
        p2_name = str(row.get("Player_2", ""))
        winner = str(row.get("Winner", ""))
        best_of = int(row.get("best_of", 3))

        # Determine underdog perspective
        if np.isnan(odds1) or np.isnan(odds2):
            p1_is_underdog = rank1 > rank2
        else:
            p1_is_underdog = odds1 > odds2

        # Apply perspective normalization to the feature vector
        if not p1_is_underdog:
            vec = num_vals.reshape(1, -1)
            vec_swapped = apply_swap(vec, self._swap_perm, self._swap_sign)
            num_vals = vec_swapped[0]

        X_num = bp.scaler.transform(num_vals.reshape(1, -1)).astype(np.float32)

        # Compute the 11 upset features from the (now normalized) base features
        feat_dict = {c: float(num_vals[i]) for i, c in enumerate(base_num_cols)}
        upset_feats = self._compute_upset_features_single(
            feat_dict, rank1, rank2, odds1, odds2,
        )

        # Build upset feature vector
        upset_feat_names = bp.scaler.upset_features_
        upset_vals = np.array(
            [float(upset_feats.get(c, np.nan)) for c in upset_feat_names],
            dtype=np.float32,
        )
        mask_u = np.isnan(upset_vals)
        upset_vals[mask_u] = bp.scaler.upset_medians_[mask_u]
        upset_scaled = bp.scaler.upset_scaler_.transform(
            upset_vals.reshape(1, -1)
        ).astype(np.float32)
        X_num = np.hstack([X_num, upset_scaled]).astype(np.float32)

        # Categorical
        if (bp.encoder is not None
                and hasattr(bp.encoder, "feature_names_in_")
                and len(bp.encoder.feature_names_in_) > 0):
            cat_names = list(bp.encoder.feature_names_in_)
            cat_df = pd.DataFrame(
                [[str(row.get(c, "NA")) for c in cat_names]],
                columns=cat_names,
            )
            X_cat = bp.encoder.transform(cat_df.astype(str)).astype(np.float32)
        else:
            X_cat = np.zeros((1, 0), dtype=np.float32)

        X = np.hstack([X_num, X_cat]).astype(np.float32)

        # Predict
        models = bp.models.get(best_of) or bp.models.get(3)
        if models is None:
            return {"error": "No model available"}

        raw_p = float(models["lgb"].predict(X)[0])
        calibrator = models["calibrator"]
        if calibrator is not None:
            p = float(np.clip(calibrator.predict([raw_p])[0], 1e-6, 1 - 1e-6))
        else:
            p = float(np.clip(raw_p, 1e-6, 1 - 1e-6))

        if p >= 0.6:
            confidence = "high"
        elif p >= 0.45:
            confidence = "moderate"
        else:
            confidence = "low"

        underdog = p1_name if p1_is_underdog else p2_name
        favorite = p2_name if p1_is_underdog else p1_name
        actual_upset = (
            (winner == underdog)
            if not np.isnan(odds1) and not np.isnan(odds2)
            else None
        )

        return {
            "date": str(row.get("Date", ""))[:10],
            "player1": p1_name,
            "player2": p2_name,
            "winner": winner,
            "underdog": underdog,
            "favorite": favorite,
            "surface": str(row.get("Surface", "")),
            "best_of": best_of,
            "rank_1": rank1,
            "rank_2": rank2,
            "odds_1": odds1,
            "odds_2": odds2,
            "underdog_odds": max(odds1, odds2) if not np.isnan(odds1) else None,
            "favorite_odds": min(odds1, odds2) if not np.isnan(odds1) else None,
            "upset_probability": round(p, 4),
            "is_predicted_upset": p >= 0.5,
            "confidence": confidence,
            "actual_upset": actual_upset,
            "correct": (
                (p >= 0.5) == actual_upset if actual_upset is not None else None
            ),
            "analysis": {
                "market_rank_disagreement": float(upset_feats.get("market_rank_disagreement") or 0),
                "underdog_form_trend": float(upset_feats.get("underdog_form_trend") or 0),
                "favorite_fatigue_7d": float(upset_feats.get("favorite_fatigue_7d") or 0),
                "surface_expertise_gap": float(upset_feats.get("surface_expertise_gap") or 0),
                "underdog_ace_edge": float(upset_feats.get("underdog_ace_edge") or 0),
                "h2h_signal": float(upset_feats.get("upset_h2h_signal") or 0.5),
                "odds_sensitivity": float(upset_feats.get("odds_sensitivity") or 0),
            },
        }

    def analizza_passato(
        self,
        player1: str,
        player2: str,
        date_str: str,
        surface: Optional[str] = None,
    ) -> Optional[dict]:
        """Analyze a historical match using the exact features as they were at that time.

        Finds the match in the feature cache (which already has shift(1) rolling
        stats), extracts all features, and predicts — no data leakage.

        Parameters
        ----------
        player1, player2 : player names (fuzzy-matched)
        date_str : YYYY-MM-DD of the match
        surface : optional surface filter if multiple matches on the same date

        Returns
        -------
        dict with prediction + actual result, or None if match not found.
        """
        row = self._find_historical_match(player1, player2, date_str, surface)
        if row is None:
            print(f"  Match non trovato: {player1} vs {player2} ({date_str})")
            return None
        return self._predict_from_row(row)

    def analizza_h2h(
        self,
        player1: str,
        player2: str,
    ) -> List[dict]:
        """Analyze ALL historical matches between two players.

        Returns a list of prediction dicts, one per match, sorted by date.
        Each includes the model's prediction and whether it was correct.
        """
        p1 = self._resolve_player(player1)
        p2 = self._resolve_player(player2)

        mask = (
            ((self.cache["Player_1"] == p1) & (self.cache["Player_2"] == p2))
            | ((self.cache["Player_1"] == p2) & (self.cache["Player_2"] == p1))
        )
        matches = self.cache.loc[mask].sort_values("Date")

        results = []
        for idx, row in matches.iterrows():
            r = self._predict_from_row(row)
            if r and "error" not in r:
                results.append(r)
        return results


# ──────────────────────────────────────────────────────────────────────────────
# 12) CLI entry point
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="ATP Upset Detector Training")
    parser.add_argument(
        "--bo", default="both", choices=["3", "5", "both"],
        help="Best-of filter: 3, 5, or both",
    )
    parser.add_argument("--min-year", type=int, default=2003)
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print(f"UPSET DETECTOR TRAINING   bo={args.bo}   min_year={args.min_year}")
    print(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    # Load features
    df = load_or_build_features(force=args.force_features)
    df["Date"] = pd.to_datetime(df["Date"])
    df = filter_post_stats(df, min_year=args.min_year)
    print(f"After year filter (>= {args.min_year}): {len(df):,} matches")

    metrics = train_upset_detector(df, bo_filter=args.bo)

    print(f"\nDone: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()

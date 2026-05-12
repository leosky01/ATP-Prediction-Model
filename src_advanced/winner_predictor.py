#!/usr/bin/env python3
"""
Match Winner Predictor with Comprehensive Player Profiles.

Builds detailed player profiles from TML data (serve, return, performance,
clutch, surface-specific stats) and uses them to predict match winners.

Goal: beat the bookmaker's odds-implied probabilities.

Usage:
    python -m src_advanced.winner_predictor --evaluate
    python -m src_advanced.winner_predictor --train
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    MERGED_DATASET_PATH,
    MODELS_ADV_DIR,
    RANDOM_SEED,
)


# ── Score parsing utilities ──────────────────────────────────────────────────


def parse_score_info(score_str: str):
    """
    Parse a tennis score string and extract:
    - n_sets_played
    - has_tiebreak (any set went to 7-6 or 6-7)
    - sets_won_by_winner
    - went_to_deciding_set (3rd in BO3, 5th in BO5)
    - comeback (lost 1st set but won match)

    Returns dict or None if score is invalid.
    """
    if not isinstance(score_str, str):
        return None

    sets = score_str.strip().split()
    if len(sets) < 2:
        return None

    valid_sets = []
    for s in sets:
        s = s.strip().upper()
        if s in ("RET", "W/O", "DEF", "ABD", "WALKOVER", ""):
            break
        if re.match(r"\d+-\d+", s):
            valid_sets.append(s)

    if len(valid_sets) < 2:
        return None

    n_sets = len(valid_sets)
    has_tb = False
    winner_sets = 0
    loser_sets = 0
    first_set_winner = None

    for i, s in enumerate(valid_sets):
        a, b = s.split("-")
        a, b = int(a), int(b)
        if a == 7 and b == 6:
            has_tb = True
        elif a == 6 and b == 7:
            has_tb = True

        if a > b:
            winner_sets += 1
            if i == 0:
                first_set_winner = "winner"
        else:
            loser_sets += 1
            if i == 0:
                first_set_winner = "loser"

    deciding = (n_sets == 3 and winner_sets == 2 and loser_sets == 1) or \
               (n_sets == 5 and winner_sets == 3 and loser_sets == 2)
    comeback = first_set_winner == "loser"

    return {
        "n_sets": n_sets,
        "has_tiebreak": has_tb,
        "deciding_set": deciding,
        "comeback": comeback,
        "straight_sets": loser_sets == 0,
    }


# ── Player profile builder ───────────────────────────────────────────────────


class PlayerProfileBuilder:
    """
    Builds comprehensive player profiles from match-level TML data.

    For each player, tracks rolling statistics across multiple dimensions:
    - Serve performance
    - Return performance
    - Win/loss patterns
    - Clutch performance
    - Surface-specific variants
    - Fatigue / recent activity
    """

    # Stats to compute per match (from p1/p2 perspective)
    SERVE_STATS = [
        "ace_rate", "df_rate", "first_serve_pct",
        "first_serve_win_pct", "second_serve_win_pct",
        "serve_point_win_pct", "hold_rate", "bp_save_pct",
    ]

    def __init__(self):
        self.player_history: Dict[str, list] = defaultdict(list)

    def process_match(
        self,
        player: str,
        date: pd.Timestamp,
        surface: str,
        tourney_level: str,
        round_name: str,
        won: bool,
        best_of: int,
        score_str: str,
        minutes: float,
        # Serve stats
        ace: float, double_fault: float, svpt: float,
        first_in: float, first_won: float, second_won: float,
        sv_gms: float, bp_saved: float, bp_faced: float,
        # Opponent serve stats (for return profile)
        opp_ace: float, opp_df: float, opp_svpt: float,
        opp_first_in: float, opp_first_won: float, opp_second_won: float,
        opp_sv_gms: float, opp_bp_saved: float, opp_bp_faced: float,
    ):
        """Record a match for a player."""
        # Compute serve stats
        stats = {
            "date": date,
            "surface": surface,
            "tourney_level": tourney_level,
            "round": round_name,
            "won": int(won),
            "best_of": best_of,
            "minutes": minutes,
        }

        # Serve stats
        if svpt > 0 and not np.isnan(svpt):
            stats["ace_rate"] = (ace / svpt) if not np.isnan(ace) else np.nan
            stats["df_rate"] = (double_fault / svpt) if not np.isnan(double_fault) else np.nan
            stats["first_serve_pct"] = (first_in / svpt) if not np.isnan(first_in) else np.nan
            stats["first_serve_win_pct"] = (first_won / first_in) if (not np.isnan(first_won) and first_in > 0) else np.nan
            stats["second_serve_win_pct"] = (second_won / (svpt - first_in)) if (not np.isnan(second_won) and (svpt - first_in) > 0) else np.nan
            stats["serve_point_win_pct"] = ((first_won + second_won) / svpt) if (not np.isnan(first_won) and not np.isnan(second_won)) else np.nan
        else:
            for s in self.SERVE_STATS:
                stats[s] = np.nan

        # Hold rate
        if sv_gms > 0 and not np.isnan(sv_gms):
            broken = max(0, bp_faced - bp_saved) if (not np.isnan(bp_faced) and not np.isnan(bp_saved)) else 0
            stats["hold_rate"] = (sv_gms - broken) / sv_gms
        else:
            stats["hold_rate"] = np.nan

        # BP save pct
        if bp_faced > 0 and not np.isnan(bp_faced) and not np.isnan(bp_saved):
            stats["bp_save_pct"] = bp_saved / bp_faced
        else:
            stats["bp_save_pct"] = np.nan

        # Return stats (from opponent's serve data)
        if opp_svpt > 0 and not np.isnan(opp_svpt):
            stats["return_point_win_pct"] = 1.0 - ((opp_first_won + opp_second_won) / opp_svpt) if (not np.isnan(opp_first_won) and not np.isnan(opp_second_won)) else np.nan
        else:
            stats["return_point_win_pct"] = np.nan

        if opp_sv_gms > 0 and not np.isnan(opp_sv_gms):
            opp_broken = max(0, opp_bp_faced - opp_bp_saved) if (not np.isnan(opp_bp_faced) and not np.isnan(opp_bp_saved)) else 0
            stats["break_pct"] = opp_broken / opp_sv_gms
        else:
            stats["break_pct"] = np.nan

        # Score-based stats
        score_info = parse_score_info(score_str)
        if score_info:
            stats["has_tiebreak"] = int(score_info["has_tiebreak"])
            stats["tiebreak_won"] = int(score_info["has_tiebreak"] and won)  # approximate
            stats["deciding_set"] = int(score_info["deciding_set"])
            stats["deciding_set_won"] = int(score_info["deciding_set"] and won)
            stats["comeback"] = int(score_info["comeback"])
            stats["straight_sets"] = int(score_info["straight_sets"])
            stats["n_sets_played"] = score_info["n_sets"]
        else:
            for k in ["has_tiebreak", "tiebreak_won", "deciding_set",
                       "deciding_set_won", "comeback", "straight_sets", "n_sets_played"]:
                stats[k] = np.nan

        self.player_history[player].append(stats)

    def build_profiles(self) -> pd.DataFrame:
        """
        Build rolling profiles for all players.
        Returns a DataFrame with per-player rolling stats keyed by (player, date).
        """
        windows = [10, 20, 50]
        rows = []

        for player, history in self.player_history.items():
            if len(history) < 3:
                continue

            df = pd.DataFrame(history)
            df = df.sort_values("date").reset_index(drop=True)

            # Overall rolling stats
            stat_cols = self.SERVE_STATS + [
                "return_point_win_pct", "break_pct",
                "has_tiebreak", "deciding_set", "deciding_set_won",
                "comeback", "straight_sets", "won",
            ]

            for w in windows:
                for col in stat_cols:
                    if col not in df.columns:
                        continue
                    rolling = df[col].rolling(w, min_periods=3).mean().shift(1)
                    df[f"{col}_w{w}"] = rolling

            # Surface-specific rolling (only w=20 for efficiency)
            for surface in ["Hard", "Clay", "Grass"]:
                surf_mask = df["surface"] == surface
                if surf_mask.sum() < 3:
                    continue
                surf_df = df.loc[surf_mask].copy()
                for col in ["won", "hold_rate", "break_pct", "serve_point_win_pct", "return_point_win_pct"]:
                    if col not in surf_df.columns:
                        continue
                    rolling = surf_df[col].rolling(15, min_periods=3).mean().shift(1)
                    df.loc[surf_mask, f"{col}_surf_{surface}"] = rolling

            # Fatigue: matches in last N days (computed separately)
            dates = df["date"].values
            for days in [7, 14, 30]:
                counts = []
                for i in range(len(dates)):
                    cutoff = dates[i] - pd.Timedelta(days=days)
                    n = np.sum((dates[:i] > cutoff))
                    counts.append(n)
                df[f"matches_{days}d"] = counts

            df["player"] = player
            rows.append(df)

        if not rows:
            return pd.DataFrame()

        result = pd.concat(rows, ignore_index=True)
        return result


# ── Main pipeline ─────────────────────────────────────────────────────────────


def build_winner_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build the complete match winner dataset with player profiles.

    Returns (df, num_cols, cat_cols).
    """
    print("Loading data...")
    df = pd.read_csv(MERGED_DATASET_PATH, parse_dates=["Date"], low_memory=False)
    print(f"  Loaded {len(df):,} matches")

    # Filter matches with TML stats
    has_stats = df["p1_svpt"].notna() & df["p2_svpt"].notna()
    df = df[has_stats].copy().reset_index(drop=True)
    print(f"  With TML stats: {len(df):,}")

    # Determine winner (p1 or p2)
    df["_p1_won"] = (df["Winner"] == df["Player_1"]).astype(int)

    # Parse score info
    print("Parsing scores...")
    score_info = df["Score"].apply(parse_score_info)
    for key in ["n_sets", "has_tiebreak", "deciding_set", "comeback", "straight_sets"]:
        df[f"_score_{key}"] = score_info.apply(lambda x: x.get(key) if x else np.nan)

    # Build player profiles
    print("Building player profiles...")
    builder = PlayerProfileBuilder()

    for _, row in df.iterrows():
        p1_won = row["_p1_won"] == 1
        bo = int(row.get("best_of", 3) or 3)
        surface = str(row.get("Surface", ""))
        level = str(row.get("tourney_level", ""))
        round_name = str(row.get("Round", ""))
        minutes = float(row.get("minutes", np.nan))
        score = str(row.get("Score", ""))

        # Process for Player 1
        builder.process_match(
            player=row["Player_1"],
            date=row["Date"],
            surface=surface, tourney_level=level, round_name=round_name,
            won=p1_won, best_of=bo, score_str=score, minutes=minutes,
            ace=row.get("p1_ace", np.nan), double_fault=row.get("p1_df", np.nan),
            svpt=row.get("p1_svpt", np.nan),
            first_in=row.get("p1_1stIn", np.nan),
            first_won=row.get("p1_1stWon", np.nan),
            second_won=row.get("p1_2ndWon", np.nan),
            sv_gms=row.get("p1_SvGms", np.nan),
            bp_saved=row.get("p1_bpSaved", np.nan),
            bp_faced=row.get("p1_bpFaced", np.nan),
            # Opponent (p2) serve stats for p1's return profile
            opp_ace=row.get("p2_ace", np.nan), opp_df=row.get("p2_df", np.nan),
            opp_svpt=row.get("p2_svpt", np.nan),
            opp_first_in=row.get("p2_1stIn", np.nan),
            opp_first_won=row.get("p2_1stWon", np.nan),
            opp_second_won=row.get("p2_2ndWon", np.nan),
            opp_sv_gms=row.get("p2_SvGms", np.nan),
            opp_bp_saved=row.get("p2_bpSaved", np.nan),
            opp_bp_faced=row.get("p2_bpFaced", np.nan),
        )

        # Process for Player 2
        builder.process_match(
            player=row["Player_2"],
            date=row["Date"],
            surface=surface, tourney_level=level, round_name=round_name,
            won=not p1_won, best_of=bo, score_str=score, minutes=minutes,
            ace=row.get("p2_ace", np.nan), double_fault=row.get("p2_df", np.nan),
            svpt=row.get("p2_svpt", np.nan),
            first_in=row.get("p2_1stIn", np.nan),
            first_won=row.get("p2_1stWon", np.nan),
            second_won=row.get("p2_2ndWon", np.nan),
            sv_gms=row.get("p2_SvGms", np.nan),
            bp_saved=row.get("p2_bpSaved", np.nan),
            bp_faced=row.get("p2_bpFaced", np.nan),
            # Opponent (p1) serve stats for p2's return profile
            opp_ace=row.get("p1_ace", np.nan), opp_df=row.get("p1_df", np.nan),
            opp_svpt=row.get("p1_svpt", np.nan),
            opp_first_in=row.get("p1_1stIn", np.nan),
            opp_first_won=row.get("p1_1stWon", np.nan),
            opp_second_won=row.get("p1_2ndWon", np.nan),
            opp_sv_gms=row.get("p1_SvGms", np.nan),
            opp_bp_saved=row.get("p1_bpSaved", np.nan),
            opp_bp_faced=row.get("p1_bpFaced", np.nan),
        )

    print("Computing rolling profiles...")
    profiles = builder.build_profiles()
    print(f"  Profile rows: {len(profiles):,}")

    # ── Merge profiles back to matches ──
    print("Merging profiles to matches...")

    # Create lookup: (player, date) → profile row
    profiles["date"] = pd.to_datetime(profiles["date"])

    # We need to match by (player, date) - merge_asof for closest prior date
    profile_cols = [c for c in profiles.columns if c not in ["date", "player", "surface", "tourney_level", "round", "best_of", "minutes"]]

    # Build lookup dict for speed
    player_profiles = {}
    for player, grp in profiles.groupby("player"):
        grp = grp.sort_values("date")
        player_profiles[player] = grp

    # For each match, get p1 and p2 profile features
    print("  Extracting features per match...")
    p1_features = {}
    p2_features = {}

    for col in profile_cols:
        p1_features[col] = np.full(len(df), np.nan)
        p2_features[col] = np.full(len(df), np.nan)

    for i, row in df.iterrows():
        for prefix, player_col in [("p1", "Player_1"), ("p2", "Player_2")]:
            player = row[player_col]
            date = row["Date"]
            if player not in player_profiles:
                continue
            grp = player_profiles[player]
            # Find most recent profile before this match
            prior = grp[grp["date"] < date]
            if len(prior) == 0:
                continue
            latest = prior.iloc[-1]
            for col in profile_cols:
                if col in latest.index:
                    val = latest[col]
                    if prefix == "p1":
                        p1_features[col][i] = val
                    else:
                        p2_features[col][i] = val

    # ── Build diff/ratio features ──
    print("  Building diff/ratio features...")

    # Stats to create diff features for
    diff_stats = [
        "ace_rate", "df_rate", "first_serve_pct", "first_serve_win_pct",
        "second_serve_win_pct", "serve_point_win_pct", "hold_rate",
        "bp_save_pct", "return_point_win_pct", "break_pct",
        "has_tiebreak", "deciding_set", "deciding_set_won",
        "comeback", "straight_sets", "won",
    ]

    feature_names = []

    # Rolling diff/ratio features
    for w in [10, 20, 50]:
        for stat in diff_stats:
            col = f"{stat}_w{w}"
            if col in p1_features and col in p2_features:
                v1, v2 = p1_features[col], p2_features[col]
                # Diff
                diff_col = f"{stat}_diff_w{w}"
                df[diff_col] = v1 - v2
                feature_names.append(diff_col)
                # Ratio (safe)
                ratio_col = f"{stat}_ratio_w{w}"
                with np.errstate(divide='ignore', invalid='ignore'):
                    df[ratio_col] = np.where(
                        (np.abs(v2) > 1e-8) & ~np.isnan(v2) & ~np.isnan(v1),
                        v1 / v2, np.nan
                    )
                feature_names.append(ratio_col)

    # Surface-specific features
    for surface in ["Hard", "Clay", "Grass"]:
        for stat in ["won", "hold_rate", "break_pct", "serve_point_win_pct", "return_point_win_pct"]:
            col = f"{stat}_surf_{surface}"
            if col in p1_features and col in p2_features:
                diff_col = f"{stat}_diff_surf_{surface}"
                df[diff_col] = p1_features[col] - p2_features[col]
                feature_names.append(diff_col)

    # Fatigue features
    for days in [7, 14, 30]:
        col = f"matches_{days}d"
        if col in p1_features and col in p2_features:
            diff_col = f"fatigue_diff_{days}d"
            df[diff_col] = p1_features[col] - p2_features[col]
            feature_names.append(diff_col)

    # Odds features
    odd1 = pd.to_numeric(df["Odd_1"], errors="coerce")
    odd2 = pd.to_numeric(df["Odd_2"], errors="coerce")

    # Filter invalid odds
    valid_odds = (odd1 > 1) & (odd2 > 1)
    df["implied_prob_1"] = np.where(valid_odds, 1.0 / odd1, np.nan)
    df["implied_prob_2"] = np.where(valid_odds, 1.0 / odd2, np.nan)
    df["prob_diff"] = df["implied_prob_1"] - df["implied_prob_2"]
    df["favorite_strength"] = np.where(valid_odds, 1.0 / np.minimum(odd1, odd2), np.nan)
    df["odds_ratio"] = np.where(
        valid_odds & (odd2 > 0),
        odd1 / odd2, np.nan
    )

    odds_features = ["implied_prob_1", "implied_prob_2", "prob_diff",
                     "favorite_strength", "odds_ratio"]
    feature_names.extend(odds_features)

    # Rank features
    rank1 = pd.to_numeric(df["Rank_1"], errors="coerce")
    rank2 = pd.to_numeric(df["Rank_2"], errors="coerce")
    df["rank_diff"] = rank1 - rank2
    df["rank_diff_abs"] = np.abs(rank1 - rank2)
    feature_names.extend(["rank_diff", "rank_diff_abs"])

    # Points diff
    pts1 = pd.to_numeric(df.get("Pts_1", pd.Series(np.nan, index=df.index)), errors="coerce")
    pts2 = pd.to_numeric(df.get("Pts_2", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["pts_diff"] = pts1 - pts2
    feature_names.append("pts_diff")

    # Context features
    df["_surface"] = df["Surface"].fillna("Hard")
    df["_level"] = df["tourney_level"].fillna("250")
    df["_round"] = df["Round"].fillna("1st Round")
    df["_best_of"] = pd.to_numeric(df.get("best_of", 3), errors="coerce").fillna(3).astype(int)

    cat_cols = ["_surface", "_level", "_round"]
    # Don't add _best_of to cat, keep as numerical

    num_cols = [c for c in feature_names if c in df.columns]

    # Drop rows with no target or no odds
    valid = df["_p1_won"].notna() & df["implied_prob_1"].notna()
    df = df[valid].copy().reset_index(drop=True)

    print(f"  Final dataset: {len(df):,} matches, {len(num_cols)} numerical features, {len(cat_cols)} categorical")
    return df, num_cols, cat_cols


def train_and_evaluate():
    """Train match winner predictor and evaluate against odds."""

    df, num_cols, cat_cols = build_winner_dataset()

    # Temporal split
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_i = np.arange(0, train_end)
    val_i = np.arange(train_end, val_end)
    test_i = np.arange(val_end, n)

    print(f"\nTemporal split:")
    print(f"  Train: {len(train_i):,} ({df['Date'].iloc[train_i[0]]:%Y-%m-%d} → {df['Date'].iloc[train_i[-1]]:%Y-%m-%d})")
    print(f"  Val:   {len(val_i):,} ({df['Date'].iloc[val_i[0]]:%Y-%m-%d} → {df['Date'].iloc[val_i[-1]]:%Y-%m-%d})")
    print(f"  Test:  {len(test_i):,} ({df['Date'].iloc[test_i[0]]:%Y-%m-%d} → {df['Date'].iloc[test_i[-1]]:%Y-%m-%d})")

    # Prepare features
    X_num = df[num_cols].to_numpy(dtype=np.float32)

    # Fill NaN with training median
    medians = np.nanmedian(X_num[train_i], axis=0)
    medians = np.nan_to_num(medians, nan=0.0)
    for j in range(X_num.shape[1]):
        mask = np.isnan(X_num[:, j])
        X_num[mask, j] = medians[j]

    scaler = StandardScaler()
    scaler.fit(X_num[train_i])
    X_num_scaled = scaler.transform(X_num).astype(np.float32)
    scaler.train_medians_ = medians
    scaler.num_cols_ = num_cols

    # Categorical encoding
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if cat_cols:
        encoder.fit(df.loc[train_i, cat_cols].fillna("NA").astype(str))
        X_cat = encoder.transform(df[cat_cols].fillna("NA").astype(str)).astype(np.float32)
    else:
        X_cat = np.zeros((len(df), 0), dtype=np.float32)

    X = np.hstack([X_num_scaled, X_cat]).astype(np.float32)
    y = df["_p1_won"].values.astype(np.float32)

    X_train, X_val, X_test = X[train_i], X[val_i], X[test_i]
    y_train, y_val, y_test = y[train_i], y[val_i], y[test_i]

    # ── Baseline: odds-implied prediction ──
    odds_pred = df["implied_prob_1"].values[test_i]
    odds_auc = roc_auc_score(y_test, odds_pred)
    odds_acc = accuracy_score(y_test, (odds_pred > 0.5).astype(int))
    odds_brier = brier_score_loss(y_test, np.clip(odds_pred, 1e-6, 1 - 1e-6))
    fav_wins = ((odds_pred > 0.5) == y_test.astype(bool)).mean()

    print(f"\n{'='*70}")
    print("BASELINE: Odds-implied prediction")
    print(f"{'='*70}")
    print(f"  AUC:       {odds_auc:.4f}")
    print(f"  Accuracy:  {odds_acc:.4f}  (favorite wins {fav_wins:.1%})")
    print(f"  Brier:     {odds_brier:.4f}")

    # ── Train LightGBM ──
    print(f"\n{'='*70}")
    print("TRAINING LightGBM")
    print(f"{'='*70}")

    params = dict(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        num_leaves=100,
        max_depth=10,
        learning_rate=0.02,
        min_child_samples=80,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_alpha=1.0,
        reg_lambda=0.5,
        n_estimators=5000,
        random_state=RANDOM_SEED,
        verbosity=-1,
        n_jobs=-1,
    )

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(150, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    best_iter = model.best_iteration_ or model.n_estimators
    print(f"  Best iteration: {best_iter}")

    # Refit on train+val
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    refit_params = params.copy()
    refit_params["n_estimators"] = int(best_iter)
    model_final = lgb.LGBMClassifier(**refit_params)
    model_final.fit(X_trainval, y_trainval)

    # Evaluate
    probs = model_final.predict_proba(X_test)[:, 1]
    lgb_auc = roc_auc_score(y_test, probs)
    lgb_acc = accuracy_score(y_test, (probs > 0.5).astype(int))
    lgb_brier = brier_score_loss(y_test, np.clip(probs, 1e-6, 1 - 1e-6))
    lgb_logloss = log_loss(y_test, np.clip(probs, 1e-6, 1 - 1e-6))

    print(f"\n  LightGBM results:")
    print(f"  AUC:       {lgb_auc:.4f}  (odds: {odds_auc:.4f}  diff: {lgb_auc - odds_auc:+.4f})")
    print(f"  Accuracy:  {lgb_acc:.4f}  (odds: {odds_acc:.4f}  diff: {lgb_acc - odds_acc:+.4f})")
    print(f"  Brier:     {lgb_brier:.4f}  (odds: {odds_brier:.4f}  diff: {lgb_brier - odds_brier:+.4f})")
    print(f"  LogLoss:   {lgb_logloss:.4f}")

    # Calibration
    val_probs = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_probs, y_val)
    probs_cal = calibrator.predict(probs)
    probs_cal = np.clip(probs_cal, 1e-6, 1 - 1e-6)
    cal_brier = brier_score_loss(y_test, probs_cal)
    print(f"  Calibrated Brier: {cal_brier:.4f}")

    # ── Feature importance ──
    imp = model_final.feature_importances_
    all_feat_names = num_cols + list(encoder.get_feature_names_out()) if cat_cols else num_cols
    top_idx = np.argsort(imp)[::-1][:25]
    print(f"\n  Top 25 features:")
    for i, idx in enumerate(top_idx):
        if idx < len(all_feat_names):
            print(f"    {i+1:2d}. {all_feat_names[idx]:<40s} importance={imp[idx]:.0f}")

    # ── Segment analysis: where do we beat the odds? ──
    print(f"\n{'='*70}")
    print("SEGMENT ANALYSIS: Where do we beat the odds?")
    print(f"{'='*70}")

    df_test = df.iloc[test_i].copy().reset_index(drop=True)

    # Where model agrees with odds vs disagrees
    model_pick = (probs > 0.5).astype(int)
    odds_pick = (odds_pred > 0.5).astype(int)
    agree = model_pick == odds_pick

    print(f"\n  Model-Odds agreement: {agree.mean():.1%}")
    agree_acc = accuracy_score(y_test[agree], model_pick[agree])
    disagree_acc = accuracy_score(y_test[~agree], model_pick[~agree])
    odds_disagree_acc = accuracy_score(y_test[~agree], odds_pick[~agree])
    print(f"  When agree:        Model Acc={agree_acc:.4f}")
    print(f"  When disagree:     Model Acc={disagree_acc:.4f}  Odds Acc={odds_disagree_acc:.4f}")

    # By confidence level (model vs odds gap)
    edge = probs - odds_pred  # positive = model thinks P1 more likely than odds
    abs_edge = np.abs(edge)
    for lo, hi, label in [(0, 0.03, "low edge (<3%)"), (0.03, 0.08, "medium edge (3-8%)"),
                          (0.08, 0.15, "high edge (8-15%)"), (0.15, 1.0, "very high edge (>15%)")]:
        m = (abs_edge >= lo) & (abs_edge < hi)
        if m.sum() < 20:
            continue
        # Follow the model's pick
        model_correct = (model_pick[m] == y_test[m].astype(int)).mean()
        odds_correct = (odds_pick[m] == y_test[m].astype(int)).mean()
        print(f"  {label:<30s}: N={m.sum():,}  Model={model_correct:.4f}  Odds={odds_correct:.4f}  Δ={model_correct - odds_correct:+.4f}")

    # By surface
    print(f"\n  By Surface:")
    for surf in ["Hard", "Clay", "Grass"]:
        m = df_test["Surface"] == surf
        if m.sum() < 50:
            continue
        auc_m = roc_auc_score(y_test[m], probs[m])
        auc_o = roc_auc_score(y_test[m], odds_pred[m])
        acc_m = accuracy_score(y_test[m], model_pick[m])
        acc_o = accuracy_score(y_test[m], odds_pick[m])
        print(f"    {surf:<10s}: N={m.sum():,}  Model AUC={auc_m:.4f}  Odds AUC={auc_o:.4f}  Δ={auc_m-auc_o:+.4f}  Model Acc={acc_m:.4f}")

    # By tournament level
    print(f"\n  By Tournament Level:")
    for level in ["250", "500", "M", "G"]:
        m = df_test["tourney_level"] == level
        if m.sum() < 50:
            continue
        auc_m = roc_auc_score(y_test[m], probs[m])
        auc_o = roc_auc_score(y_test[m], odds_pred[m])
        acc_m = accuracy_score(y_test[m], model_pick[m])
        acc_o = accuracy_score(y_test[m], odds_pick[m])
        print(f"    {level:<10s}: N={m.sum():,}  Model AUC={auc_m:.4f}  Odds AUC={auc_o:.4f}  Δ={auc_m-auc_o:+.4f}")

    # By round
    print(f"\n  By Round:")
    for rnd in ["1st Round", "2nd Round", "3rd Round", "Quarterfinals", "Semifinals", "The Final"]:
        m = df_test["Round"] == rnd
        if m.sum() < 30:
            continue
        auc_m = roc_auc_score(y_test[m], probs[m])
        auc_o = roc_auc_score(y_test[m], odds_pred[m])
        print(f"    {rnd:<15s}: N={m.sum():,}  Model AUC={auc_m:.4f}  Odds AUC={auc_o:.4f}  Δ={auc_m-auc_o:+.4f}")

    # Save model
    MODELS_ADV_DIR.mkdir(parents=True, exist_ok=True)
    model_final.booster_.save_model(str(MODELS_ADV_DIR / "winner_lgb.txt"))
    joblib.dump(scaler, MODELS_ADV_DIR / "winner_scaler.joblib")
    joblib.dump(encoder, MODELS_ADV_DIR / "winner_encoder.joblib")
    joblib.dump(calibrator, MODELS_ADV_DIR / "winner_calibrator.joblib")
    joblib.dump({
        "input_dim": X.shape[1],
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "best_iter": int(best_iter),
        "test_auc": float(lgb_auc),
        "test_acc": float(lgb_acc),
        "test_brier": float(lgb_brier),
        "odds_auc": float(odds_auc),
        "odds_acc": float(odds_acc),
        "odds_brier": float(odds_brier),
    }, MODELS_ADV_DIR / "winner_meta.joblib")

    metrics = {
        "model_auc": float(lgb_auc),
        "model_acc": float(lgb_acc),
        "model_brier": float(lgb_brier),
        "odds_auc": float(odds_auc),
        "odds_acc": float(odds_acc),
        "odds_brier": float(odds_brier),
        "calibrated_brier": float(cal_brier),
        "n_features": len(num_cols),
        "n_train": len(train_i),
        "n_test": len(test_i),
        "test_period": f"{df['Date'].iloc[test_i[0]]:%Y-%m-%d} → {df['Date'].iloc[test_i[-1]]:%Y-%m-%d}",
        "timestamp": datetime.now().isoformat(),
    }
    with open(MODELS_ADV_DIR / "winner_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {MODELS_ADV_DIR}/winner_*.joblib")
    print(f"Metrics saved to {MODELS_ADV_DIR}/winner_metrics.json")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true", default=True)
    args = parser.parse_args()
    train_and_evaluate()

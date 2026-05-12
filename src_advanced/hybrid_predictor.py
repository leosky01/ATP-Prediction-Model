#!/usr/bin/env python3
"""
Hybrid Markov + ML predictor for tennis O/U.

Stage 1: Estimate p_A, p_B (serve point win rates) from rolling stats
Stage 2: Feed into Markov simulator → get P(over), expected_total
Stage 3: Add Markov outputs as features → final calibrated prediction

Usage:
    python -m src_advanced.hybrid_predictor --evaluate
    python -m src_advanced.hybrid_predictor --tune
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss,
    mean_absolute_error, r2_score, roc_auc_score,
)
from sklearn.isotonic import IsotonicRegression

from .config import (
    GAMES_THRESHOLD_BO3, GAMES_THRESHOLD_BO5,
    MERGED_DATASET_PATH, MODELS_ADV_DIR, RANDOM_SEED,
)
from .feature_cache import load_or_build_features
from .markov_simulator import TennisSimulator
from .train_v3 import temporal_split


# ── Stage 1: Estimate serve point win rates ──────────────────────────────────


def compute_serve_rates_from_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute serve point win rates from TML match stats.

    p_serve = (1stWon + 2ndWon) / svpt

    Also compute game hold rate from:
    hold_rate ≈ games_held / sv_gms ≈ (sv_gms - breaks) / sv_gms
    """
    df = df.copy()

    for prefix in ["p1", "p2"]:
        svpt = df.get(f"{prefix}_svpt", pd.Series(np.nan, index=df.index))
        first_won = df.get(f"{prefix}_1stWon", pd.Series(np.nan, index=df.index))
        second_won = df.get(f"{prefix}_2ndWon", pd.Series(np.nan, index=df.index))
        svgms = df.get(f"{prefix}_SvGms", pd.Series(np.nan, index=df.index))
        bp_saved = df.get(f"{prefix}_bpSaved", pd.Series(np.nan, index=df.index))
        bp_faced = df.get(f"{prefix}_bpFaced", pd.Series(np.nan, index=df.index))
        ace = df.get(f"{prefix}_ace", pd.Series(np.nan, index=df.index))

        # Serve point win rate
        valid = (svpt > 0) & svpt.notna() & first_won.notna() & second_won.notna()
        df[f"{prefix}_serve_pct"] = np.where(
            valid, (first_won + second_won) / svpt, np.nan
        )

        # Game hold rate
        valid_g = (svgms > 0) & svgms.notna()
        broken = np.where(
            bp_faced.notna() & bp_saved.notna(),
            np.maximum(0, bp_faced - bp_saved), 0
        )
        df[f"{prefix}_hold_rate"] = np.where(
            valid_g, np.maximum(0, svgms - broken) / svgms, np.nan
        )

        # Ace rate (per service point)
        df[f"{prefix}_ace_rate"] = np.where(
            valid, ace / svpt, np.nan
        )

    return df


def compute_rolling_serve_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling serve point win rates per player.
    These are the key inputs to the Markov model.
    """
    from .config import ROLLING_WINDOWS

    df = df.copy()
    all_players = set(df["Player_1"].dropna()) | set(df["Player_2"].dropna())

    # Track per-player serve stats
    player_data: Dict[str, list] = {}

    for player in all_players:
        is_p1 = df["Player_1"] == player
        is_p2 = df["Player_2"] == player

        rows = []
        for mask, prefix in [(is_p1, "p1"), (is_p2, "p2")]:
            sub = df.loc[mask, ["Date"]].copy()
            sub["serve_pct"] = df.loc[mask, f"{prefix}_serve_pct"].values
            sub["hold_rate"] = df.loc[mask, f"{prefix}_hold_rate"].values
            sub["ace_rate"] = df.loc[mask, f"{prefix}_ace_rate"].values
            rows.append(sub)

        player_df = pd.concat(rows, ignore_index=True).sort_values("Date")
        player_df = player_df.drop_duplicates(subset=["Date"], keep="last")

        for w in ROLLING_WINDOWS:
            player_df[f"serve_pct_w{w}"] = player_df["serve_pct"].rolling(w, min_periods=3).mean().shift(1)
            player_df[f"hold_rate_w{w}"] = player_df["hold_rate"].rolling(w, min_periods=3).mean().shift(1)
            player_df[f"ace_rate_w{w}"] = player_df["ace_rate"].rolling(w, min_periods=3).mean().shift(1)

        player_data[player] = player_df

    # Merge back to match-level
    # Use merge_asof for speed
    all_rolling = pd.concat(player_data.values(), ignore_index=True)

    for stat in ["serve_pct", "hold_rate", "ace_rate"]:
        for w in ROLLING_WINDOWS:
            col = f"{stat}_w{w}"
            for prefix in ["p1", "p2"]:
                player_col = "Player_1" if prefix == "p1" else "Player_2"
                lookup = all_rolling[["Date", col]].copy()
                lookup = lookup.rename(columns={col: f"{prefix}_{col}"})
                lookup["player"] = lookup.index.map(lambda x: None)  # placeholder

    # Simpler approach: build lookup tables
    for stat in ["serve_pct", "hold_rate", "ace_rate"]:
        for w in ROLLING_WINDOWS:
            col = f"{stat}_w{w}"

            for prefix in ["p1", "p2"]:
                player_col = "Player_1" if prefix == "p1" else "Player_2"
                new_col = f"{prefix}_{col}"

                # Build per-player rolling lookup
                lookup_data = all_rolling[["Date", col]].copy()
                lookup_data["player"] = np.nan  # placeholder

                # Actually, merge by player + date
                # This needs player info... let me use the player_data dict
                vals = np.full(len(df), np.nan)

                # Build a dict: player → DataFrame(Date, col)
                player_lookups = {}
                for player, pdf in player_data.items():
                    if col in pdf.columns:
                        player_lookups[player] = pdf.set_index("Date")[col].sort_index()

                for i, row in df.iterrows():
                    p = row[player_col]
                    if pd.isna(p) or p not in player_lookups:
                        continue
                    lookup = player_lookups[p]
                    # Find most recent value before this date
                    before = lookup.loc[:row["Date"]]
                    if len(before) > 0:
                        vals[i] = before.iloc[-1]

                df[new_col] = vals

    return df


# ── Stage 2+3: Full hybrid pipeline ─────────────────────────────────────────


def run_hybrid_evaluation():
    """
    Run the full hybrid Markov + ML pipeline:
    1. Compute serve rates from TML data
    2. Compute rolling serve rates per player
    3. For each match: estimate p_A, p_B → Markov → P(over)
    4. Evaluate against actual O/U outcomes
    """
    print("=" * 70)
    print("HYBRID MARKOV + ML PREDICTOR")
    print("=" * 70)

    # Load features
    df = load_or_build_features()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"].dt.year >= 2003].reset_index(drop=True)
    print(f"Total matches: {len(df):,}")

    # Stage 1: Compute serve rates
    print("\n1) Computing serve point win rates...")
    df = compute_serve_rates_from_data(df)

    serve_coverage = df["p1_serve_pct"].notna().mean()
    print(f"   Serve pct coverage: {serve_coverage:.1%}")

    # Compute rolling serve stats
    print("\n2) Computing rolling serve rates per player...")
    df = compute_rolling_serve_rates(df)

    # Filter valid games
    valid = (
        df["over_under"].notna() & df["total_games"].notna() & df["best_of"].notna()
        & df["p1_serve_pct_w10"].notna() & df["p2_serve_pct_w10"].notna()
    )
    df_sub = df[valid].copy().reset_index(drop=True)
    df_sub = df_sub[df_sub["total_games"] <= 80].reset_index(drop=True)
    print(f"   Valid for hybrid: {len(df_sub):,}")

    # Temporal split
    train_i, val_i, test_i = temporal_split(df_sub["Date"])
    print(f"   Train: {len(train_i):,}  Val: {len(val_i):,}  Test: {len(test_i):,}")
    print(f"   Test period: {df_sub['Date'].iloc[test_i].min():%Y-%m-%d} → {df_sub['Date'].iloc[test_i].max():%Y-%m-%d}")

    results = {}

    for bo in [3, 5]:
        threshold = GAMES_THRESHOLD_BO3 if bo == 3 else GAMES_THRESHOLD_BO5

        bo_mask = df_sub["best_of"].values == bo
        if bo_mask.sum() < 100:
            continue

        print(f"\n{'='*70}")
        print(f"  BO{bo}  (threshold={threshold})  N={bo_mask.sum():,}")
        print(f"{'='*70}")

        # Prepare data for this BO
        df_bo = df_sub.copy()
        y_ou = df_bo["over_under"].values.astype(np.float32)
        y_total = df_bo["total_games"].values.astype(np.float32)

        # ── Stage 2: Markov predictions using rolling serve rates ──
        print(f"   Stage 2: Running Markov simulator...")

        # Use different rolling windows for robustness
        markov_probs = {}  # window → array of P(over)
        markov_expected = {}  # window → array of expected total

        for w in [10, 20, 50]:
            p1_col = f"p1_serve_pct_w{w}"
            p2_col = f"p2_serve_pct_w{w}"

            probs = np.full(len(df_bo), np.nan)
            expected = np.full(len(df_bo), np.nan)

            p1_vals = df_bo[p1_col].values
            p2_vals = df_bo[p2_col].values

            valid_mask = p1_vals.notna() if hasattr(p1_vals, 'notna') else ~np.isnan(p1_vals)
            valid_mask &= p2_vals.notna() if hasattr(p2_vals, 'notna') else ~np.isnan(p2_vals)

            # Vectorized Markov simulation for valid matches
            valid_idx = np.where(valid_mask)[0]

            # Batch compute for speed
            batch_size = 1000
            for start in range(0, len(valid_idx), batch_size):
                batch_idx = valid_idx[start:start + batch_size]
                for i in batch_idx:
                    pA = float(p1_vals[i]) if not np.isnan(p1_vals[i]) else 0.65
                    pB = float(p2_vals[i]) if not np.isnan(p2_vals[i]) else 0.65
                    sim = TennisSimulator(pA, pB)
                    r = sim.simulate_match(best_of=bo)
                    probs[i] = r["p_over"].get(threshold, 0.5)
                    expected[i] = r["expected_total_games"]

            markov_probs[w] = probs
            markov_expected[w] = expected

            # Evaluate Markov-only on test
            te_mask = bo_mask[test_i]
            te_idx = np.where(te_mask)[0]
            valid_te = ~np.isnan(probs[te_idx])

            if valid_te.sum() > 50:
                markov_auc = roc_auc_score(
                    y_ou[te_idx][valid_te],
                    probs[te_idx][valid_te]
                )
                markov_acc = accuracy_score(
                    y_ou[te_idx][valid_te],
                    (probs[te_idx][valid_te] > 0.5).astype(int)
                )
                print(f"     Markov(w={w}): AUC={markov_auc:.4f}  Acc={markov_acc:.4f}  N={valid_te.sum():,}")

        # Ensemble Markov predictions (average across windows)
        valid_windows = [w for w in [10, 20, 50] if markov_probs[w] is not None]
        markov_ensemble = np.nanmean(
            np.stack([markov_probs[w] for w in valid_windows]), axis=0
        )
        markov_exp_ensemble = np.nanmean(
            np.stack([markov_expected[w] for w in valid_windows]), axis=0
        )

        # Evaluate Markov ensemble on test
        te_mask = bo_mask[test_i]
        te_idx = np.where(te_mask)[0]
        valid_te = ~np.isnan(markov_ensemble[te_idx])

        if valid_te.sum() > 50:
            markov_ens_auc = roc_auc_score(
                y_ou[te_idx][valid_te], markov_ensemble[te_idx][valid_te]
            )
            markov_ens_acc = accuracy_score(
                y_ou[te_idx][valid_te],
                (markov_ensemble[te_idx][valid_te] > 0.5).astype(int)
            )
            markov_mae = mean_absolute_error(
                y_total[te_idx][valid_te],
                markov_exp_ensemble[te_idx][valid_te]
            )
            markov_r2 = r2_score(
                y_total[te_idx][valid_te],
                markov_exp_ensemble[te_idx][valid_te]
            )
            print(f"     Markov ensemble: AUC={markov_ens_auc:.4f}  Acc={markov_ens_acc:.4f}")
            print(f"     Markov total:    MAE={markov_mae:.2f}  R²={markov_r2:.4f}")

        # ── Stage 3: Compare with V3 LightGBM ensemble ──
        print(f"\n   Stage 3: Comparison with V3 ensemble...")
        lgb_ou = lgb.Booster(model_file=str(MODELS_ADV_DIR / f"games_v3_bo{bo}_lgb_ou.txt"))
        from .train_v3 import prepare_features
        X_all, scaler, encoder, num_cols, cat_cols = prepare_features(df_bo, train_i)

        te_mask_x = bo_mask[test_i]
        X_te = X_all[test_i][te_mask_x]
        y_te_ou = y_ou[test_i][te_mask_x]
        y_te_total = y_total[test_i][te_mask_x]

        lgb_probs = lgb_ou.predict(X_te)
        lgb_auc = roc_auc_score(y_te_ou, lgb_probs)
        lgb_acc = accuracy_score(y_te_ou, (lgb_probs > 0.5).astype(int))
        print(f"     V3 LightGBM: AUC={lgb_auc:.4f}  Acc={lgb_acc:.4f}")

        # ── Stage 3b: Hybrid = Markov + LightGBM blend ──
        markov_te = markov_ensemble[test_i][te_mask_x]
        valid_both = ~np.isnan(markov_te)

        if valid_both.sum() > 50:
            # Simple average
            hybrid_simple = 0.5 * lgb_probs[valid_both] + 0.5 * markov_te[valid_both]
            hybrid_auc = roc_auc_score(y_te_ou[valid_both], hybrid_simple)
            hybrid_acc = accuracy_score(y_te_ou[valid_both], (hybrid_simple > 0.5).astype(int))
            print(f"     Hybrid (50/50): AUC={hybrid_auc:.4f}  Acc={hybrid_acc:.4f}  N={valid_both.sum():,}")

            # Weighted blend (optimize weight on val set)
            va_mask = bo_mask[val_i]
            X_va = X_all[val_i][va_mask]
            y_va_ou = y_ou[val_i][va_mask]
            markov_va = markov_ensemble[val_i][va_mask]
            lgb_va = lgb_ou.predict(X_va)
            valid_va = ~np.isnan(markov_va)

            if valid_va.sum() > 50:
                best_w, best_auc = 0.5, 0
                for w in np.arange(0.1, 0.9, 0.05):
                    blend = w * lgb_va[valid_va] + (1 - w) * markov_va[valid_va]
                    auc = roc_auc_score(y_va_ou[valid_va], blend)
                    if auc > best_auc:
                        best_auc, best_w = auc, w

                hybrid_weighted = best_w * lgb_probs[valid_both] + (1 - best_w) * markov_te[valid_both]
                hybrid_w_auc = roc_auc_score(y_te_ou[valid_both], hybrid_weighted)
                hybrid_w_acc = accuracy_score(y_te_ou[valid_both], (hybrid_weighted > 0.5).astype(int))
                print(f"     Hybrid ({best_w:.0%}/{1-best_w:.0%}): AUC={hybrid_w_auc:.4f}  Acc={hybrid_w_acc:.4f}")

                # Calibrate with isotonic
                blend_va = best_w * lgb_va[valid_va] + (1 - best_w) * markov_va[valid_va]
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(blend_va, y_va_ou[valid_va])
                hybrid_cal = cal.predict(hybrid_weighted)
                hybrid_cal_auc = roc_auc_score(y_te_ou[valid_both], hybrid_cal)
                hybrid_cal_acc = accuracy_score(y_te_ou[valid_both], (hybrid_cal > 0.5).astype(int))
                hybrid_cal_brier = brier_score_loss(y_te_ou[valid_both], hybrid_cal)
                print(f"     Hybrid calibrated: AUC={hybrid_cal_auc:.4f}  Acc={hybrid_cal_acc:.4f}  Brier={hybrid_cal_brier:.4f}")

                # ── Segment analysis for the hybrid model ──
                print(f"\n   ── Hybrid Model Segment Analysis ──")
                df_te = df_bo.iloc[test_i][te_mask_x].copy().reset_index(drop=True)
                df_te_valid = df_te[valid_both].copy().reset_index(drop=True)
                y_seg = y_te_ou[valid_both]
                p_seg = hybrid_cal

                # By confidence
                for lo, hi, label in [(0, 0.40, "strong UNDER"), (0.40, 0.48, "weak UNDER"),
                                      (0.48, 0.52, "uncertain"), (0.52, 0.60, "weak OVER"),
                                      (0.60, 1.01, "strong OVER")]:
                    m = (p_seg >= lo) & (p_seg < hi)
                    if m.sum() < 20:
                        continue
                    acc = accuracy_score(y_seg[m], (p_seg[m] > 0.5).astype(int))
                    print(f"     {label:<20s}: Acc={acc:.4f}  N={m.sum():,}  O/U rate={y_seg[m].mean():.3f}")

                # By surface
                if "Surface" in df_te_valid.columns:
                    for surf in ["Hard", "Clay", "Grass"]:
                        m = df_te_valid["Surface"] == surf
                        if m.sum() < 30:
                            continue
                        try:
                            auc_s = roc_auc_score(y_seg[m], p_seg[m])
                            acc_s = accuracy_score(y_seg[m], (p_seg[m] > 0.5).astype(int))
                            print(f"     {surf:<20s}: AUC={auc_s:.4f}  Acc={acc_s:.4f}  N={m.sum():,}")
                        except:
                            pass

                # By combined hold rate
                if "p1_hold_rate" in df_te_valid.columns:
                    hold_sum = (
                        df_te_valid["p1_hold_rate"].fillna(0.8).values +
                        df_te_valid["p2_hold_rate"].fillna(0.8).values
                    )
                    for lo, hi, label in [(0, 1.5, "low hold"), (1.5, 1.7, "medium hold"), (1.7, 2.1, "high hold")]:
                        m = (hold_sum >= lo) & (hold_sum < hi)
                        if m.sum() < 30:
                            continue
                        try:
                            auc_s = roc_auc_score(y_seg[m], p_seg[m])
                            acc_s = accuracy_score(y_seg[m], (p_seg[m] > 0.5).astype(int))
                            print(f"     {label:<20s}: AUC={auc_s:.4f}  Acc={acc_s:.4f}  N={m.sum():,}")
                        except:
                            pass

        results[f"bo{bo}"] = {
            "markov_auc": float(markov_ens_auc) if valid_te.sum() > 50 else None,
            "lgbm_auc": float(lgb_auc),
            "hybrid_auc": float(hybrid_auc) if valid_both.sum() > 50 else None,
            "hybrid_cal_auc": float(hybrid_cal_auc) if valid_both.sum() > 50 else None,
            "n_test": int(valid_both.sum()) if valid_both is not None else 0,
        }

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for bo_key, r in results.items():
        print(f"  {bo_key.upper()}:")
        print(f"    Markov only:   AUC={r.get('markov_auc', 'N/A')}")
        print(f"    LightGBM only: AUC={r['lgbm_auc']:.4f}")
        print(f"    Hybrid:        AUC={r.get('hybrid_auc', 'N/A')}")
        print(f"    Hybrid (cal):  AUC={r.get('hybrid_cal_auc', 'N/A')}")

    # Save
    out_path = MODELS_ADV_DIR / "hybrid_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true", default=True)
    args = parser.parse_args()
    run_hybrid_evaluation()

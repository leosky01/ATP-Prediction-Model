#!/usr/bin/env python3
"""
Segment analysis: find sub-groups where the O/U model performs best.
Useful for building a betting strategy around high-accuracy segments.

Usage:
    python -m src_advanced.segment_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score

from .config import MERGED_DATASET_PATH, MODELS_ADV_DIR, RANDOM_SEED
from .feature_cache import load_or_build_features
from .feature_engineering import get_feature_columns
from .train_v3 import temporal_split, prepare_features


def analyze_segments():
    print("=" * 70)
    print("SEGMENT ANALYSIS — Finding high-accuracy sub-groups")
    print("=" * 70)

    # Load features (cached)
    df = load_or_build_features()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"].dt.year >= 2003].reset_index(drop=True)
    print(f"Total matches: {len(df):,}")

    # Filter valid games
    valid = df["over_under"].notna() & df["total_games"].notna() & df["best_of"].notna()
    df_sub = df[valid].copy().reset_index(drop=True)
    df_sub = df_sub[df_sub["total_games"] <= 80].reset_index(drop=True)
    print(f"Valid games: {len(df_sub):,}")

    # Temporal split (same as training)
    train_i, val_i, test_i = temporal_split(df_sub["Date"])
    print(f"Test set: {len(test_i):,} matches")
    print(f"Test period: {df_sub['Date'].iloc[test_i].min():%Y-%m-%d} → {df_sub['Date'].iloc[test_i].max():%Y-%m-%d}")

    df_test = df_sub.iloc[test_i].copy().reset_index(drop=True)
    test_idx_global = test_i

    # Load V3 model predictions (we need to recompute them)
    # Load scaler/encoder
    scaler = joblib.load(MODELS_ADV_DIR / "games_v3_scaler.joblib")
    encoder = joblib.load(MODELS_ADV_DIR / "games_v3_encoder.joblib")
    meta = joblib.load(MODELS_ADV_DIR / "games_v3_meta.joblib")

    # Prepare features for full dataset
    X, _, _, num_cols, cat_cols = prepare_features(df_sub, train_i)

    results = {}

    for bo in [3, 5]:
        print(f"\n{'='*70}")
        print(f"  BEST OF {bo}")
        print(f"{'='*70}")

        bo_mask_test = df_sub.iloc[test_i]["best_of"].values == bo
        if bo_mask_test.sum() < 100:
            print(f"  Not enough data for BO{bo}")
            continue

        X_te = X[test_i][bo_mask_test]
        y_te = df_sub.iloc[test_i]["over_under"].values[bo_mask_test]
        df_bo = df_sub.iloc[test_i][bo_mask_test].copy().reset_index(drop=True)

        # Get predictions from ensemble
        # 1) LGBM
        lgb_ou = lgb.Booster(model_file=str(MODELS_ADV_DIR / f"games_v3_bo{bo}_lgb_ou.txt"))
        lgb_probs = lgb_ou.predict(X_te)

        # 2) NN ensemble
        import torch
        from .train_v3 import OUNet, predict_nn
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nn_probs_list = []
        for si in range(meta.get(f"bo{bo}_n_nn_seeds", 3)):
            p = MODELS_ADV_DIR / f"games_v3_bo{bo}_nn_s{si}.pt"
            if not p.exists():
                continue
            m = OUNet(meta.get("input_dim", 196)).to(device)
            m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
            m.eval()
            nn_probs_list.append(predict_nn(m, X_te))

        nn_probs = np.mean(nn_probs_list, axis=0) if nn_probs_list else lgb_probs

        # 3) Stacker + calibrator
        stacker = joblib.load(MODELS_ADV_DIR / f"games_v3_bo{bo}_stacker.joblib")
        calibrator = joblib.load(MODELS_ADV_DIR / f"games_v3_bo{bo}_calibrator.joblib")

        stack_input = np.column_stack([lgb_probs, nn_probs])
        ens_raw = stacker.predict_proba(stack_input)[:, 1]
        ens_probs = calibrator.predict(ens_raw)
        ens_probs = np.clip(ens_probs, 1e-6, 1 - 1e-6)

        # Overall metrics
        overall_auc = roc_auc_score(y_te, ens_probs)
        overall_acc = accuracy_score(y_te, (ens_probs > 0.5).astype(int))
        overall_brier = brier_score_loss(y_te, ens_probs)
        print(f"  Overall: AUC={overall_auc:.4f}  Acc={overall_acc:.4f}  Brier={overall_brier:.4f}  N={len(y_te):,}")

        # ── SEGMENT ANALYSIS ──
        segments = {}

        # 1) By Surface
        print(f"\n  ── By Surface ──")
        for surf in df_bo["Surface"].dropna().unique():
            mask = df_bo["Surface"] == surf
            if mask.sum() < 50:
                continue
            auc = roc_auc_score(y_te[mask], ens_probs[mask])
            acc = accuracy_score(y_te[mask], (ens_probs[mask] > 0.5).astype(int))
            bri = brier_score_loss(y_te[mask], ens_probs[mask])
            segments[f"surface_{surf}"] = {"auc": auc, "acc": acc, "brier": bri, "n": int(mask.sum()),
                                           "ou_rate": float(y_te[mask].mean())}
            print(f"    {surf:<12s}: AUC={auc:.4f}  Acc={acc:.4f}  N={mask.sum():,}  O/U rate={y_te[mask].mean():.3f}")

        # 2) By Tournament Level
        print(f"\n  ── By Tournament Level ──")
        if "tourney_level" in df_bo.columns:
            for level in df_bo["tourney_level"].dropna().unique():
                mask = df_bo["tourney_level"] == level
                if mask.sum() < 50:
                    continue
                auc = roc_auc_score(y_te[mask], ens_probs[mask])
                acc = accuracy_score(y_te[mask], (ens_probs[mask] > 0.5).astype(int))
                segments[f"level_{level}"] = {"auc": auc, "acc": acc, "n": int(mask.sum()),
                                              "ou_rate": float(y_te[mask].mean())}
                print(f"    {level:<12s}: AUC={auc:.4f}  Acc={acc:.4f}  N={mask.sum():,}")

        # 3) By Round
        print(f"\n  ── By Round ──")
        round_col = "Round" if "Round" in df_bo.columns else "round"
        if round_col in df_bo.columns:
            for rnd in df_bo[round_col].dropna().unique():
                mask = df_bo[round_col] == rnd
                if mask.sum() < 50:
                    continue
                auc = roc_auc_score(y_te[mask], ens_probs[mask])
                acc = accuracy_score(y_te[mask], (ens_probs[mask] > 0.5).astype(int))
                segments[f"round_{rnd}"] = {"auc": auc, "acc": acc, "n": int(mask.sum()),
                                            "ou_rate": float(y_te[mask].mean())}
                print(f"    {str(rnd):<12s}: AUC={auc:.4f}  Acc={acc:.4f}  N={mask.sum():,}")

        # 4) By odds range (favorite strength)
        print(f"\n  ── By Favorite Strength (odds) ──")
        if "favorite_strength" in df_bo.columns:
            fs = df_bo["favorite_strength"].values
            bins = [(0, 0.45, "weak_fav (<0.45)"), (0.45, 0.55, "competitive (0.45-0.55)"),
                    (0.55, 0.70, "medium_fav (0.55-0.70)"), (0.70, 1.01, "strong_fav (>0.70)")]
            for lo, hi, label in bins:
                mask = (fs >= lo) & (fs < hi) & ~np.isnan(fs)
                if mask.sum() < 50:
                    continue
                auc = roc_auc_score(y_te[mask], ens_probs[mask])
                acc = accuracy_score(y_te[mask], (ens_probs[mask] > 0.5).astype(int))
                segments[f"fav_{label}"] = {"auc": auc, "acc": acc, "n": int(mask.sum()),
                                            "ou_rate": float(y_te[mask].mean())}
                print(f"    {label:<30s}: AUC={auc:.4f}  Acc={acc:.4f}  N={mask.sum():,}")

        # 5) By confidence bucket (model's own probability)
        print(f"\n  ── By Model Confidence ──")
        conf_bins = [(0.0, 0.40, "low (p<0.40)"), (0.40, 0.48, "weak_under"),
                     (0.48, 0.52, "uncertain"), (0.52, 0.60, "weak_over"),
                     (0.60, 0.70, "moderate (0.60-0.70)"), (0.70, 1.01, "high (p>0.70)")]
        for lo, hi, label in conf_bins:
            mask = (ens_probs >= lo) & (ens_probs < hi)
            if mask.sum() < 30:
                continue
            acc = accuracy_score(y_te[mask], (ens_probs[mask] > 0.5).astype(int))
            ou_rate = y_te[mask].mean()
            predicted_over = (ens_probs[mask] > 0.5).mean()
            correct = ((ens_probs[mask] > 0.5).astype(int) == y_te[mask]).mean()
            segments[f"confidence_{label}"] = {"acc": acc, "n": int(mask.sum()),
                                               "ou_rate": float(ou_rate),
                                               "predicted_over_rate": float(predicted_over)}
            print(f"    {label:<25s}: Acc={correct:.4f}  N={mask.sum():,}  O/U rate={ou_rate:.3f}  pred_over={predicted_over:.3f}")

        # 6) Combined: Surface × Tournament Level
        print(f"\n  ── Combined: Surface × Level ──")
        if "tourney_level" in df_bo.columns and "Surface" in df_bo.columns:
            for surf in ["Hard", "Clay", "Grass"]:
                for level in df_bo["tourney_level"].dropna().unique():
                    mask = (df_bo["Surface"] == surf) & (df_bo["tourney_level"] == level)
                    if mask.sum() < 30:
                        continue
                    auc = roc_auc_score(y_te[mask], ens_probs[mask])
                    acc = accuracy_score(y_te[mask], (ens_probs[mask] > 0.5).astype(int))
                    key = f"{surf}_{level}"
                    segments[f"combo_{key}"] = {"auc": auc, "acc": acc, "n": int(mask.sum()),
                                               "ou_rate": float(y_te[mask].mean())}
                    print(f"    {key:<20s}: AUC={auc:.4f}  Acc={acc:.4f}  N={mask.sum():,}")

        # 7) By implied total games (predicted)
        print(f"\n  ── By Predicted Total Games Range ──")
        lgb_total = lgb.Booster(model_file=str(MODELS_ADV_DIR / f"games_v3_bo{bo}_lgb_total.txt"))
        total_preds = lgb_total.predict(X_te)
        thresholds = {3: 22.5, 5: 38.5}
        th = thresholds[bo]
        # Distance from threshold
        dist = total_preds - th
        dist_bins = [(-999, -5, "well_under (<th-5)"), (-5, -2, "moderate_under"),
                     (-2, 2, "near_threshold"), (2, 5, "moderate_over"),
                     (5, 999, "well_over (>th+5)")]
        for lo, hi, label in dist_bins:
            mask = (dist >= lo) & (dist < hi)
            if mask.sum() < 30:
                continue
            auc = roc_auc_score(y_te[mask], ens_probs[mask])
            acc = accuracy_score(y_te[mask], (ens_probs[mask] > 0.5).astype(int))
            segments[f"total_dist_{label}"] = {"auc": auc, "acc": acc, "n": int(mask.sum()),
                                               "mean_pred_total": float(total_preds[mask].mean()),
                                               "mean_actual_total": float(df_bo["total_games"].values[mask].mean())}
            print(f"    {label:<25s}: AUC={auc:.4f}  Acc={acc:.4f}  N={mask.sum():,}  "
                  f"pred_total={total_preds[mask].mean():.1f}  actual={df_bo['total_games'].values[mask].mean():.1f}")

        # Sort all segments by AUC
        print(f"\n  {'='*70}")
        print(f"  TOP 15 SEGMENTS (by AUC) — BO{bo}")
        print(f"  {'='*70}")
        sorted_segs = sorted(
            [(k, v) for k, v in segments.items() if "auc" in v and v["n"] >= 30],
            key=lambda x: x[1].get("auc", 0), reverse=True
        )
        for i, (name, m) in enumerate(sorted_segs[:15]):
            print(f"  {i+1:2d}. {name:<35s} AUC={m['auc']:.4f}  Acc={m.get('acc',0):.4f}  N={m['n']:,}  O/U={m.get('ou_rate',0):.3f}")

        results[f"bo{bo}"] = {
            "overall_auc": overall_auc,
            "overall_acc": overall_acc,
            "overall_brier": overall_brier,
            "n_test": int(len(y_te)),
            "segments": segments,
            "top_segments": [(k, v) for k, v in sorted_segs[:20]],
        }

    # Save
    out_path = MODELS_ADV_DIR / "segment_analysis.json"
    # Convert tuples to dicts for JSON serialization
    json_results = {}
    for bo_key, bo_data in results.items():
        json_results[bo_key] = {
            "overall_auc": bo_data["overall_auc"],
            "overall_acc": bo_data["overall_acc"],
            "overall_brier": bo_data["overall_brier"],
            "n_test": bo_data["n_test"],
            "segments": bo_data["segments"],
            "top_segments": [{"name": k, **v} for k, v in bo_data["top_segments"]],
        }
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # ── BETTING STRATEGY SUMMARY ──
    print(f"\n{'='*70}")
    print("BETTING STRATEGY IMPLICATIONS")
    print(f"{'='*70}")
    for bo_key in results:
        bo_data = results[bo_key]
        top = bo_data["top_segments"]
        print(f"\n  {bo_key.upper()} — Best segments:")
        for name, m in top[:5]:
            edge = m.get("auc", 0.5) - 0.5
            if edge > 0.05:
                print(f"    * {name:<35s} AUC={m['auc']:.4f}  Acc={m.get('acc',0):.4f}  N={m['n']:,}")
                print(f"      → Edge over random: +{edge:.3f} (potentially profitable)")

    return results


if __name__ == "__main__":
    analyze_segments()

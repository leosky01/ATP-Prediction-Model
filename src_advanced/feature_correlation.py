#!/usr/bin/env python3
"""
ATP Advanced Prediction System – Feature Correlation Analysis

Computes Pearson correlation and Mutual Information between all features
and prediction targets (over_under, total_games, minutes).
Generates heatmap, CSV rankings, and console report.

Usage:
    python -m src_advanced.feature_correlation --target all
    python -m src_advanced.feature_correlation --target over_under
    python -m src_advanced.feature_correlation --target total_games
    python -m src_advanced.feature_correlation --target minutes
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from .config import FIGURES_ADV_DIR, MERGED_DATASET_PATH
from .feature_engineering import build_advanced_features, get_feature_columns

# ── Targets ──────────────────────────────────────────────────────────────────

TARGETS = {
    "over_under": {"col": "over_under", "type": "classification"},
    "total_games": {"col": "total_games", "type": "regression"},
    "minutes": {"col": "minutes", "type": "regression"},
}


# ── Pearson correlation ─────────────────────────────────────────────────────

def compute_pearson(df: pd.DataFrame, num_cols: list, target_col: str) -> pd.Series:
    """Compute Pearson |r| for each numerical feature vs target."""
    valid = df[target_col].notna()
    df_valid = df.loc[valid, num_cols + [target_col]].fillna(0)
    correlations = {}
    for col in num_cols:
        if df_valid[col].std() == 0:
            correlations[col] = 0.0
        else:
            correlations[col] = df_valid[col].corr(df_valid[target_col])
    return pd.Series(correlations, name=target_col).abs().sort_values(ascending=False)


# ── Mutual Information ──────────────────────────────────────────────────────

def compute_mutual_info(
    df: pd.DataFrame, num_cols: list, target_col: str, task_type: str
) -> pd.Series:
    """Compute Mutual Information for each numerical feature vs target."""
    valid = df[target_col].notna()
    df_valid = df.loc[valid, num_cols + [target_col]].fillna(0)

    X = df_valid[num_cols].values.astype(np.float32)
    y = df_valid[target_col].values

    if task_type == "classification":
        y = y.astype(int)
        mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    else:
        mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)

    mi_series = pd.Series(mi, index=num_cols, name=target_col).sort_values(
        ascending=False
    )
    return mi_series


# ── Heatmap ─────────────────────────────────────────────────────────────────

def plot_heatmap(
    pearson_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    odds_features: list,
    output_path: Path,
):
    """Generate correlation heatmap for odds features vs targets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(odds_features) * 0.4)))

    # Pearson heatmap (odds features only)
    odds_pearson = pearson_df.loc[pearson_df.index.isin(odds_features)]
    if not odds_pearson.empty:
        im1 = axes[0].imshow(odds_pearson.values, cmap="RdBu_r", aspect="auto", vmin=0, vmax=0.5)
        axes[0].set_xticks(range(len(odds_pearson.columns)))
        axes[0].set_xticklabels(odds_pearson.columns, rotation=45, ha="right")
        axes[0].set_yticks(range(len(odds_pearson.index)))
        axes[0].set_yticklabels(odds_pearson.index)
        axes[0].set_title("Pearson |r|: Odds Features vs Targets")
        fig.colorbar(im1, ax=axes[0], shrink=0.8)

        # Annotate cells
        for i in range(len(odds_pearson.index)):
            for j in range(len(odds_pearson.columns)):
                val = odds_pearson.values[i, j]
                axes[0].text(j, i, f"{val:.3f}", ha="center", va="center",
                           fontsize=8, color="white" if val > 0.3 else "black")

    # MI heatmap (odds features only)
    odds_mi = mi_df.loc[mi_df.index.isin(odds_features)]
    if not odds_mi.empty:
        im2 = axes[1].imshow(odds_mi.values, cmap="YlOrRd", aspect="auto")
        axes[1].set_xticks(range(len(odds_mi.columns)))
        axes[1].set_xticklabels(odds_mi.columns, rotation=45, ha="right")
        axes[1].set_yticks(range(len(odds_mi.index)))
        axes[1].set_yticklabels(odds_mi.index)
        axes[1].set_title("Mutual Information: Odds Features vs Targets")
        fig.colorbar(im2, ax=axes[1], shrink=0.8)

        for i in range(len(odds_mi.index)):
            for j in range(len(odds_mi.columns)):
                val = odds_mi.values[i, j]
                axes[1].text(j, i, f"{val:.4f}", ha="center", va="center",
                           fontsize=8, color="white" if val > 0.02 else "black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Heatmap saved: {output_path}")


# ── Main analysis ───────────────────────────────────────────────────────────

def run_analysis(target_names: list[str]):
    """Run correlation analysis for specified targets."""
    print("=" * 60)
    print("ATP ADVANCED – FEATURE CORRELATION ANALYSIS")
    print(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    if not MERGED_DATASET_PATH.exists():
        print("ERROR: Merged dataset not found. Run training first.")
        return

    df = pd.read_csv(MERGED_DATASET_PATH, parse_dates=["Date"])
    print(f"   Loaded {len(df):,} matches")

    df = build_advanced_features(df)

    num_cols, _ = get_feature_columns(df)
    print(f"   Features: {len(num_cols)} numerical")

    # Identify odds-related features
    odds_features = [
        f for f in num_cols
        if any(kw in f.lower() for kw in ["odd", "prob", "competitiveness", "favorite"])
    ]
    print(f"   Odds-related features: {len(odds_features)}")

    # Filter valid targets
    valid_targets = {}
    for name in target_names:
        if name not in TARGETS:
            print(f"   WARNING: unknown target '{name}', skipping")
            continue
        tcol = TARGETS[name]["col"]
        valid_count = df[tcol].notna().sum()
        print(f"   Target '{name}': {valid_count:,} valid rows")
        if valid_count > 100:
            valid_targets[name] = TARGETS[name]

    if not valid_targets:
        print("ERROR: No valid targets to analyze.")
        return

    # Compute Pearson correlations
    print("\n--- Pearson |r| Correlations ---")
    pearson_results = {}
    for name, tinfo in valid_targets.items():
        pearson_results[name] = compute_pearson(df, num_cols, tinfo["col"])
        top20 = pearson_results[name].head(20)
        print(f"\n   Top 20 features for {name} (Pearson |r|):")
        for feat, val in top20.items():
            marker = " <-- ODDS" if feat in odds_features else ""
            print(f"     {feat:<40} {val:.4f}{marker}")

    # Compute Mutual Information
    print("\n--- Mutual Information ---")
    mi_results = {}
    for name, tinfo in valid_targets.items():
        print(f"\n   Computing MI for {name}...")
        mi_results[name] = compute_mutual_info(df, num_cols, tinfo["col"], tinfo["type"])
        top20 = mi_results[name].head(20)
        print(f"   Top 20 features for {name} (MI):")
        for feat, val in top20.items():
            marker = " <-- ODDS" if feat in odds_features else ""
            print(f"     {feat:<40} {val:.4f}{marker}")

    # Build combined DataFrames
    pearson_df = pd.DataFrame(pearson_results).fillna(0)
    mi_df = pd.DataFrame(mi_results).fillna(0)

    # Save CSV rankings
    FIGURES_ADV_DIR.mkdir(parents=True, exist_ok=True)

    pearson_csv = FIGURES_ADV_DIR / "pearson_correlations.csv"
    mi_csv = FIGURES_ADV_DIR / "mutual_information.csv"

    pearson_df.to_csv(pearson_csv)
    mi_df.to_csv(mi_csv)
    print(f"\n   Pearson CSV: {pearson_csv}")
    print(f"   MI CSV:      {mi_csv}")

    # Generate heatmap
    heatmap_path = FIGURES_ADV_DIR / "odds_correlation_heatmap.png"
    plot_heatmap(pearson_df, mi_df, odds_features, heatmap_path)

    # Summary: odds features in top-10
    print("\n--- Odds Features in Top-10 ---")
    for name in valid_targets:
        p_top10 = set(pearson_results[name].head(10).index)
        m_top10 = set(mi_results[name].head(10).index)
        p_odds = p_top10 & set(odds_features)
        m_odds = m_top10 & set(odds_features)
        print(f"   {name}:")
        print(f"     Pearson top-10 odds features: {p_odds if p_odds else 'none'}")
        print(f"     MI top-10 odds features:      {m_odds if m_odds else 'none'}")

    print(f"\nDone: {datetime.now():%Y-%m-%d %H:%M:%S}")


def main():
    parser = argparse.ArgumentParser(description="Feature Correlation Analysis")
    parser.add_argument(
        "--target", type=str, default="all",
        choices=["all", "over_under", "total_games", "minutes"],
        help="Which target(s) to analyze",
    )
    args = parser.parse_args()

    if args.target == "all":
        targets = list(TARGETS.keys())
    else:
        targets = [args.target]

    run_analysis(targets)


if __name__ == "__main__":
    main()

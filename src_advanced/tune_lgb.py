#!/usr/bin/env python3
"""
LightGBM hyperparameter tuning with Optuna for O/U prediction.

Usage:
    python -m src_advanced.tune_lgb --n-trials 100
    python -m src_advanced.tune_lgb --n-trials 50 --study-name lgb_v2
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score

from .config import MERGED_DATASET_PATH, MODELS_ADV_DIR, RANDOM_SEED
from .feature_engineering import build_advanced_features, get_feature_columns
from .train import make_temporal_splits, prepare_features


def objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_ou_train: np.ndarray,
    y_total_train: np.ndarray,
    X_val: np.ndarray,
    y_ou_val: np.ndarray,
    y_total_val: np.ndarray,
) -> float:
    """Optuna objective: maximize O/U AUC on validation set."""

    # Hyperparameter search space
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 3.0),
        "random_state": RANDOM_SEED,
    }

    # Train O/U model
    model_ou = lgb.LGBMClassifier(**params)
    model_ou.fit(
        X_train, y_ou_train,
        eval_set=[(X_val, y_ou_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # Predict and compute AUC
    ou_probs = model_ou.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_ou_val, ou_probs)

    # Also train a total games regression model with same params
    reg_params = params.copy()
    reg_params["objective"] = "regression"
    reg_params["metric"] = "mae"

    model_total = lgb.LGBMRegressor(**reg_params)
    model_total.fit(
        X_train, y_total_train,
        eval_set=[(X_val, y_total_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    total_preds = model_total.predict(X_val)
    valid_mask = ~np.isnan(y_total_val)
    if valid_mask.any():
        mae = mean_absolute_error(y_total_val[valid_mask], total_preds[valid_mask])
        r2 = r2_score(y_total_val[valid_mask], total_preds[valid_mask])
        # Report total games metrics as supplementary
        trial.set_user_attr("total_mae", mae)
        trial.set_user_attr("total_r2", r2)

    trial.set_user_attr("best_iteration_ou", model_ou.best_iteration_)
    if hasattr(model_total, "best_iteration_"):
        trial.set_user_attr("best_iteration_total", model_total.best_iteration_)

    return auc


def run_tuning(n_trials: int = 100, study_name: str = "lgb_ou_v3"):
    """Run Optuna hyperparameter optimization."""
    import pandas as pd

    print("=" * 60)
    print("LIGHTGBM OPTUNA TUNING")
    print(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Trials: {n_trials}")
    print("=" * 60)

    # Load data
    print("\n1) Loading data...")
    df = pd.read_csv(MERGED_DATASET_PATH, parse_dates=["Date"])
    print(f"   Loaded {len(df):,} matches")

    # Build features (includes new O/U-specific features)
    print("\n2) Building features...")
    df = build_advanced_features(df)

    # Filter valid games
    valid = df["over_under"].notna() & df["total_games"].notna()
    df_sub = df[valid].copy().reset_index(drop=True)
    print(f"   Valid matches: {len(df_sub):,}")

    # Temporal split
    train_i, val_i, test_i = make_temporal_splits(len(df_sub))
    print(f"   Split: train={len(train_i):,} val={len(val_i):,} test={len(test_i):,}")

    # Prepare features
    X, scaler, encoder, num_cols, cat_cols = prepare_features(df_sub, train_i)
    print(f"   Feature dim: {X.shape[1]}")

    y_ou = df_sub["over_under"].values.astype(np.float32)
    y_total = df_sub["total_games"].values.astype(np.float32)

    X_train, X_val, X_test = X[train_i], X[val_i], X[test_i]
    y_ou_train, y_ou_val, y_ou_test = y_ou[train_i], y_ou[val_i], y_ou[test_i]
    y_total_train, y_total_val, y_total_test = y_total[train_i], y_total[val_i], y_total[test_i]

    # Run Optuna
    print(f"\n3) Starting Optuna optimization ({n_trials} trials)...")
    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
    )

    study.optimize(
        lambda trial: objective(
            trial, X_train, y_ou_train, y_total_train,
            X_val, y_ou_val, y_total_val,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Results
    print("\n" + "=" * 60)
    print("TUNING RESULTS")
    print("=" * 60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best O/U AUC (val): {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")

    if "total_mae" in study.best_trial.user_attrs:
        print(f"Total Games MAE (val): {study.best_trial.user_attrs['total_mae']:.2f}")
    if "total_r2" in study.best_trial.user_attrs:
        print(f"Total Games R^2 (val): {study.best_trial.user_attrs['total_r2']:.4f}")

    # Train final model with best params on train+val, evaluate on test
    print("\n4) Training final model with best params on train+val...")
    best_params = study.best_params.copy()
    best_params["random_state"] = RANDOM_SEED
    best_params["verbosity"] = -1

    # O/U classifier
    ou_params = best_params.copy()
    ou_params["objective"] = "binary"
    ou_params["metric"] = "auc"
    X_trainval = np.vstack([X_train, X_val])
    y_ou_trainval = np.concatenate([y_ou_train, y_ou_val])
    y_total_trainval = np.concatenate([y_total_train, y_total_val])

    best_iter_ou = study.best_trial.user_attrs.get("best_iteration_ou", None)
    if best_iter_ou is not None and best_iter_ou > 0:
        ou_params["n_estimators"] = best_iter_ou

    model_ou = lgb.LGBMClassifier(**ou_params)
    model_ou.fit(X_trainval, y_ou_trainval)

    # Total games regressor
    total_params = best_params.copy()
    total_params["objective"] = "regression"
    total_params["metric"] = "mae"
    best_iter_total = study.best_trial.user_attrs.get("best_iteration_total", None)
    if best_iter_total is not None and best_iter_total > 0:
        total_params["n_estimators"] = best_iter_total

    model_total = lgb.LGBMRegressor(**total_params)
    model_total.fit(X_trainval, y_total_trainval)

    # Test evaluation
    print("\n5) Test set evaluation...")
    ou_probs_test = model_ou.predict_proba(X_test)[:, 1]
    total_preds_test = model_total.predict(X_test)

    test_auc = roc_auc_score(y_ou_test, ou_probs_test)
    test_acc = ((ou_probs_test > 0.5).astype(int) == y_ou_test).mean()
    valid_total = ~np.isnan(y_total_test)
    test_mae = mean_absolute_error(y_total_test[valid_total], total_preds_test[valid_total]) if valid_total.any() else float("nan")
    test_r2 = r2_score(y_total_test[valid_total], total_preds_test[valid_total]) if valid_total.any() else float("nan")

    print(f"   O/U AUC:      {test_auc:.4f}")
    print(f"   O/U Accuracy: {test_acc:.4f}")
    print(f"   Total MAE:    {test_mae:.2f}")
    print(f"   Total R^2:    {test_r2:.4f}")

    # Save models
    print("\n6) Saving tuned models...")
    MODELS_ADV_DIR.mkdir(parents=True, exist_ok=True)

    model_ou.booster_.save_model(str(MODELS_ADV_DIR / "games_lgb_ou.txt"))
    model_total.booster_.save_model(str(MODELS_ADV_DIR / "games_lgb_total.txt"))
    joblib.dump(scaler, MODELS_ADV_DIR / "games_lgb_scaler.joblib")
    joblib.dump(encoder, MODELS_ADV_DIR / "games_lgb_encoder.joblib")
    joblib.dump({
        "input_dim": X.shape[1],
        "best_params": best_params,
        "test_auc": test_auc,
        "test_acc": test_acc,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "train_medians": getattr(scaler, "train_medians_", None),
        "feature_importance_ou": dict(
            zip(num_cols, model_ou.feature_importances_)
        ),
    }, MODELS_ADV_DIR / "games_lgb_meta.joblib")

    # Save study results
    results_path = MODELS_ADV_DIR / "optuna_study_results.json"
    study_data = {
        "study_name": study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": n_trials,
        "timestamp": datetime.now().isoformat(),
        "test_metrics": {
            "auc": test_auc,
            "accuracy": test_acc,
            "total_mae": test_mae,
            "total_r2": test_r2,
        },
        "top_10_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "total_mae": t.user_attrs.get("total_mae"),
                "total_r2": t.user_attrs.get("total_r2"),
            }
            for t in sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:10]
        ],
    }
    with open(results_path, "w") as f:
        json.dump(study_data, f, indent=2)
    print(f"   Study results saved to {results_path}")

    # Print top features
    imp = model_ou.feature_importances_
    top_idx = np.argsort(imp)[::-1][:15]
    print("\n   Top 15 features (O/U model):")
    for i, idx in enumerate(top_idx):
        if idx < len(num_cols):
            print(f"   {i+1:2d}. {num_cols[idx]:<30s} importance={imp[idx]:.0f}")

    print(f"\nDone: {datetime.now():%Y-%m-%d %H:%M:%S}")
    return study


def main():
    parser = argparse.ArgumentParser(description="LightGBM Optuna Tuning")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--study-name", type=str, default="lgb_ou_v3", help="Study name")
    args = parser.parse_args()

    run_tuning(n_trials=args.n_trials, study_name=args.study_name)


if __name__ == "__main__":
    main()

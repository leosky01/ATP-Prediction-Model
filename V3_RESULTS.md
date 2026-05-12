# ATP Advanced Prediction System – V3 Results

## Overview

V3 introduces a fully redesigned training pipeline with proper temporal
splits, symmetric data augmentation, LightGBM + Neural Net ensembles,
isotonic calibration, and 188 engineered features (up from 112). All
four prediction heads share the same feature space.

### Pipeline
- **Features (188 numerical + 12 cat one-hot = 200)**:
  - Context (rank, odds, age, height)
  - Rolling stats (10/20/50-match windows) — both diff/ratio AND raw per-player
  - Surface-specific stats per player
  - H2H (raw + surface)
  - Bio (height, hand, dominance, age)
  - Tournament context (avg games, draw size, prestige)
  - Fatigue (matches in 7/14/30 days)
- **Temporal split**: 70/15/15 chronological. Test period 2022-06 → 2026-01.
- **Symmetric augmentation**: P1/P2 swap → doubles training set, used for NN only.
- **Ensemble**: LightGBM (refit on train+val) + 3-seed NN average → logistic stacker → isotonic calibrator.

## Test-set Metrics

### Games – Over/Under (line 22.5 BO3, 38.5 BO5)

| Split | n_test | Test AUC | Accuracy | Brier |
|-------|-------:|---------:|---------:|------:|
| BO3   |  6,345 | **0.5757** | 55.5% | 0.2439 |
| BO5   |  1,403 | **0.6326** | 61.0% | 0.2307 |

Component breakdown (BO5):
| Model | Test AUC |
|-------|---------:|
| LightGBM | 0.6378 |
| NN (3-seed avg) | 0.6268 |
| Stacker + Calibrator | 0.6326 |

Baseline ("always majority class"):
- BO3: 53.4% accuracy → +2.1pp from model
- BO5: 58.7% accuracy → +2.3pp from model

### Games – Total Games (regression)

| Split | MAE | R² |
|-------|----:|---:|
| BO3 | 5.00 | 0.002 |
| BO5 | 7.46 | 0.068 |

Note: Total-games regression on the test horizon has weak signal in
public-data features (high inherent match variance).

### Duration (minutes)

| Metric | Value |
|--------|------:|
| MAE | 28.8 min |
| RMSE | 36.1 min |
| R² | **0.259** |
| Within 15 min | 29.2% |
| Within 30 min | 59.3% |

### Number of Sets

| Split | Classes | Test AUC (OvR) | Accuracy | Baseline |
|-------|---------|---------------:|---------:|---------:|
| BO3   | {2, 3}     | 0.5453 | 63.4% | 63.4% |
| BO5   | {3, 4, 5}  | 0.5837 | 43.8% | 44.4% |

Note: This task replaces the old `score_predictor` (which leaked
P1/P2-order info). BO3 set-count is essentially noise-limited (AUC ≈ 0.55).

### Player Stats (multi-target regression)

| Stat | MAE | R² |
|------|----:|---:|
| p1_ace / p2_ace | 2.94 / 2.85 | **0.41 / 0.43** |
| p1_df  / p2_df  | 1.62 / 1.62 | 0.25 / 0.23 |
| p1_bpFaced / p2_bpFaced | 3.02 / 3.03 | 0.21 / 0.21 |
| p1_bpSaved / p2_bpSaved | 2.34 / 2.38 | 0.11 / 0.10 |
| p1_1st_pct / p2_1st_pct | 0.054 / 0.054 | 0.17 / 0.17 |
| p1_1st_win_pct / p2_1st_win_pct | 0.065 / 0.065 | 0.26 / 0.26 |

## Comparison vs V2 baseline

| Task | V2 (best Optuna) | V3 |
|------|-----------------:|---:|
| BO3 OU AUC | 0.5806 | 0.5757 (calibrated) |
| BO5 OU AUC | n/a (mixed) | **0.6326** |
| Duration MAE | ~32 min | **28.8 min** |
| Stats R² | mixed | aces 0.41 |

## Usage

Train:
```bash
python -m src_advanced.train_v3 --model all     # all 4 heads
python -m src_advanced.train_v3 --model games   # just games
```

Predict:
```bash
python -m src_advanced.predict_v3 --sample 50000 --best-of 3
```

Programmatic:
```python
from src_advanced.predict_v3 import AdvancedPredictorV3
predictor = AdvancedPredictorV3()
out = predictor.predict(features_dict, best_of=3)
# out['games']['probability_over'], out['duration']['predicted_minutes'], ...
```

## Artifacts

All saved in `models_advanced/`:
- `games_v3_bo{3,5}_lgb_ou.txt`, `games_v3_bo{3,5}_lgb_total.txt`
- `games_v3_bo{3,5}_nn_s{0,1,2}.pt`
- `games_v3_bo{3,5}_stacker.joblib`, `games_v3_bo{3,5}_calibrator.joblib`
- `duration_v3_lgb.txt`, `duration_v3_nn.pt`
- `nsets_v3_bo{3,5}_lgb.txt`
- `stats_v3_{target}.txt`
- Shared: `*_scaler.joblib`, `*_encoder.joblib`, `*_meta.joblib`
- `features_cache.parquet` (188 features × 67,722 matches)
- `metrics_v3.json` (all reported numbers)

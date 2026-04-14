# ATP Tennis Match Prediction

Machine learning system for predicting ATP tennis match outcomes using an **EnhancedMLP** neural network trained on 24+ years of historical match data (2000–2024), integrated with a custom **Elo rating system** and **isotonic calibration**.

## Architecture Overview

```
                    ┌──────────────┐
                    │  Raw Match   │
                    │  Data (CSV)  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Feature    │
                    │ Engineering  │──► 14 numerical + one-hot categorical
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │                         │
     ┌────────▼────────┐      ┌────────▼────────┐
     │   EnhancedMLP   │      │   Elo Tracker   │
     │  (PyTorch MLP)  │      │ (overall+surf)  │
     └────────┬────────┘      └────────┬────────┘
              │                         │
              └────────────┬────────────┘
                           │
                    ┌──────▼───────┐
                    │  Blend +     │
                    │  Calibration │──► Final prediction + confidence
                    └──────────────┘
```

## Quick Start

```bash
git clone https://github.com/leosky01/<REPO_NAME>.git
cd <REPO_NAME>
pip install -r requirements.txt
```

### Train the model

```bash
python -m src.train
```

This runs the full pipeline: data loading → feature engineering → training → evaluation → artifact saving. Outputs model, scaler, encoder, and analysis plots to `models/`.

### Calibrate the model

```bash
python -m src.calibrate
```

Applies isotonic regression calibration (separate for favourites and underdogs) and evaluates baseline, conservative, and balanced prediction strategies.

### Predict a match

```bash
python -m src.predict predict \
    --player1 "Sinner J." --player2 "Alcaraz C." \
    --rank1 1 --rank2 3 \
    --odds1 1.90 --odds2 1.95 \
    --surface Clay --tournament "Roland Garros" \
    --date 2025-06-08
```

Supports three modes: `mlp` (neural network only), `elo` (Elo ratings only), `blend` (weighted combination).

### Manage Elo ratings

```bash
# Build Elo state from scratch
python -m src.elo_tracker build --end-date 2025-12-31

# Update incrementally with new matches
python -m src.elo_tracker update

# Query a player's Elo
python -m src.elo_tracker query --player "Sinner J." --surface Clay

# Add a single match
python -m src.elo_tracker add-match \
    --player1 "Sinner J." --player2 "Alcaraz C." \
    --winner "Sinner J." --surface Clay --date 2025-06-08
```

## Project Structure

```
src/
├── config.py          # Centralised configuration (paths, hyperparameters)
├── model.py           # EnhancedMLP network + TennisDataset
├── features.py        # Feature engineering pipeline
├── train.py           # Training & evaluation script
├── calibrate.py       # Isotonic calibration & strategy evaluation
├── predict.py         # Live prediction system (MLP + Elo blend)
├── elo_tracker.py     # Advanced Elo system with decay + per-surface
└── __init__.py
data/                  # Dataset (git-ignored)
models/                # Trained artifacts (git-ignored, regenerated)
```

## Dataset

| Property | Value |
|----------|-------|
| Period | 2000 – 2024 |
| Matches | ~67,600 |
| Surfaces | Hard (54%), Clay (32%), Grass (11%), Carpet (2%) |
| Tournaments | ATP250, ATP500, Masters 1000, Grand Slams |

**Columns**: `Tournament, Date, Series, Court, Surface, Round, Best of, Player_1, Player_2, Winner, Rank_1, Rank_2, Pts_1, Pts_2, Odd_1, Odd_2, Score`

## Model Details

### EnhancedMLP Architecture

`input → BatchNorm → 512 → LeakyReLU → Dropout(0.49) → BN → 128 → LeakyReLU → Dropout(0.42) → BN → 64 → LeakyReLU → Dropout(0.17) → 1`

- **Loss**: BCEWithLogitsLoss with class weighting
- **Optimiser**: AdamW (lr=9e-5, weight_decay=8.2e-5)
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.28)
- **Early stopping**: patience=30, max 100 epochs

### Features (14 numerical + 2 categorical)

| Category | Features |
|----------|----------|
| Ranking | `rank_diff`, `min_rank`, `max_rank` |
| Odds | `odds_ratio`, `min_odds`, `max_odds` |
| Form | `win_rate_diff`, `min/max_win_rate` (rolling 10-match window) |
| Streaks | `streak_diff`, `min/max_streak` |
| Head-to-Head | `h2h_win_rate`, `h2h_matches` |
| Categorical | `Surface`, `Tournament` (one-hot encoded) |

### Elo Rating System

- Overall + per-surface Elo ratings
- Dynamic K-factor: `K = 250 / (n_matches + 5)^0.4`
- Temporal decay: exponential decay after 90+ days of inactivity
- Comeback boost: 1.5x K-factor for returning players
- Incremental updates with cursor-based state management

### Calibration

- **Isotonic regression** fitted separately for favourites and underdogs
- Dynamic prediction threshold based on ranking gap and surface
- Upset signal detection (form differential, streaks, H2H, odds/ranking inconsistency)
- Three strategies: Baseline (>0.5), Conservative (>0.55), Balanced (dynamic threshold)

## Prediction Modes

| Mode | Description |
|------|-------------|
| `mlp` | Neural network probability with bias correction |
| `elo` | Pure Elo expected outcome |
| `blend` | Weighted: `alpha * mlp_prob + (1-alpha) * elo_prob` |

The blend mode also computes:
- **Confidence score**: multi-factor (decision certainty, calibration agreement, signal strength, H2H reliability)
- **Upset probability**: based on upset signals, ranking gap, and odds ratio
- **Momentum**: `0.7 * win_rate + 0.3 * streak_normalised` per player

## Important Notes

- The current training uses a **random train/test split**. Real-world accuracy with proper time-based validation is lower (~67-68%).
- Betting odds are highly efficient — ML predictions alone are **not sufficient for profitable betting**.
- Model artifacts in `models/` are git-ignored and must be regenerated via training.

## Requirements

- Python 3.9+
- PyTorch >= 2.0
- scikit-learn >= 1.3
- pandas, numpy, matplotlib, seaborn

See `requirements.txt` for the full list.

## License

This project is for educational and research purposes.

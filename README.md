# ATP Tennis Match Prediction

Machine learning system for predicting ATP tennis match outcomes, built around an **EnhancedMLP** neural network trained on 24+ years of historical data.

## Repository Structure

```
ATP-Tennis-Prediction/
├── data/
│   └── atp_tennis_cleaned.csv   # Main dataset (66,696 matches, 2000-2024)
├── notebooks/
│   └── Completo_1.ipynb         # Main notebook: EDA, training, evaluation
├── src/
│   ├── __init__.py
│   ├── elo_tracker.py           # Advanced Elo rating system
│   └── train.py                 # Standalone training & analysis script
├── models/                      # Trained model artifacts (git-ignored)
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick Start

```bash
git clone https://github.com/leosky01/ATP-Tennis-Prediction-.git
cd ATP-Tennis-Prediction-
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook notebooks/Completo_1.ipynb
```

### Run standalone training

```bash
python -m src.train
```

## Dataset

| Property | Value |
|----------|-------|
| Period | 2000 – 2024 |
| Matches | 66,696 |
| Surfaces | Hard (54%), Clay (32%), Grass (11%), Carpet (2%) |
| Tournaments | ATP250, ATP500, Masters 1000, Grand Slam |

**Columns**: Tournament, Date, Series, Court, Surface, Round, Best of, Player\_1, Player\_2, Winner, Rank\_1, Rank\_2, Pts\_1, Pts\_2, Odd\_1, Odd\_2, Score.

## Model

The core model is an **EnhancedMLP** (PyTorch) with architecture `input → 512 → 128 → 64 → 1`, using BatchNorm, LeakyReLU and Dropout. Training uses `BCEWithLogitsLoss` with class weighting, AdamW optimiser and ReduceLROnPlateau scheduler.

### Feature Engineering (14 numerical + 2 categorical)

- **Ranking**: rank\_diff, min\_rank, max\_rank
- **Odds**: odds\_ratio, min\_odds, max\_odds
- **Form**: win\_rate\_diff, min/max\_win\_rate (rolling 10-match window)
- **Streaks**: streak\_diff, min/max\_streak
- **Head-to-Head**: h2h\_win\_rate, h2h\_matches
- **Categorical**: Surface, Tournament (one-hot encoded)

### Calibration

Isotonic regression calibration with dynamic threshold adjustment and upset signal detection.

## Notes

- The notebook uses a **random train/test split**; real-world accuracy with proper time-based validation is lower (~67-68%).
- Betting odds are highly efficient — ML predictions alone are **not sufficient for profitable betting**.

## Requirements

- Python 3.8+
- PyTorch, scikit-learn, pandas, numpy, matplotlib, seaborn
- See `requirements.txt` for full list

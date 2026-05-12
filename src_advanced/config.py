"""
ATP Advanced Prediction System – Configuration

Paths, hyperparameters, feature column lists for the parallel prediction system.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
TML_DIR = DATA_DIR / "tml"
TML_RAW_DIR = TML_DIR / "raw"
MODELS_ADV_DIR = BASE_DIR / "models_advanced"
FIGURES_ADV_DIR = MODELS_ADV_DIR / "figures"

EXISTING_DATASET = DATA_DIR / "atp_tennis_cleaned.csv"
TML_DATABASE_CSV = TML_DIR / "atp_database.csv"
NAME_MAPPING_PATH = TML_DIR / "name_mapping.joblib"
MERGED_DATASET_PATH = TML_DIR / "merged_dataset.csv"

# ── TML Download ─────────────────────────────────────────────────────────────
TML_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "Tennismylife/TML-Database/master/{year}.csv"
)
TML_BIO_URL = (
    "https://raw.githubusercontent.com/"
    "Tennismylife/TML-Database/master/ATP_Database.csv"
)
TML_YEARS = list(range(2000, 2027))

# ── TML column names ─────────────────────────────────────────────────────────
TML_MATCH_COLS = [
    "tourney_name", "surface", "draw_size", "tourney_level", "indoor",
    "tourney_date", "match_num", "best_of", "round", "minutes", "score",
    "winner_name", "winner_seed", "winner_rank", "winner_rank_points",
    "winner_hand", "winner_ht", "winner_ioc", "winner_age",
    "loser_name", "loser_seed", "loser_rank", "loser_rank_points",
    "loser_hand", "loser_ht", "loser_ioc", "loser_age",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
]

TML_STAT_COLS = [
    "ace", "df", "svpt", "1stIn", "1stWon", "2ndWon",
    "SvGms", "bpSaved", "bpFaced",
]

# ── Feature columns (built by feature_engineering.py) ───────────────────────

CAT_COLS_ADVANCED = ["Surface", "tourney_level", "indoor"]

# Numerical features are generated programmatically in feature_engineering.py
# and exported as NUM_COLS_ADVANCED list.  The canonical list is built there.

# ── Rolling windows ─────────────────────────────────────────────────────────
ROLLING_WINDOWS = [10, 20, 50]

# ── Model hyperparameters ────────────────────────────────────────────────────

# GamesPredictor (over/under + regression)
GAMES_HIDDEN = [512, 256, 128, 64]
GAMES_DROPOUT = [0.4, 0.3, 0.2, 0.1]
GAMES_LR = 1e-4
GAMES_WEIGHT_DECAY = 1e-4
GAMES_PATIENCE = 25
GAMES_MAX_EPOCHS = 150
GAMES_THRESHOLD_BO3 = 22.5
GAMES_THRESHOLD_BO5 = 38.5
GAMES_LOSS_WEIGHTS = (0.7, 0.3)  # (BCE, MSE)

# ScorePredictor BO3 / BO5
SCORE_HIDDEN = [512, 256, 128]
SCORE_DROPOUT = [0.4, 0.3, 0.2]
SCORE_LR = 1e-4
SCORE_WEIGHT_DECAY = 1e-4
SCORE_PATIENCE = 25
SCORE_MAX_EPOCHS = 150

# DurationPredictor (Gaussian NLL)
DURATION_HIDDEN = [512, 256, 128]
DURATION_DROPOUT = [0.3, 0.25, 0.15]
DURATION_LR = 5e-4
DURATION_WEIGHT_DECAY = 1e-4
DURATION_PATIENCE = 25
DURATION_MAX_EPOCHS = 150

# StatsPredictor (multi-task)
STATS_ENCODER_HIDDEN = [512, 256, 128]
STATS_ENCODER_DROPOUT = [0.3, 0.25, 0.15]
STATS_HEAD_HIDDEN = [64]
STATS_HEAD_DROPOUT = [0.1]
STATS_LR = 5e-4
STATS_WEIGHT_DECAY = 1e-4
STATS_PATIENCE = 25
STATS_MAX_EPOCHS = 150

# ── Training defaults ────────────────────────────────────────────────────────
RANDOM_SEED = 42
BATCH_SIZE = 256
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15       # remainder -> test
SCHEDULER_MODE = "min"
SCHEDULER_PATIENCE = 7
SCHEDULER_FACTOR = 0.5
LEAKY_RELU_SLOPE = 0.1

# ── V2 Architecture & Training params ────────────────────────────────────────
USE_SE_BLOCKS = True
SE_REDUCTION = 4
WARMUP_EPOCHS = 5
GRADIENT_CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.1
SCORE_FOCAL_GAMMA = 2.0

# Games V2
GAMES_ENCODER_DIMS = [512, 512, 256, 128]
GAMES_ENCODER_BLOCKS = 1  # ResidualSE blocks per layer
GAMES_ENCODER_DROPOUT = 0.3
GAMES_HEAD_HIDDEN = 64
GAMES_HEAD_BLOCKS = 2
GAMES_HEAD_DROPOUT = 0.2

# Score V2
SCORE_ENCODER_DIMS = [512, 256, 128]
SCORE_ENCODER_BLOCKS = 1
SCORE_ENCODER_DROPOUT = 0.3
SCORE_HEAD_HIDDEN = 64
SCORE_HEAD_BLOCKS = 2
SCORE_HEAD_DROPOUT = 0.2

# Duration V2
DURATION_ENCODER_DIMS = [512, 256, 128]
DURATION_ENCODER_BLOCKS = 1
DURATION_ENCODER_DROPOUT = 0.25
DURATION_HEAD_HIDDEN = 64
DURATION_HEAD_BLOCKS = 3
DURATION_HEAD_DROPOUT = 0.2

# Stats V2
STATS_ENCODER_DIMS_V2 = [512, 256, 128]
STATS_ENCODER_BLOCKS = 1
STATS_ENCODER_DROPOUT_V2 = 0.25
STATS_HEAD_HIDDEN_V2 = 96
STATS_HEAD_BLOCKS = 2
STATS_HEAD_DROPOUT_V2 = 0.15

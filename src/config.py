"""
ATP Tennis Match Predictor – Configuration
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = MODELS_DIR  # figures saved alongside model artifacts

DATASET_PATH = DATA_DIR / "atp_tennis_cleaned.csv"
MODEL_PATH = MODELS_DIR / "best_model.pt"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
ENCODER_PATH = MODELS_DIR / "encoder.joblib"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.txt"
OPTIMIZATION_CONFIG_PATH = MODELS_DIR / "optimization_config.joblib"
ELO_STATE_PATH = MODELS_DIR / "elo_tracker" / "elo_state.joblib"

# ── Dataset ──────────────────────────────────────────────────────────────────
DATE_START = "2010-01-01"
DATE_END = "2025-12-31"

# ── Features ─────────────────────────────────────────────────────────────────
NUM_COLS = [
    "rank_diff", "odds_ratio", "win_rate_diff", "streak_diff",
    "min_rank", "max_rank", "min_odds", "max_odds",
    "max_win_rate", "min_win_rate", "max_streak", "min_streak",
    "h2h_win_rate", "h2h_matches",
]
CAT_COLS = ["Surface", "Tournament"]

ROLLING_WINDOW = 10  # matches for win-rate calculation

# ── Model architecture ───────────────────────────────────────────────────────
HIDDEN_SIZES = [512, 128, 64]
DROPOUT_RATES = [0.4929775, 0.4220651, 0.1701431]
LEAKY_RELU_SLOPE = 0.1

# ── Training ─────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15       # remainder → test
BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE = 30
LR = 9.00064e-05
WEIGHT_DECAY = 8.196026e-05
POS_WEIGHT_OFFSET = 5.05
SCHEDULER_MODE = "max"
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.28

# ── Elo defaults ─────────────────────────────────────────────────────────────
ELO_INITIAL = 1500
ELO_K_MULTIPLIER = 250
ELO_K_OFFSET = 5
ELO_K_SHAPE = 0.4
ELO_DECAY_LAMBDA = 0.01
ELO_DECAY_THRESHOLD_DAYS = 90
ELO_COMEBACK_K_BOOST = 1.5

# ── Video generation ─────────────────────────────────────────────────────────
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = OUTPUT_DIR / "temp"
VIDEOS_DIR = OUTPUT_DIR / "videos"
THUMBNAILS_DIR = OUTPUT_DIR / "thumbnails"

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 1
THUMB_WIDTH = 1280
THUMB_HEIGHT = 720

TTS_VOICE = "it-IT-ElsaNeural"
TTS_RATE = "-5%"

# Dark theme colours — vivid, high contrast for video
BG_COLOR = "#0D1117"
TEXT_COLOR = "#FFFFFF"
P1_COLOR = "#4FC3F7"       # bright cyan-blue
P2_COLOR = "#FF5252"       # vivid red
WINNER_COLOR = "#69F0AE"   # bright green
ACCENT_COLOR = "#CE93D8"   # bright purple accent
GRID_COLOR = "#1E2733"
HIGHLIGHT_BG = "#162032"   # subtle row highlight

# Section durations (seconds) – used for narration timing
SECTION_DURATIONS = {
    "intro": 8,
    "prediction_card": 10,
    "feature_circles": 15,
    "player_comparison": 12,
    "confidence_upset": 8,
    "model_breakdown": 8,
    "outro": 5,
}

# Feature labels (Italian) for circle chart
FEATURE_LABELS_IT = {
    "rank_diff": "Diff. Ranking",
    "odds_ratio": "Quota",
    "win_rate_diff": "Forma",
    "streak_diff": "Serie",
    "h2h_win_rate": "Scontri Diretti",
    "momentum_diff": "Momentum",
    "surface_perf_diff": "Rendimento Superficie",
}

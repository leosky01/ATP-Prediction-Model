"""
Feature cache utility.

Caches the output of `build_advanced_features` to a parquet file so that
training/evaluation runs avoid re-running the expensive (~5 min) row-by-row
feature engineering.

Cache invalidation is based on the modification time of the merged dataset
and the source code of `feature_engineering.py`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .config import MERGED_DATASET_PATH, MODELS_ADV_DIR


CACHE_PATH = MODELS_ADV_DIR / "features_cache.parquet"
CACHE_META_PATH = MODELS_ADV_DIR / "features_cache.meta"


def _compute_signature() -> str:
    """Hash of the merged dataset mtime + size + feature_engineering source."""
    h = hashlib.sha256()
    if MERGED_DATASET_PATH.exists():
        st = MERGED_DATASET_PATH.stat()
        h.update(f"{st.st_mtime}-{st.st_size}".encode())
    fe_path = Path(__file__).with_name("feature_engineering.py")
    if fe_path.exists():
        h.update(fe_path.read_bytes())
    return h.hexdigest()


def load_or_build_features(force: bool = False) -> pd.DataFrame:
    """Return the feature DataFrame, building it once and caching it.

    Parameters
    ----------
    force : bool
        If True, ignore the cache and rebuild from scratch.
    """
    from .feature_engineering import build_advanced_features

    sig = _compute_signature()
    if (not force) and CACHE_PATH.exists() and CACHE_META_PATH.exists():
        if CACHE_META_PATH.read_text().strip() == sig:
            print(f"[feature_cache] Hit ({CACHE_PATH.name})")
            df = pd.read_parquet(CACHE_PATH)
            return df
        else:
            print(f"[feature_cache] Signature mismatch, rebuilding")

    print(f"[feature_cache] Building features from scratch...")
    df_raw = pd.read_csv(MERGED_DATASET_PATH, parse_dates=["Date"], low_memory=False)
    df = build_advanced_features(df_raw)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Drop columns that pyarrow can't serialize (mixed-type cat cols are fine)
    try:
        df.to_parquet(CACHE_PATH, index=False)
        CACHE_META_PATH.write_text(sig)
        print(f"[feature_cache] Saved ({CACHE_PATH.name}, {len(df):,} rows)")
    except Exception as e:
        print(f"[feature_cache] WARNING: failed to save parquet: {e}")
    return df

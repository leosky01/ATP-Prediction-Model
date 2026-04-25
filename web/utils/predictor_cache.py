"""Singleton ATPPredictor cached across Streamlit reruns."""

import streamlit as st


def _load_predictor():
    """Load ATPPredictor using config defaults (includes bn1 remapping internally)."""
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.predict import ATPPredictor

    return ATPPredictor()


@st.cache_resource
def get_predictor():
    """Load and cache the ATPPredictor instance."""
    return _load_predictor()


@st.cache_resource
def get_history():
    """Return the ATPHistoryCalculator attached to the predictor."""
    return get_predictor().history

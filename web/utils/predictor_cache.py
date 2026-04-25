"""Singleton ATPPredictor cached across Streamlit reruns."""

import streamlit as st


def _load_predictor():
    """Load ATPPredictor handling key remapping for invariant model."""
    import sys
    from pathlib import Path
    import torch

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.predict import ATPPredictor
    from src.model import EnhancedMLP
    from src.config import NUM_COLS

    model_path = project_root / "models" / "best_model_invariant.pt"
    scaler_path = project_root / "models" / "scaler_invariant.joblib"
    encoder_path = project_root / "models" / "cat_encoder_invariant.joblib"
    elo_path = project_root / "models" / "elo_state.joblib"

    # Remap bn1 -> input_bn in state dict to match current architecture
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    if "bn1.weight" in state_dict and "input_bn.weight" not in state_dict:
        for old_key in list(state_dict.keys()):
            if old_key.startswith("bn1."):
                new_key = "input_bn." + old_key[4:]
                state_dict[new_key] = state_dict.pop(old_key)

    # Save remapped state dict to a temp file, load predictor from that
    import tempfile, os
    tmp_dir = tempfile.mkdtemp()
    tmp_model_path = os.path.join(tmp_dir, "model_remapped.pt")
    torch.save(state_dict, tmp_model_path)

    try:
        predictor = ATPPredictor(
            model_path=Path(tmp_model_path),
            scaler_path=scaler_path,
            encoder_path=encoder_path,
            elo_path=elo_path,
        )
    finally:
        os.remove(tmp_model_path)
        os.rmdir(tmp_dir)

    return predictor


@st.cache_resource
def get_predictor():
    """Load and cache the ATPPredictor instance."""
    return _load_predictor()


@st.cache_resource
def get_history():
    """Return the ATPHistoryCalculator attached to the predictor."""
    return get_predictor().history

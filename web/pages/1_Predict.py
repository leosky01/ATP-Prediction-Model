"""Predict page: input match details, see 8 interactive Plotly charts."""

import datetime
import json
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from web.db.database import init_db
init_db()

from web.auth.manager import can_predict, record_prediction_after, is_logged_in, get_current_user_id
from web.utils.predictor_cache import get_predictor, get_history
from web.utils.data_helpers import get_player_list, get_series_list, get_surface_list
from web.charts.prediction_card import create_prediction_card
from web.charts.feature_circles import create_feature_circles
from web.charts.butterfly_comparison import create_butterfly_comparison
from web.charts.confidence_gauge import create_confidence_gauge
from web.charts.model_breakdown import create_model_breakdown
from web.charts.elo_trend import create_elo_trend
from web.charts.last_10_matches import create_last_10_matches
from web.charts.surface_radar import create_surface_radar

# Import helpers from make_video (reuse, no duplication)
from src.make_video import _get_elo_trend, _get_last_10, _get_surface_stats_all, _compute_contributions

st.set_page_config(page_title="Predizione", page_icon="🎾", layout="wide")

# Load data
players = get_player_list()
categories = get_series_list()
surfaces = get_surface_list()

# ── Paywall check ────────────────────────────────────────────────────────
_ctx_headers = getattr(st, "context", None)
if _ctx_headers is not None:
    try:
        ip = st.query_params.get("ip", "") or st.context.headers.get("X-Forwarded-For", "unknown")
    except (AttributeError, TypeError):
        ip = st.query_params.get("ip", "") or "unknown"
else:
    ip = st.query_params.get("ip", "") or "unknown"
can_do, reason = can_predict(ip)

if not can_do:
    st.error(f"Accesso limitato: {reason}")
    st.markdown("### Sblocca le predizioni")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Registrati** per ottenere 3 predizioni gratuite")
        if st.button("Vai ad Account"):
            st.switch_page("pages/3_Account.py")
    with col2:
        st.markdown("**Abbonamento settimanale** a soli 5 EUR")
        if st.button("Abbonati ora"):
            st.switch_page("pages/3_Account.py")
    st.stop()

# ── Sidebar: Input form ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Match Input")

    player1 = st.selectbox("Giocatore 1", players, index=players.index("Sinner J.") if "Sinner J." in players else 0)
    player2 = st.selectbox("Giocatore 2", players, index=players.index("Alcaraz C.") if "Alcaraz C." in players else 1)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        rank1 = st.number_input("Ranking P1", min_value=1, max_value=2000, value=1)
    with col_r2:
        rank2 = st.number_input("Ranking P2", min_value=1, max_value=2000, value=3)

    col_o1, col_o2 = st.columns(2)
    with col_o1:
        odds1 = st.number_input("Quota P1", min_value=1.01, max_value=50.0, value=1.90, step=0.05)
    with col_o2:
        odds2 = st.number_input("Quota P2", min_value=1.01, max_value=50.0, value=1.95, step=0.05)

    surface = st.selectbox("Superficie", surfaces, index=0)
    category = st.selectbox("Categoria Torneo", categories, index=0)

    match_date = st.date_input("Data", value=datetime.date.today())

    st.divider()
    mode = st.radio("Modalita", ["blend", "mlp", "elo"], horizontal=True)
    alpha = st.slider("Alpha (MLP weight)", 0.0, 1.0, 0.65, 0.05, disabled=(mode != "blend"))

    st.divider()
    predict_btn = st.button("Predici", type="primary", use_container_width=True)

# ── Prediction ───────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Calcolo predizione in corso..."):
        predictor = get_predictor()
        result = predictor.predict(
            player1=player1, player2=player2,
            rank1=rank1, rank2=rank2,
            odds1=odds1, odds2=odds2,
            surface=surface, tournament=category,
            date_str=str(match_date),
            mode=mode, alpha=alpha,
        )

        # Record prediction
        uid = get_current_user_id() if is_logged_in() else None
        record_prediction_after(
            user_id=uid, ip_address=ip,
            player1=player1, player2=player2,
            rank1=rank1, rank2=rank2,
            odds1=odds1, odds2=odds2,
            surface=surface, tournament=category,
            result_json=json.dumps(result),
            confidence=result["confidence"],
        )

        st.session_state["last_result"] = result
        st.session_state["last_input"] = {
            "player1": player1, "player2": player2,
            "rank1": rank1, "rank2": rank2,
            "odds1": odds1, "odds2": odds2,
            "surface": surface, "tournament": category,
            "date": str(match_date),
            "mode": mode, "alpha": alpha,
        }

# ── Results display ──────────────────────────────────────────────────────
if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    inp = st.session_state["last_input"]

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Vincitore", res["prediction"])
    with col2:
        conf_color = "normal" if res["confidence"] > 0.5 else "inverse"
        st.metric("Confidenza", f"{res['confidence']:.0%}")
    with col3:
        st.metric("Upset", f"{res['upset_probability']:.0%}")
    with col4:
        st.metric(f"P({inp['player1']})", f"{res['prob_p1']:.1%}")

    st.markdown("---")

    # Tabs with 8 charts
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Predizione", "Feature", "Farfalla", "Confidenza",
        "Modelli", "Elo Trend", "Ultime 10", "Radar"
    ])

    with tab1:
        fig1 = create_prediction_card(
            inp["player1"], inp["player2"],
            inp["rank1"], inp["rank2"],
            res["prob_p1"], res["prob_p2"],
            res["confidence"],
            inp["tournament"], inp["surface"],
            res["prediction"],
        )
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        contributions = _compute_contributions(
            res, inp["rank1"], inp["rank2"], inp["odds1"], inp["odds2"],
        )
        fig2 = create_feature_circles(contributions)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st1 = res.get("player1_stats", {})
        st2 = res.get("player2_stats", {})
        surf1 = res.get("surface_p1", {})
        surf2 = res.get("surface_p2", {})
        h2h = res.get("h2h", {})
        comparison_stats = {
            "Win Rate": (st1.get("win_rate", 0.5), st2.get("win_rate", 0.5)),
            "Serie Vittorie": (
                min(max(st1.get("streak", 0), 0) / 10, 1.0),
                min(max(st2.get("streak", 0), 0) / 10, 1.0),
            ),
            "Momentum": (st1.get("momentum", 0.5), st2.get("momentum", 0.5)),
            "Superficie": (surf1.get("win_rate", 0.5), surf2.get("win_rate", 0.5)),
            "Scontri Diretti": (h2h.get("p1_win_rate", 0.5), 1 - h2h.get("p1_win_rate", 0.5)),
        }
        fig3 = create_butterfly_comparison(inp["player1"], inp["player2"], comparison_stats)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        fig4 = create_confidence_gauge(res["confidence"], res["upset_probability"])
        st.plotly_chart(fig4, use_container_width=True)

    with tab5:
        fig5 = create_model_breakdown(
            res["mlp_prob"], res["elo_prob"], res["blend_prob"],
            res["alpha"], inp["player1"],
        )
        st.plotly_chart(fig5, use_container_width=True)

    with tab6:
        history = get_history()
        try:
            p1_dates, p1_elos = _get_elo_trend(history, inp["player1"])
            p2_dates, p2_elos = _get_elo_trend(history, inp["player2"])
            fig6 = create_elo_trend(
                inp["player1"], inp["player2"],
                p1_dates, p1_elos, p2_dates, p2_elos,
                surface=inp["surface"],
            )
            st.plotly_chart(fig6, use_container_width=True)
        except Exception as e:
            st.warning(f"Dati Elo non disponibili: {e}")

    with tab7:
        history = get_history()
        try:
            p1_last10 = _get_last_10(history, inp["player1"])
            p2_last10 = _get_last_10(history, inp["player2"])
            fig7 = create_last_10_matches(inp["player1"], inp["player2"], p1_last10, p2_last10)
            st.plotly_chart(fig7, use_container_width=True)
        except Exception as e:
            st.warning(f"Dati partite non disponibili: {e}")

    with tab8:
        history = get_history()
        try:
            p1_surf = _get_surface_stats_all(history, inp["player1"])
            p2_surf = _get_surface_stats_all(history, inp["player2"])
            fig8 = create_surface_radar(inp["player1"], inp["player2"], p1_surf, p2_surf)
            st.plotly_chart(fig8, use_container_width=True)
        except Exception as e:
            st.warning(f"Dati superficie non disponibili: {e}")

    # Raw JSON expander
    with st.expander("Dettagli completi (JSON)"):
        st.json(res)
else:
    st.info("Inserisci i dati del match nella sidebar e clicca **Predici** per vedere i risultati.")

"""Account page: login, register, subscription management."""

import sys
from pathlib import Path

import streamlit as st

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from web.auth.manager import (
    is_logged_in, login_user, logout_user, register_user,
    get_current_user_id, has_active_subscription, get_credits_remaining,
    activate_weekly_subscription, add_credits,
)
from web.auth.paypal_client import (
    create_ppu_order, capture_order, create_weekly_subscription,
    get_paypal_auth,
)

st.set_page_config(page_title="Account", page_icon="👤", layout="centered")

# ── Handle PayPal callback ───────────────────────────────────────────────
params = st.query_params
if "token" in params and is_logged_in():
    order_id = params["token"]
    with st.spinner("Verificando pagamento..."):
        success, detail = capture_order(order_id)
        if success:
            st.success("Pagamento completato!")
            # Parse amount from detail
            units = detail.get("purchase_units", [{}])
            amount_str = units[0].get("amount", {}).get("value", "1.00") if units else "1.00"
            try:
                euros = float(amount_str)
            except ValueError:
                euros = 1.00
            credits = max(1, int(euros))  # 1 EUR = 1 prediction
            add_credits(get_current_user_id(), credits, paypal_order_id=order_id, euros=euros)
        else:
            st.error(f"Pagamento non completato: {detail}")
    # Clear params
    st.query_params.clear()
    st.rerun()

# ── Logged in ────────────────────────────────────────────────────────────
if is_logged_in():
    st.title("Il tuo Account")
    st.success(f"Connesso come **{st.session_state['username']}** ({st.session_state['email']})")

    # Subscription status
    uid = get_current_user_id()
    has_sub = has_active_subscription(uid)
    credits = get_credits_remaining(uid)

    st.subheader("Stato Abbonamento")
    col1, col2 = st.columns(2)
    with col1:
        if has_sub:
            st.metric("Abbonamento", "ATTIVO", delta="Illimitato")
        else:
            st.metric("Abbonamento", "Non attivo")
    with col2:
        st.metric("Credito PPV", f"{credits} predizioni")

    # Purchase options
    st.divider()
    st.subheader("Acquista Predizioni")

    # PayPal integration
    pp_auth = get_paypal_auth()

    col_ppu1, col_ppu2, col_ppu3 = st.columns(3)
    with col_ppu1:
        if st.button("1 predizione (1 EUR)", use_container_width=True):
            if pp_auth:
                url = create_ppu_order(1.00)
                if url:
                    st.markdown(f"[Clicca qui per pagare con PayPal]({url})")
                else:
                    st.error("Errore nella creazione dell'ordine PayPal.")
            else:
                add_credits(uid, 1, euros=1.00)
                st.success("1 predizione aggiunta (modalita test)")
                st.rerun()

    with col_ppu2:
        if st.button("5 predizioni (5 EUR)", use_container_width=True):
            if pp_auth:
                url = create_ppu_order(5.00)
                if url:
                    st.markdown(f"[Clicca qui per pagare con PayPal]({url})")
                else:
                    st.error("Errore nella creazione dell'ordine PayPal.")
            else:
                add_credits(uid, 5, euros=5.00)
                st.success("5 predizioni aggiunte (modalita test)")
                st.rerun()

    with col_ppu3:
        if st.button("10 predizioni (10 EUR)", use_container_width=True):
            if pp_auth:
                url = create_ppu_order(10.00)
                if url:
                    st.markdown(f"[Clicca qui per pagare con PayPal]({url})")
                else:
                    st.error("Errore nella creazione dell'ordine PayPal.")
            else:
                add_credits(uid, 10, euros=10.00)
                st.success("10 predizioni aggiunte (modalita test)")
                st.rerun()

    st.divider()
    st.subheader("Abbonamento Settimanale")
    st.markdown("**5 EUR / settimana** — Predizioni illimitate per 7 giorni")

    if st.button("Attiva Abbonamento Settimanale", type="primary", use_container_width=True):
        if pp_auth:
            url = create_weekly_subscription()
            if url:
                st.markdown(f"[Clicca qui per attivare con PayPal]({url})")
            else:
                st.error("Errore nella creazione dell'abbonamento PayPal.")
        else:
            activate_weekly_subscription(uid)
            st.success("Abbonamento attivato! (modalita test)")
            st.rerun()

    # Logout
    st.divider()
    if st.button("Logout"):
        logout_user()
        st.rerun()

# ── Not logged in ────────────────────────────────────────────────────────
else:
    st.title("Account")

    tab_login, tab_register = st.tabs(["Login", "Registrati"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                success, msg = login_user(email, password)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    with tab_register:
        with st.form("register_form"):
            reg_email = st.text_input("Email *")
            reg_username = st.text_input("Username *")
            reg_password = st.text_input("Password *", type="password")
            reg_confirm = st.text_input("Conferma Password *", type="password")
            reg_submitted = st.form_submit_button("Registrati")
            if reg_submitted:
                if not all([reg_email, reg_username, reg_password]):
                    st.error("Compila tutti i campi.")
                elif reg_password != reg_confirm:
                    st.error("Le password non coincidono.")
                elif len(reg_password) < 6:
                    st.error("La password deve avere almeno 6 caratteri.")
                else:
                    success, msg = register_user(reg_email, reg_username, reg_password)
                    if success:
                        st.success(msg + " Effettua il login.")
                    else:
                        st.error(msg)

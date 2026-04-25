"""PayPal REST API v2 client for creating orders and subscriptions.

Uses direct HTTP requests to PayPal API (no deprecated SDK).
Supports sandbox (development) and production modes.
"""

import base64
import os
import requests


def get_paypal_config() -> dict:
    """Load PayPal config from environment or Streamlit secrets."""
    try:
        import streamlit as st
        secrets = st.secrets
        return {
            "client_id": secrets.get("PAYPAL_CLIENT_ID", os.environ.get("PAYPAL_CLIENT_ID", "")),
            "client_secret": secrets.get("PAYPAL_CLIENT_SECRET", os.environ.get("PAYPAL_CLIENT_SECRET", "")),
            "base_url": secrets.get("PAYPAL_BASE_URL", os.environ.get("PAYPAL_BASE_URL", "https://api-m.sandbox.paypal.com")),
            "weekly_plan_id": secrets.get("WEEKLY_PLAN_ID", os.environ.get("WEEKLY_PLAN_ID", "")),
        }
    except Exception:
        return {
            "client_id": os.environ.get("PAYPAL_CLIENT_ID", ""),
            "client_secret": os.environ.get("PAYPAL_CLIENT_SECRET", ""),
            "base_url": os.environ.get("PAYPAL_BASE_URL", "https://api-m.sandbox.paypal.com"),
            "weekly_plan_id": os.environ.get("WEEKLY_PLAN_ID", ""),
        }


def get_paypal_auth() -> str | None:
    """Get PayPal OAuth access token. Returns None if credentials not configured."""
    config = get_paypal_config()
    client_id = config["client_id"]
    client_secret = config["client_secret"]
    base_url = config["base_url"]

    if not client_id or client_id == "your_client_id_here":
        return None

    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    try:
        resp = requests.post(
            f"{base_url}/v1/oauth2/token",
            headers={"Authorization": f"Basic {auth}"},
            data={"grant_type": "client_credentials"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]
    except Exception as e:
        print(f"PayPal auth error: {e}")
        return None


def create_ppu_order(amount_eur: float, return_url: str = "", cancel_url: str = "") -> str | None:
    """
    Create a PayPal order for pay-per-use predictions.
    Returns the approval URL for the user, or None on error.
    """
    token = get_paypal_auth()
    if not token:
        return None

    config = get_paypal_config()
    base_url = config["base_url"]

    if not return_url:
        import streamlit as st
        base = st.secrets.get("APP_BASE_URL", "http://localhost:8501")
        return_url = f"{base}/Account"
    if not cancel_url:
        cancel_url = return_url

    order_body = {
        "intent": "CAPTURE",
        "purchase_units": [{
            "amount": {
                "currency_code": "EUR",
                "value": f"{amount_eur:.2f}",
            },
            "description": f"ATP Predictor — {int(amount_eur)} predizione/i",
        }],
        "application_context": {
            "return_url": return_url,
            "cancel_url": cancel_url,
            "brand_name": "ATP Predictor",
            "user_action": "PAY_NOW",
        },
    }

    try:
        resp = requests.post(
            f"{base_url}/v2/checkout/orders",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=order_body,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract approval URL
        for link in data.get("links", []):
            if link["rel"] == "approve":
                return link["href"]

        return None
    except Exception as e:
        print(f"PayPal create order error: {e}")
        return None


def capture_order(order_id: str) -> tuple[bool, dict]:
    """
    Capture a previously approved PayPal order.
    Returns (success, order_details).
    """
    token = get_paypal_auth()
    if not token:
        return False, {"error": "PayPal auth failed"}

    config = get_paypal_config()
    base_url = config["base_url"]

    try:
        resp = requests.post(
            f"{base_url}/v2/checkout/orders/{order_id}/capture",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        data = resp.json()

        if resp.status_code == 201 and data.get("status") == "COMPLETED":
            return True, data
        else:
            return False, data
    except Exception as e:
        return False, {"error": str(e)}


def create_weekly_subscription(return_url: str = "", cancel_url: str = "") -> str | None:
    """
    Create a PayPal subscription for the weekly plan.
    Returns the approval URL, or None on error.
    """
    token = get_paypal_auth()
    if not token:
        return None

    config = get_paypal_config()
    base_url = config["base_url"]
    plan_id = config["weekly_plan_id"]

    if not plan_id or plan_id == "your_weekly_plan_id_here":
        return None

    if not return_url:
        import streamlit as st
        base = st.secrets.get("APP_BASE_URL", "http://localhost:8501")
        return_url = f"{base}/Account"
    if not cancel_url:
        cancel_url = return_url

    sub_body = {
        "plan_id": plan_id,
        "application_context": {
            "return_url": return_url,
            "cancel_url": cancel_url,
            "brand_name": "ATP Predictor",
        },
    }

    try:
        resp = requests.post(
            f"{base_url}/v1/billing/subscriptions",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=sub_body,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        for link in data.get("links", []):
            if link["rel"] == "approve":
                return link["href"]

        return None
    except Exception as e:
        print(f"PayPal subscription error: {e}")
        return None


def verify_webhook_signature(headers: dict, body: bytes) -> bool:
    """
    Verify PayPal webhook signature.
    In production, implement full signature verification.
    For development, returns True.
    """
    # TODO: Implement full webhook signature verification for production
    # https://developer.paypal.com/api/rest/webhooks/rest/#verify-webhook-signature
    return True

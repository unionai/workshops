"""Fraud detection dashboard — interactive UI for the scoring API.

Connects to the deployed scoring endpoint (or local app.py) and provides
an interactive UI for reviewing and testing transactions.

Usage:
    # Local (run `python app.py` in another terminal first)
    python dashboard.py

    # Against remote scoring API
    API_URL=https://<app-url> python dashboard.py

    # Deploy to Flyte
    flyte deploy dashboard.py dashboard_env
"""

import argparse
import os
import logging
import requests
import gradio as gr
import uvicorn
from fastapi import FastAPI

import flyte
from flyte.app import AppEndpoint, Parameter
from flyte.app.extras import FastAPIAppEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL_ENV = "API_URL"

CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net",
    "grocery_pos", "health_fitness", "home", "kids_pets",
    "misc_net", "misc_pos", "personal_care", "shopping_net",
    "shopping_pos", "travel",
]


def score_transaction(user_id, amt, category, merch_lat, merch_long, hour, day_of_week):
    """Call the scoring API and format the response."""
    api_url = os.environ.get(API_URL_ENV, "http://localhost:8080")
    logger.info(f"Calling {api_url}/score with user_id={user_id}, amt={amt}, category={category}")
    try:
        resp = requests.get(
            f"{api_url}/score",
            params={
                "user_id": int(user_id),
                "amt": float(amt),
                "category": category,
                "merch_lat": float(merch_lat),
                "merch_long": float(merch_long),
                "hour": int(hour),
                "day_of_week": int(day_of_week),
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Got response: {data.get('fraud_prediction')} ({data.get('fraud_probability')})")
    except requests.ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return f"**Connection Error** — Cannot reach `{api_url}`\n\nIs the scoring app running?", "", "", ""
    except requests.Timeout:
        logger.error("Request timed out (60s)")
        return f"**Timeout** — Scoring app at `{api_url}` did not respond within 60s", "", "", ""
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"**Error:** {e}", "", "", ""

    # Format prediction
    prob = data["fraud_probability"]
    risk = data["risk_level"]
    prediction = data["fraud_prediction"]

    if risk == "HIGH":
        verdict = f"## 🚨 {prediction} (HIGH RISK)\n\nFraud probability: **{prob:.1%}**"
    elif risk == "MEDIUM":
        verdict = f"## ⚠️ {prediction} (MEDIUM RISK)\n\nFraud probability: **{prob:.1%}**"
    else:
        verdict = f"## ✅ {prediction} (LOW RISK)\n\nFraud probability: **{prob:.1%}**"

    # Format signals
    signals = data.get("signals", [])
    if signals:
        signals_md = "\n".join(f"- {s}" for s in signals)
    else:
        signals_md = "No risk signals detected."

    # Format user profile
    profile = data.get("user_profile", {})
    profile_md = (
        f"- **Transactions:** {profile.get('txn_count', 'N/A')}\n"
        f"- **Avg amount:** ${profile.get('mean_amt', 0):.2f}\n"
        f"- **Std dev:** ${profile.get('std_amt', 0):.2f}\n"
        f"- **Age:** {profile.get('age', 'N/A')}"
    )

    # Format transaction + scoring details
    txn = data.get("transaction", {})
    scoring = data.get("scoring", {})
    model_prob = data.get("model_probability", prob)
    txn_md = (
        f"- **Amount:** ${txn.get('amt', amt):.2f}\n"
        f"- **Category:** {txn.get('category', category)}\n"
        f"- **Hour (UTC):** {txn.get('hour', 'N/A')}\n"
        f"- **Day of week:** {txn.get('day_of_week', 'N/A')}\n"
        f"- **Amount z-score:** {scoring.get('amt_zscore', 'N/A')}\n"
        f"- **Distance from home:** {scoring.get('distance_from_home', 'N/A')} mi\n"
        f"- **Model probability:** {model_prob:.1%}"
        + (f"\n- **Rule override applied**" if scoring.get('rule_applied') else "")
    )

    return verdict, signals_md, profile_md, txn_md


def build_gradio_app():
    """Build the Gradio Blocks app."""
    with gr.Blocks(title="Fraud Detection Dashboard", theme=gr.themes.Soft()) as blocks:
        gr.Markdown("# Fraud Detection Dashboard\nScore transactions against the ML model + Feast feature store.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Transaction Details")
                user_id = gr.Number(label="User ID", value=42, precision=0)
                amt = gr.Number(label="Amount ($)", value=25.00)
                category = gr.Dropdown(
                    label="Merchant Category",
                    choices=CATEGORIES,
                    value="grocery_pos",
                )
                merch_lat = gr.Number(label="Merchant Latitude", value=33.9)
                merch_long = gr.Number(label="Merchant Longitude", value=-80.3)
                hour = gr.Slider(label="Hour (UTC)", minimum=0, maximum=23, step=1, value=22)
                day_of_week = gr.Dropdown(
                    label="Day of Week",
                    choices=[(d, i) for i, d in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])],
                    value=3,
                )
                score_btn = gr.Button("Score Transaction", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Results")
                verdict = gr.Markdown(label="Prediction", value="*Click Score to analyze a transaction.*")
                signals = gr.Markdown(label="Risk Signals")

                with gr.Accordion("User Profile (from Feast)", open=False):
                    profile = gr.Markdown()

                with gr.Accordion("Transaction Info", open=False):
                    txn_info = gr.Markdown()

        score_btn.click(
            fn=score_transaction,
            inputs=[user_id, amt, category, merch_lat, merch_long, hour, day_of_week],
            outputs=[verdict, signals, profile, txn_info],
            show_progress="minimal",
        )

        gr.Markdown("---")
        gr.Markdown(
            "### Quick Tests\n"
            "Try these scenarios to see how the model responds:\n\n"
            "| Scenario | User ID | Amount | Category | Lat | Long | Hour | Day |\n"
            "|----------|---------|--------|----------|-----|------|------|-----|\n"
            "| Normal grocery | 42 | $25 | grocery_pos | 33.9 | -80.3 | 14 | Wed |\n"
            "| Late-night large purchase | 42 | $2,500 | shopping_net | 33.9 | -80.3 | 2 | Wed |\n"
            "| Far-away transaction | 42 | $100 | travel | 48.8 | 2.3 | 22 | Thu |\n"
            "| Suspicious: big + far + late | 42 | $9,999 | shopping_net | 48.8 | 2.3 | 23 | Thu |"
        )

    return blocks


# ------------------------------------------------------------------
# App factory — called by uvicorn on the worker (never pickled)
# ------------------------------------------------------------------

def create_app():
    """Build the full FastAPI + Gradio app. Called fresh on the worker."""
    fastapi_app = FastAPI(title="Fraud Detection Dashboard")

    @fastapi_app.get("/health")
    async def health():
        api_url = os.environ.get(API_URL_ENV, "http://localhost:8080")
        try:
            resp = requests.get(f"{api_url}/health", timeout=10)
            return {"status": "healthy", "api_url": api_url, "scorer_status": resp.json()}
        except Exception as e:
            return {"status": "unhealthy", "api_url": api_url, "error": str(e)}

    gradio_app = build_gradio_app()
    gr.mount_gradio_app(fastapi_app, gradio_app, path="/")

    return fastapi_app


# ------------------------------------------------------------------
# Flyte deployment — bare app for pickling, factory for serving
# ------------------------------------------------------------------

# Bare FastAPI app (no Gradio) — this is what Flyte pickles
_bare_app = FastAPI(title="Fraud Detection Dashboard")

dashboard_env = FastAPIAppEnvironment(
    name="fraud-dashboard",
    app=_bare_app,
    description="Interactive dashboard for fraud detection scoring",
    parameters=[
        Parameter(
            name="api_url",
            value=AppEndpoint(app_name="fraud-scorer"),
            env_var=API_URL_ENV,
        ),
    ],
    # Factory pattern: uvicorn creates the full app (with Gradio) on the worker
    uvicorn_config=uvicorn.Config("dashboard:create_app", factory=True, port=8080),
    image=flyte.Image.from_debian_base().with_pip_packages(
        "gradio", "requests", "fastapi", "uvicorn",
    ),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    requires_auth=False,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud detection dashboard")
    parser.add_argument("--api-url", help="Override scoring API URL (default: auto-discover fraud-scorer app)")
    args = parser.parse_args()

    if args.api_url:
        # Override AppEndpoint with explicit URL for deploy
        dashboard_env.parameters = [
            Parameter(name="api_url", value=args.api_url, env_var=API_URL_ENV),
        ]
        print(f"Using scoring API: {args.api_url}")

    # Local: launch Gradio directly
    api_url = os.environ.get(API_URL_ENV, "http://localhost:8080")
    print(f"Scoring API: {api_url}")
    print("Make sure the scoring app is running (python app.py)")
    gradio_app = build_gradio_app()
    gradio_app.launch()

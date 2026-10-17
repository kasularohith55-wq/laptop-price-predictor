"""Flask backend: serves the UI and a JSON prediction API."""
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "model.pkl"))

app = Flask(__name__)

_bundle = None


def get_bundle() -> dict:
    """Load the trained model once, on first use."""
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} not found. Run `python train_model.py`."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


CATEGORICAL = ["brand", "processor", "storage_type", "gpu", "os"]
NUMERIC = {
    "ram_gb": (2, 128),
    "storage_gb": (64, 8192),
    "screen_size": (10, 20),
    "weight_kg": (0.5, 5),
    "touchscreen": (0, 1),
}


def validate(payload: dict) -> tuple[dict | None, str | None]:
    """Return (clean_row, None) or (None, error_message)."""
    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object."

    options = get_bundle()["options"]
    row = {}
    for field in CATEGORICAL:
        value = payload.get(field)
        if value not in options[field]:
            return None, f"Invalid or missing '{field}'. Choose one of: {', '.join(options[field])}"
        row[field] = value

    for field, (low, high) in NUMERIC.items():
        try:
            value = float(payload.get(field))
        except (TypeError, ValueError):
            return None, f"'{field}' must be a number."
        if not low <= value <= high:
            return None, f"'{field}' must be between {low} and {high}."
        row[field] = value
    return row, None


@app.get("/")
def index():
    return render_template("index.html", options=get_bundle()["options"])


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/api/options")
def api_options():
    return jsonify(get_bundle()["options"])


@app.post("/predict")
def predict():
    row, error = validate(request.get_json(silent=True))
    if error:
        return jsonify(error=error), 400

    bundle = get_bundle()
    price = float(bundle["pipeline"].predict(pd.DataFrame([row]))[0])

    # Spread across individual trees gives a rough price range
    pipe = bundle["pipeline"]
    X = pipe.named_steps["prep"].transform(pd.DataFrame([row]))
    forest = pipe.named_steps["model"].regressor_
    tree_prices = pipe.named_steps["model"].inverse_func(
        [t.predict(X)[0] for t in forest.estimators_]
    )
    low, high = (float(v) for v in pd.Series(tree_prices).quantile([0.1, 0.9]))

    return jsonify(
        price=round(price, -2),
        low=round(low, -2),
        high=round(high, -2),
        currency="INR",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

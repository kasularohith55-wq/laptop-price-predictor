"""Build a synthetic laptop dataset (prices in INR), train a Random Forest
regressor on it, and save the model to model.pkl.

The data is synthetic: prices come from a pricing formula plus random noise.
To train on a real dataset instead, replace build_dataset() with something like
`pd.read_csv("your_laptops.csv")` using the same columns.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MODEL_PATH = Path(__file__).parent / "model.pkl"
SEED = 42
N_ROWS = 3000

CATEGORICAL = ["brand", "processor", "storage_type", "gpu", "os"]
NUMERIC = ["ram_gb", "storage_gb", "screen_size", "weight_kg", "touchscreen"]
TARGET = "price_inr"

# ---------------------------------------------------------------- dataset
BRAND_PREMIUM = {
    "Apple": 1.35, "MSI": 1.15, "Dell": 1.05, "HP": 1.00,
    "Lenovo": 0.98, "Asus": 1.00, "Acer": 0.92,
}
PROCESSOR_BASE = {
    "Intel Core i3": 18000, "Intel Core i5": 28000, "Intel Core i7": 42000,
    "Intel Core i9": 68000, "AMD Ryzen 3": 17000, "AMD Ryzen 5": 27000,
    "AMD Ryzen 7": 40000, "AMD Ryzen 9": 65000,
    "Apple M1": 45000, "Apple M2": 58000, "Apple M3": 72000,
}
GPU_ADD = {
    "Integrated": 0, "AMD Radeon": 8000, "NVIDIA GTX 1650": 14000,
    "NVIDIA RTX 3050": 22000, "NVIDIA RTX 4060": 38000, "NVIDIA RTX 4070": 60000,
}
RAM_ADD = {4: 0, 8: 3500, 16: 9000, 32: 20000, 64: 42000}
STORAGE_ADD = {256: 0, 512: 3500, 1024: 9000, 2048: 20000}
OS_ADJ = {"Windows 11": 0, "macOS": 0, "Linux": -5000, "DOS": -8000}


def build_row(rng: np.random.Generator) -> dict:
    brand = rng.choice(list(BRAND_PREMIUM), p=[0.08, 0.07, 0.2, 0.22, 0.2, 0.15, 0.08])

    if brand == "Apple":
        processor = rng.choice(["Apple M1", "Apple M2", "Apple M3"], p=[0.35, 0.4, 0.25])
        os_name, gpu = "macOS", "Integrated"
        screen = float(rng.choice([13.3, 14.2, 15.3, 16.2]))
        ram = int(rng.choice([8, 16, 32], p=[0.5, 0.35, 0.15]))
        storage_type = "SSD"
    else:
        pool = [p for p in PROCESSOR_BASE if not p.startswith("Apple")]
        processor = rng.choice(pool, p=[0.1, 0.18, 0.14, 0.04, 0.1, 0.2, 0.17, 0.07])
        os_name = rng.choice(["Windows 11", "Linux", "DOS"], p=[0.8, 0.05, 0.15])
        high_end = processor in ("Intel Core i7", "Intel Core i9", "AMD Ryzen 7", "AMD Ryzen 9")
        gpu_probs = [0.2, 0.05, 0.1, 0.25, 0.25, 0.15] if high_end else [0.6, 0.1, 0.15, 0.1, 0.04, 0.01]
        gpu = rng.choice(list(GPU_ADD), p=gpu_probs)
        screen = float(rng.choice([13.3, 14.0, 15.6, 17.3], p=[0.1, 0.25, 0.55, 0.1]))
        ram = int(rng.choice([4, 8, 16, 32, 64], p=[0.08, 0.4, 0.35, 0.14, 0.03]))
        storage_type = rng.choice(["SSD", "HDD"], p=[0.85, 0.15])

    storage = int(rng.choice([256, 512, 1024, 2048], p=[0.25, 0.4, 0.28, 0.07]))
    if storage_type == "HDD":
        storage = int(rng.choice([512, 1024, 2048]))
    touch = int(rng.random() < 0.15)
    weight = round(float(np.clip(
        0.8 + (screen - 13) * 0.22 + (0.4 if gpu != "Integrated" else 0) + rng.normal(0, 0.12),
        0.9, 3.4)), 2)

    base = PROCESSOR_BASE[processor] + GPU_ADD[gpu] + RAM_ADD[ram] + OS_ADJ[os_name]
    base += STORAGE_ADD[storage] * (1.0 if storage_type == "SSD" else 0.4)
    base += (screen - 14) * 1800 + touch * 6000
    price = max(15000, round(base * BRAND_PREMIUM[brand] * rng.normal(1.0, 0.05) / 500) * 500)

    return {
        "brand": brand, "processor": processor, "ram_gb": ram,
        "storage_type": storage_type, "storage_gb": storage, "gpu": gpu,
        "screen_size": screen, "weight_kg": weight, "os": os_name,
        "touchscreen": touch, TARGET: int(price),
    }


def build_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    return pd.DataFrame([build_row(rng) for _ in range(N_ROWS)])


# ------------------------------------------------------------------ model
def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    forest = RandomForestRegressor(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    # Predict log(price) so errors scale with price instead of dominating on expensive laptops
    model = TransformedTargetRegressor(regressor=forest, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([("prep", preprocess), ("model", model)])


def main() -> None:
    df = build_dataset()
    X, y = df[CATEGORICAL + NUMERIC], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    metrics = {
        "r2": round(float(r2_score(y_test, pred)), 4),
        "mae_inr": round(float(mean_absolute_error(y_test, pred)), 2),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    print("Test metrics:", metrics)

    # Dropdown options for the frontend, taken from the data the model saw
    options = {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL}
    options["ram_gb"] = sorted(int(v) for v in df["ram_gb"].unique())
    options["storage_gb"] = sorted(int(v) for v in df["storage_gb"].unique())
    options["screen_size"] = sorted(float(v) for v in df["screen_size"].unique())

    joblib.dump({"pipeline": pipe, "options": options, "metrics": metrics}, MODEL_PATH, compress=3)
    print(f"Saved model to {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

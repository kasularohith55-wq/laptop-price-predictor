import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app  # noqa: E402

VALID = {
    "brand": "Dell",
    "processor": "Intel Core i5",
    "ram_gb": 16,
    "storage_type": "SSD",
    "storage_gb": 512,
    "gpu": "Integrated",
    "screen_size": 15.6,
    "weight_kg": 1.8,
    "os": "Windows 11",
    "touchscreen": 0,
}


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.get_json() == {"status": "ok"}


def test_index_renders_form(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Laptop Price Predictor" in res.data
    assert b'name="brand"' in res.data


def test_options_endpoint(client):
    data = client.get("/api/options").get_json()
    assert "Dell" in data["brand"]
    assert 16 in data["ram_gb"]


def test_predict_returns_price_and_range(client):
    res = client.post("/predict", json=VALID)
    assert res.status_code == 200
    body = res.get_json()
    assert body["currency"] == "INR"
    assert body["price"] > 0
    assert body["low"] <= body["price"] <= body["high"]


def test_more_powerful_laptop_costs_more(client):
    low_spec = client.post("/predict", json={**VALID, "ram_gb": 8, "storage_gb": 256}).get_json()
    high_spec = client.post(
        "/predict",
        json={**VALID, "processor": "Intel Core i9", "ram_gb": 32, "gpu": "NVIDIA RTX 4070"},
    ).get_json()
    assert high_spec["price"] > low_spec["price"]


@pytest.mark.parametrize(
    "change",
    [
        {"brand": "NotABrand"},
        {"ram_gb": "lots"},
        {"screen_size": 99},
        {"weight_kg": -1},
    ],
)
def test_predict_rejects_bad_input(client, change):
    res = client.post("/predict", json={**VALID, **change})
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_predict_rejects_missing_field(client):
    payload = {k: v for k, v in VALID.items() if k != "gpu"}
    assert client.post("/predict", json=payload).status_code == 400


def test_predict_rejects_non_json(client):
    assert client.post("/predict", data="not json").status_code == 400

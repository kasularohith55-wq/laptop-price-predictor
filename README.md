# Laptop Price Predictor

A machine learning web application that predicts laptop market prices in Indian Rupees (₹ INR) based on hardware specifications. Built with **scikit-learn**, **Flask**, and modern responsive **HTML/CSS/JavaScript**.

> **Note on Dataset:** The default dataset is generated synthetically in `train_model.py` (3,000 samples) based on realistic market pricing logic and noise. It demonstrates an end-to-end ML pipeline with cross-feature dependencies and price distributions. To train on a real dataset, replace `build_dataset()` in `train_model.py` with `pd.read_csv("your_data.csv")` using identical column names.

---

## Features

- **Machine Learning Pipeline**: Random Forest Regressor (150 trees) trained on `log1p(price)` with one-hot encoded categorical variables and passthrough numeric features.
- **Price Range Estimation**: Calculates 10th to 90th percentile prediction intervals across individual decision trees.
- **Web Interface**: Clean, responsive UI with automatic dark mode support and interactive dropdowns populated dynamically.
- **RESTful API**: Clean JSON endpoints with strict input validation for batch or programmatic price predictions.
- **Dockerized**: Multi-stage/production-ready Dockerfile with Gunicorn server, non-root user, and automated container healthcheck.
- **Comprehensive Unit Tests**: Full test suite covering health endpoints, page rendering, options, prediction bounds, and input validation.

---

## Project Structure

```text
├── app.py              # Flask web application & JSON API routes
├── train_model.py      # Dataset generation, pipeline definition, training & serialization
├── model.pkl           # Serialized model pipeline, metrics, and dropdown options bundle
├── requirements.txt    # Production & testing Python dependencies
├── Dockerfile          # Production container setup with Gunicorn & healthcheck
├── .gitignore          # Git exclusion rules (venv, caches, secrets, OS artifacts)
├── README.md           # Project documentation and setup guide
├── templates/
│   └── index.html      # Frontend HTML template & client-side interaction script
├── static/
│   └── style.css       # Responsive styling with light/dark theme variables
└── tests/
    └── test_app.py     # Pytest test suite for Flask endpoints & model inference
```

---

## Getting Started

### Prerequisites

- **Python**: Version 3.10 or higher (Python 3.11+ recommended)
- **Git**
- *(Optional)* **Docker**

---

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd laptop-price-predictor
   ```

2. **Create and activate a virtual environment:**
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD):**
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate.bat
     ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **(Optional) Train or regenerate the model:**
   A pre-trained `model.pkl` is included. To retrain the Random Forest model and inspect metrics:
   ```bash
   python train_model.py
   ```
   *Expected output:*
   ```text
   Test metrics: {'r2': 0.9401, 'mae_inr': 5039.86, 'train_rows': 2400, 'test_rows': 600}
   Saved model to .../model.pkl (4.4 MB)
   ```

5. **Run the Flask application:**
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://localhost:5000`.

---

## Running with Docker

You can build and run the application in a standalone Docker container:

```bash
# Build Docker image
docker build -t laptop-price-predictor .

# Run container on port 5000
docker run -d -p 5000:5000 --name laptop-price-app laptop-price-predictor
```

Access the application at `http://localhost:5000`. The container includes a Docker healthcheck querying `/health`.

---

## API Documentation

### 1. Health Check
- **Endpoint:** `GET /health`
- **Response:**
  ```json
  { "status": "ok" }
  ```

### 2. Available Specification Options
- **Endpoint:** `GET /api/options`
- **Response:** Returns valid dropdown options for categorical and numerical features (brand, processor, ram_gb, storage_type, storage_gb, gpu, os, screen_size).

### 3. Predict Laptop Price
- **Endpoint:** `POST /predict`
- **Headers:** `Content-Type: application/json`
- **Request Body:**
  ```json
  {
    "brand": "Dell",
    "processor": "Intel Core i5",
    "ram_gb": 16,
    "storage_type": "SSD",
    "storage_gb": 512,
    "gpu": "Integrated",
    "screen_size": 15.6,
    "weight_kg": 1.8,
    "os": "Windows 11",
    "touchscreen": 0
  }
  ```
- **Success Response (`200 OK`):**
  ```json
  {
    "currency": "INR",
    "price": 52000.0,
    "low": 47000.0,
    "high": 57000.0
  }
  ```
- **Validation Error (`400 Bad Request`):**
  ```json
  {
    "error": "Invalid or missing 'brand'. Choose one of: Acer, Apple, Asus, Dell, HP, Lenovo, MSI"
  }
  ```

---

## Testing

Run the automated test suite using `pytest`:

```bash
pytest -v
```

All tests in `tests/test_app.py` validate route responses, form rendering, data validation rules, model prediction sanity, and error states.

---

## How the Model Works

1. **Preprocessing**:
   - Categorical features (`brand`, `processor`, `storage_type`, `gpu`, `os`) are transformed via `OneHotEncoder(handle_unknown="ignore")`.
   - Numeric features (`ram_gb`, `storage_gb`, `screen_size`, `weight_kg`, `touchscreen`) are passed through directly via `ColumnTransformer`.
2. **Model Architecture**:
   - `RandomForestRegressor` with 150 estimators and `min_samples_leaf=2`.
   - Wrapped in a `TransformedTargetRegressor` applying `log1p` on target prices during fitting and `expm1` on inference, ensuring proportional relative error minimization.
3. **Uncertainty Interval**:
   - Predictions across all individual decision trees are extracted, inverse-transformed, and sampled at the 10th and 90th percentiles to provide a realistic confidence range.

# Laptop Price Predictor

A full-stack machine learning web application that predicts laptop market prices in Indian Rupees (₹ INR) based on hardware specifications. Built with **scikit-learn**, **Flask**, and modern responsive **HTML/CSS/JavaScript**, designed as an end-to-end engineering portfolio project for B.Tech IT placements.

> [!IMPORTANT]
> **Dataset Disclosure:** The current training dataset is **synthetically generated** in `train_model.py` (3,000 samples) using hardware pricing formulas plus Gaussian noise. It demonstrates an end-to-end machine learning engineering workflow (data generation, feature engineering, transformed target regression, and prediction intervals) rather than guaranteed real-world market quotes. To train on real market data, replace `build_dataset()` in `train_model.py` with `pd.read_csv("your_data.csv")` using identical schema columns.

---

## Portfolio & Engineering Highlights

- **End-to-End Pipeline**: Encapsulates raw data preprocessing, `OneHotEncoder`, and regression into a single serialized Scikit-learn `Pipeline` bundle.
- **Log-Scale Target Transformation**: Employs `TransformedTargetRegressor` with `log1p`/`expm1` to optimize relative percentage error rather than penalizing expensive laptop outliers disproportionately.
- **Uncertainty Quantification**: Calculates dynamic 10th-to-90th percentile prediction ranges across individual trees in the Random Forest ensemble.
- **Clean Architecture**: Decoupled Flask backend providing both server-rendered UI and a strictly-validated RESTful JSON API.
- **Production Containerization**: Multi-stage `Dockerfile` with non-root security permissions, environment variable configuration, and active health checks.
- **Test-Driven Reliability**: Automated `pytest` suite ensuring 100% test pass rate across data validation, error handling, and inference endpoints.

---

## Technologies Used

| Layer | Technologies |
|---|---|
| **Machine Learning** | Python 3.11, Scikit-learn (`RandomForestRegressor`, `ColumnTransformer`, `OneHotEncoder`, `TransformedTargetRegressor`), NumPy, Pandas, Joblib |
| **Backend & API** | Flask 3.0, Gunicorn (WSGI) |
| **Frontend** | HTML5, Modern CSS3 (CSS Variables, Grid, Dark Mode via `prefers-color-scheme`), Vanilla JavaScript (ES6+ `fetch`, `async/await`) |
| **Testing & Quality** | Pytest |
| **DevOps & Containerization** | Docker, Git, GitHub |

---

## Project Structure

```text
├── app.py              # Flask web application & REST API routes
├── train_model.py      # Dataset generation, pipeline definition, training & evaluation
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

### Local Installation (Windows & Unix)

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd laptop-price-predictor
   ```

2. **Create and activate a virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt):**
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate.bat
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **(Optional) Train or inspect the model:**
   A pre-trained `model.pkl` is already included. To retrain the Random Forest model and inspect metrics:
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
- **Response (`200 OK`):**
  ```json
  { "status": "ok" }
  ```

### 2. Available Specification Options
- **Endpoint:** `GET /api/options`
- **Response (`200 OK`):** Returns valid options for categorical and numerical features (`brand`, `processor`, `ram_gb`, `storage_type`, `storage_gb`, `gpu`, `os`, `screen_size`).

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
    "price": 45600.0,
    "low": 43000.0,
    "high": 49000.0
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
python -m pytest tests -v
```

All 11 tests in `tests/test_app.py` validate route responses, form rendering, data validation rules, model prediction sanity, and error states.

---

## How the Model Works

1. **Preprocessing Pipeline**:
   - Categorical features (`brand`, `processor`, `storage_type`, `gpu`, `os`) are transformed via `OneHotEncoder(handle_unknown="ignore")`.
   - Numeric features (`ram_gb`, `storage_gb`, `screen_size`, `weight_kg`, `touchscreen`) pass through directly via `ColumnTransformer`.
2. **Model Architecture**:
   - `RandomForestRegressor` with 150 estimators and `min_samples_leaf=2`.
   - Wrapped in a `TransformedTargetRegressor` applying `log1p` on target prices during fitting and `expm1` on inference, ensuring proportional relative error minimization.
3. **Uncertainty Interval**:
   - Predictions across all individual decision trees are extracted, inverse-transformed, and sampled at the 10th and 90th percentiles to provide a realistic confidence range.

---

## Limitations & Future Scope

### Limitations
- **Synthetic Training Data**: The training dataset uses mathematical rules and distributions to model pricing patterns. It is designed to demonstrate ML pipeline engineering and API integration rather than live market prices.
- **Hardware Coverage**: Model inputs are restricted to standard laptop hardware configurations present in the training set.

### Future Improvements
- **Real-World Data Scraper**: Collect current e-commerce market pricing via automated scrapers to replace synthetic data with live market figures.
- **Expanded Specifications**: Incorporate additional hardware features such as display refresh rate (120Hz/144Hz/240Hz), color gamut (100% sRGB/DCI-P3), and battery capacity (Watt-hours).
- **Model Comparison**: Benchmark Random Forest against XGBoost and LightGBM with automated hyperparameter tuning.
- **CI/CD Automation**: Implement GitHub Actions workflows for continuous integration testing and automated Docker image publishing.

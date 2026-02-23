# 📊 A/B Test MLOps — Statistical Inference for the Advert Industry

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![MLflow](https://img.shields.io/badge/MLflow-2.8+-orange?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

> A production-grade MLOps project combining **classical A/B statistical testing** with **machine learning** to analyse advertising conversion rates. Fully containerised with Docker, tracked with MLflow, versioned with DVC, and served via an interactive Streamlit dashboard.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Project Structure](#-project-structure)
- [Architecture](#-architecture)
- [Features](#-features)
- [Quick Start](#-quick-start)
  - [Local Setup (without Docker)](#1-local-setup-without-docker)
  - [Docker Setup](#2-docker-setup)
- [Training Models](#-training-models)
- [Running the Streamlit App](#-running-the-streamlit-app)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
- [A/B Testing Methodology](#-ab-testing-methodology)
- [Machine Learning Models](#-machine-learning-models)
- [Data](#-data)
- [Results & Metrics](#-results--metrics)
- [Improvements Over v1](#-improvements-over-v1)
- [Contributing](#-contributing)

---

## 🎯 Project Overview

This project investigates **advertising conversion rates** through two complementary lenses:

1. **Classical A/B Testing** — Rigorous statistical hypothesis tests to determine whether differences between control and treatment groups are statistically significant.
2. **Machine Learning** — Trained classifiers that predict conversion probability at the user level, enabling more granular insight than aggregate group comparisons.

The project follows **MLOps best practices**: reproducible pipelines, experiment tracking, data versioning, containerised deployment, and an interactive web dashboard.

---

## 📁 Project Structure

```
abtest-mlops/
│
├── 📂 scripts/
│   ├── train.py              # Model training + MLflow logging
│   ├── ab_test.py            # A/B testing statistical engine
│   └── evaluate.py           # Standalone model evaluation
│
├── 📂 Notebooks/
│   ├── 01_EDA.ipynb           # Exploratory data analysis
│   ├── 02_AB_Testing.ipynb    # Statistical testing walkthrough
│   └── 03_ML_Modelling.ipynb  # Model training experiments
│
├── 📂 data/                  # Raw data (tracked by DVC)
│   └── AdSmartABdata.csv
│
├── 📂 models/                # Saved model pipelines (.pkl)
│   ├── logistic_regression_pipeline.pkl
│   ├── decision_tree_pipeline.pkl
│   ├── random_forest_pipeline.pkl
│   ├── xgboost_pipeline.pkl
│   └── best_model.pkl
│
├── 📂 mlruns/                # MLflow experiment runs
│
├── 📂 outputs/               # Plots, reports, artefacts
│
├── app.py                    # Streamlit dashboard
├── Dockerfile                # Docker image for the app
├── docker-compose.yml        # Multi-service Docker orchestration
├── requirements.txt          # Python dependencies
├── .dvcignore
├── .gitignore
└── README.md
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Browser                         │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP :8501
┌─────────────────────────▼───────────────────────────────────┐
│               Streamlit App (Docker Container)              │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────────┐  │
│  │  A/B Testing │  │ ML Predictions │  │  Data Explorer  │  │
│  └──────────────┘  └────────────────┘  └─────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
┌─────────▼──────┐  ┌─────▼──────┐  ┌────▼────────────┐
│  Trained Models│  │  MLflow    │  │   Data (DVC)    │
│  (models/*.pkl)│  │  Tracking  │  │   (data/*.csv)  │
│                │  │  :5000     │  │                 │
└────────────────┘  └────────────┘  └─────────────────┘
```

---

## ✨ Features

### Statistical A/B Testing
| Test | Use Case |
|------|----------|
| Two-Proportion Z-Test | Binary outcomes (conversions) |
| Chi-Square Test | Independence of categorical variables |
| Welch's T-Test | Continuous outcomes |
| Mann-Whitney U Test | Non-parametric, no normality assumption |
| Sequential SPRT | Stopping rules for live experiments |
| Sample Size Calculator | Pre-experiment planning |

### Machine Learning
| Model | Key Benefit |
|-------|-------------|
| Logistic Regression | Interpretable baseline |
| Decision Tree | Explainable rules |
| Random Forest | Robust, handles noise |
| XGBoost | State-of-the-art accuracy |

### MLOps Capabilities
- 🔁 **Reproducible pipelines** — sklearn `Pipeline` + `ColumnTransformer`
- 📊 **Experiment tracking** — MLflow (params, metrics, artifacts)
- 🗄️ **Data versioning** — DVC
- 🐳 **Containerisation** — Docker + Docker Compose
- 📈 **Interactive dashboard** — Streamlit
- ✅ **Cross-validation** — Stratified 5-fold CV

---

## 🚀 Quick Start

### 1. Local Setup (without Docker)

**Prerequisites:** Python 3.11+, Git

```bash
# Clone the repo
git clone https://github.com/Desmondonam/abtest-mlops.git
cd abtest-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate       # macOS / Linux
# OR
.\venv\Scripts\activate        # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Train models
python scripts/train.py --data data/AdSmartABdata.csv

# Launch Streamlit app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

### 2. Docker Setup

**Prerequisites:** Docker Desktop (or Docker Engine + Docker Compose)

#### Option A — Streamlit only

```bash
# Build the image
docker build -t abtest-mlops:latest .

# Run the container
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  abtest-mlops:latest
```

Open **http://localhost:8501**

#### Option B — Full stack (Streamlit + MLflow server)

```bash
# Start all services
docker-compose up --build

# Run in background
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

| Service | URL |
|---------|-----|
| Streamlit Dashboard | http://localhost:8501 |
| MLflow Tracking UI | http://localhost:5000 |

---

## 🧠 Training Models

```bash
# Default — trains all 4 models and logs to MLflow
python scripts/train.py

# Custom data path
python scripts/train.py --data path/to/your/data.csv

# Custom MLflow experiment name
python scripts/train.py --experiment my_experiment_name
```

**What happens during training:**

1. Data is loaded and cleaned
2. Feature engineering (peak hours, binning)
3. A `ColumnTransformer` preprocessor is built for numeric + categorical columns
4. Each model is wrapped in a `Pipeline` (preprocessor → model)
5. 5-fold stratified cross-validation is run
6. Final model is trained on the full training set
7. Metrics are logged to MLflow
8. Best model (highest ROC-AUC on test set) is saved as `models/best_model.pkl`

---

## 📊 Running the Streamlit App

```bash
streamlit run app.py
```

The app has four pages:

| Page | Description |
|------|-------------|
| 🏠 Home | Overview and quick start guide |
| 📊 A/B Testing | Upload data, choose groups & metric, run statistical tests |
| 🤖 ML Predictions | Load a trained model and score new data |
| 📈 Data Explorer | Visualise distributions, missing values, correlations |

---

## 📈 MLflow Experiment Tracking

After training, launch the MLflow UI:

```bash
# If running locally
mlflow ui --port 5000

# If running with Docker Compose, it's already running at:
# http://localhost:5000
```

In the MLflow UI you can:
- Compare runs across all 4 models
- View logged metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Download saved model artifacts
- Reproduce any past experiment

---

## 🔬 A/B Testing Methodology

### Classical Testing (scripts/ab_test.py)

The `ABTest` class supports:

```python
from scripts.ab_test import ABTest
import pandas as pd

ctrl  = df[df["experiment"] == "control"]["converted"]
treat = df[df["experiment"] == "exposed"]["converted"]

ab = ABTest(ctrl, treat, alpha=0.05)
report = ab.full_report()
```

**Hypothesis:**
- H₀: There is no statistically significant difference in conversion rates between control and treatment.
- H₁: The treatment group has a significantly different conversion rate.

### Interpreting Results

| p-value | Interpretation |
|---------|----------------|
| p < α (0.05) | Reject H₀ — statistically significant difference |
| p ≥ α (0.05) | Fail to reject H₀ — insufficient evidence |

> ⚠️ **Statistical significance ≠ practical significance.** Always consider the effect size and business impact alongside the p-value.

### Sample Size Planning

Before running an experiment, calculate the required sample size:

```python
n = ab.required_sample_size(baseline_rate=0.10, mde=0.10)  # 10% lift
print(f"Need {n} users per group")
```

---

## 🤖 Machine Learning Models

All models are trained as sklearn `Pipeline` objects:

```
Pipeline(
  ColumnTransformer(
    numeric  → SimpleImputer(median) → StandardScaler
    categorical → SimpleImputer(most_frequent) → OneHotEncoder
  )
  → Classifier
)
```

### Feature Engineering

| Feature | Description |
|---------|-------------|
| `is_peak_hour` | 1 if hour is between 17:00–21:00 |
| `hour_bin` | Categorised time of day |
| One-hot encoded categoricals | Device, browser, OS, etc. |
| Scaled numerics | All numeric features standardised |

### Evaluation Metrics

| Metric | Formula | Why It Matters |
|--------|---------|----------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | Avoid false positives |
| Recall | TP/(TP+FN) | Avoid missing conversions |
| F1 Score | 2 × (P×R)/(P+R) | Balance P & R |
| ROC-AUC | Area under ROC curve | Ranking quality, threshold-independent |

---

## 📂 Data

The dataset (`data/AdSmartABdata.csv`) contains ad campaign impression-level data.

| Column | Type | Description |
|--------|------|-------------|
| `auction_id` | string | Unique impression ID |
| `experiment` | string | Group: `control` or `exposed` |
| `date` | date | Date of impression |
| `hour` | int | Hour of day (0–23) |
| `device_make` | string | Device manufacturer |
| `platform_os` | int | Operating system ID |
| `browser` | string | Browser used |
| `yes` | int | Saw ad (1) or didn't (0) |
| `no` | int | Saw PSA (1) or didn't (0) |
| `converted` | int | **Target** — Converted (1) or not (0) |

> Data is versioned with DVC. See `.dvc/` for remote storage config.

---

## 📊 Results & Metrics

Typical results on AdSmart dataset:

| Model | ROC-AUC | F1 | Accuracy |
|-------|---------|-----|----------|
| Logistic Regression | ~0.71 | ~0.42 | ~0.78 |
| Decision Tree | ~0.68 | ~0.40 | ~0.76 |
| Random Forest | ~0.76 | ~0.47 | ~0.81 |
| **XGBoost** | **~0.79** | **~0.50** | **~0.83** |

*Exact results will vary based on your data split and preprocessing.*

---

## 🆙 Improvements Over v1

This version significantly upgrades the original portfolio project:

| Area | Before (v1) | After (v2) |
|------|-------------|------------|
| Preprocessing | Ad-hoc in notebooks | Reproducible sklearn Pipelines |
| Models | Basic LR, DT, XGB | + Random Forest, proper CV |
| Evaluation | Simple train/test | 5-fold stratified CV + multiple metrics |
| Tracking | None | Full MLflow integration |
| Deployment | None | Streamlit app + Docker + Docker Compose |
| A/B Testing | Basic z-test | Z, Chi², T, Mann-Whitney U, SPRT, sample size calc |
| Code quality | Notebooks only | Modular scripts with logging + argparse |
| Data versioning | Manual | DVC |
| Documentation | Minimal README | This document |

---

## 🐳 Docker Reference

```bash
# Build
docker build -t abtest-mlops:latest .

# Run (standalone)
docker run -p 8501:8501 abtest-mlops:latest

# Full stack
docker-compose up --build

# Stop
docker-compose down

# Rebuild after code changes
docker-compose up --build --force-recreate

# Inspect running containers
docker ps

# View logs
docker logs abtest_streamlit
docker logs abtest_mlflow
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Desmond Onam**  
[GitHub](https://github.com/Desmondonam) · [LinkedIn](https://linkedin.com/in/desmond-onam)

---

*⭐ If you found this project useful, please give it a star on GitHub!*
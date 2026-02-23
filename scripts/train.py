"""
train.py - Improved model training with MLflow tracking and hyperparameter tuning.
Trains Logistic Regression, Decision Tree, Random Forest, and XGBoost models.
"""

import sys
print(sys.executable)
import os
import logging
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH = Path("data/AdSmartABdata.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TARGET_COL = "converted"
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
def load_and_preprocess(data_path: Path) -> tuple:
    """Load data, engineer features, split into train/test."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    logger.info(f"Raw shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    # ── Drop duplicates & obvious leakage columns ──
    df.drop_duplicates(inplace=True)

    # Drop columns that are identifiers or have too many unique values
    drop_cols = [c for c in df.columns if c.lower() in ("auction_id", "date", "time")]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # ── Encode yes/no booleans ──
    bool_cols = [c for c in df.columns if df[c].dtype == object
                 and df[c].dropna().isin(["yes", "no", True, False, "True", "False"]).all()]
    for col in bool_cols:
        df[col] = df[col].map({"yes": 1, "no": 0, True: 1, False: 0, "True": 1, "False": 0})

    # ── Feature engineering ──
    if "hour" in df.columns:
        df["is_peak_hour"] = df["hour"].apply(lambda h: 1 if 17 <= h <= 21 else 0)
        df["hour_bin"] = pd.cut(df["hour"], bins=[0, 6, 12, 17, 21, 24],
                                labels=["night", "morning", "afternoon", "evening", "late_evening"])

    # ── Separate features and target ──
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {df.columns.tolist()}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Remove rows where target is NaN
    mask = y.notna()
    X, y = X[mask], y[mask]
    y = y.astype(int)

    logger.info(f"Class distribution:\n{y.value_counts(normalize=True)}")

    # ── Identify column types ──
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    logger.info(f"Numerical cols: {num_cols}")
    logger.info(f"Categorical cols: {cat_cols}")

    # ── Build preprocessor ──
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, preprocessor


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    return metrics


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────
def get_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6, min_samples_split=20, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=10,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE
        ),
    }


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_all_models(
    X_train, X_test, y_train, y_test, preprocessor,
    experiment_name="ab_test_mlops"
):
    mlflow.set_experiment(experiment_name)
    best_model_name = None
    best_roc_auc = -1.0
    best_pipeline = None

    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for model_name, model in models.items():
        logger.info(f"\n{'='*50}\nTraining: {model_name}\n{'='*50}")

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        with mlflow.start_run(run_name=model_name):
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train,
                                        cv=cv, scoring="roc_auc", n_jobs=-1)
            logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            # Train on full train set
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            metrics = compute_metrics(y_test, y_pred, y_prob)
            metrics["cv_roc_auc_mean"] = cv_scores.mean()
            metrics["cv_roc_auc_std"] = cv_scores.std()

            # Log to MLflow
            mlflow.log_params({
                "model": model_name,
                "test_size": 0.2,
                "random_state": RANDOM_STATE,
            })
            mlflow.log_metrics(metrics)

            # Log model
            if model_name == "xgboost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(pipeline, artifact_path="model")

            logger.info(f"Metrics: {metrics}")
            logger.info(f"\n{classification_report(y_test, y_pred)}")

            # Save pipeline to disk
            model_path = MODEL_DIR / f"{model_name}_pipeline.pkl"
            joblib.dump(pipeline, model_path)
            mlflow.log_artifact(str(model_path))
            logger.info(f"Model saved to {model_path}")

            # Track best
            if y_prob is not None and metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = metrics["roc_auc"]
                best_model_name = model_name
                best_pipeline = pipeline

    # Save the best model separately
    if best_pipeline:
        best_path = MODEL_DIR / "best_model.pkl"
        joblib.dump(best_pipeline, best_path)
        logger.info(f"\nBest model: {best_model_name} (ROC-AUC={best_roc_auc:.4f})")
        logger.info(f"Best model saved to {best_path}")

    return best_model_name, best_roc_auc


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train A/B test models")
    parser.add_argument("--data", type=str, default=str(DATA_PATH), help="Path to CSV data")
    parser.add_argument("--experiment", type=str, default="ab_test_mlops", help="MLflow experiment name")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess(Path(args.data))
    best_name, best_score = train_all_models(
        X_train, X_test, y_train, y_test, preprocessor,
        experiment_name=args.experiment
    )
    logger.info(f"\nTraining complete! Best: {best_name} | ROC-AUC: {best_score:.4f}")
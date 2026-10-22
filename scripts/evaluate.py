"""
evaluate.py - Standalone evaluation script for trained model pipelines.
Usage: python scripts/evaluate.py --model models/best_model.pkl --data data/AdSmartABdata.csv
"""

import argparse
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)

from train import load_and_preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def evaluate(model_path: str, data_path: str):
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    # Same cleaning, feature engineering and split as training, so this is the held-out test set
    _, X_test, _, y_test, _ = load_and_preprocess(Path(data_path))

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Classification Report ──
    logger.info("\n" + classification_report(y_test, y_pred))

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["No Convert", "Convert"],
                yticklabels=["No Convert", "Convert"])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")

    # ── ROC Curve ──
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[1].plot(fpr, tpr, color="#6366f1", lw=2, label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()

    # ── Precision-Recall Curve ──
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    axes[2].plot(rec, prec, color="#10b981", lw=2)
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].set_title("Precision-Recall Curve")

    plt.tight_layout()
    save_path = OUTPUT_DIR / "evaluation_plots.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Plots saved to {save_path}")
    plt.close()

    summary = {
        "roc_auc": round(auc, 4),
        "accuracy": round((y_pred == y_test).mean(), 4),
        "test_samples": len(y_test),
        "confusion_matrix": cm.tolist(),
    }
    logger.info(f"Summary: {summary}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best_model.pkl")
    parser.add_argument("--data", type=str, default="data/AdSmartABdata.csv")
    args = parser.parse_args()
    evaluate(args.model, args.data)
"""Compute metrics and generate evaluation outputs."""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
)

sys.path.insert(0, os.path.dirname(__file__))
from config import CLASSIFIERS, RESULTS_DIR, FIGURES_DIR


# Helpers

def _load_models(y_test: np.ndarray):
    """Load saved models and generate predictions."""
    records = {}
    for name in CLASSIFIERS:
        path = os.path.join(RESULTS_DIR, f"{name.replace(' ', '_')}.pkl")
        if not os.path.exists(path):
            print(f"  [WARN] Model not found for {name}: {path}")
            continue
        with open(path, "rb") as f:
            clf = pickle.load(f)

        X_test = np.load(os.path.join(RESULTS_DIR, "X_test.npy"), allow_pickle=True)
        y_pred  = clf.predict(X_test)
        y_proba = None
        if hasattr(clf, "predict_proba"):
            try:
                y_proba = clf.predict_proba(X_test)[:, 1]
            except Exception:
                pass

        records[name] = {"clf": clf, "y_pred": y_pred, "y_proba": y_proba}
    return records


def compute_metrics(y_test: np.ndarray, records: dict) -> pd.DataFrame:
    rows = []
    for name, data in records.items():
        y_pred  = data["y_pred"]
        y_proba = data["y_proba"]
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
        rows.append({
            "Classifier":  name,
            "Accuracy":    accuracy_score(y_test, y_pred),
            "Precision":   precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall":      recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1-Score":    f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "ROC-AUC":     auc,
        })
    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return df


# Plots

def plot_metric_comparison(df: pd.DataFrame):
    """Plot grouped bars for core metrics."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    df_m = df.melt(id_vars="Classifier", value_vars=metrics,
                   var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df_m, x="Classifier", y="Score", hue="Metric", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_title("Classifier Performance Comparison – TruthSeeker Dataset")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "metric_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def plot_accuracy_bar(df: pd.DataFrame):
    """Plot accuracy ranking."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df["Classifier"][::-1], df["Accuracy"][::-1], color="steelblue")
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accuracy")
    ax.set_title("Accuracy Ranking – TruthSeeker Fake News Detection")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "accuracy_ranking.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def plot_confusion_matrices(y_test: np.ndarray, records: dict):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    n = len(records)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = np.array(axes).flatten()

    for idx, (name, data) in enumerate(records.items()):
        cm = confusion_matrix(y_test, data["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"],
            ax=axes[idx],
        )
        axes[idx].set_title(name, fontsize=9)
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Confusion Matrices – TruthSeeker Dataset", fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def plot_roc_curves(y_test: np.ndarray, records: dict):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1)

    for name, data in records.items():
        if data["y_proba"] is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, data["y_proba"])
        auc = roc_auc_score(y_test, data["y_proba"])
        ax.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – TruthSeeker Fake News Detection")
    ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "roc_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def print_classification_reports(y_test: np.ndarray, records: dict):
    sep = "-" * 55
    for name, data in records.items():
        print(f"\n{sep}")
        print(f"  {name}")
        print(sep)
        print(classification_report(y_test, data["y_pred"],
                                    target_names=["Fake", "Real"],
                                    zero_division=0))


# Main

def evaluate_all():
    print("=" * 60)
    print("  TruthSeeker – Evaluating trained classifiers")
    print("=" * 60)

    y_test_path = os.path.join(RESULTS_DIR, "y_test.npy")
    if not os.path.exists(y_test_path):
        print("[ERROR] y_test.npy not found. Run train.py first.")
        sys.exit(1)

    y_test  = np.load(y_test_path, allow_pickle=True)
    records = _load_models(y_test)

    if not records:
        print("[ERROR] No trained models found in", RESULTS_DIR)
        sys.exit(1)

    # Save metrics table.
    df = compute_metrics(y_test, records)
    table_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    df.to_csv(table_path, index=False)
    print(f"\n[INFO] Comparison table saved → {table_path}")
    print("\n" + df.to_string(index=False))

    # Print class reports.
    print_classification_reports(y_test, records)

    # Generate plots.
    print("\n[INFO] Generating plots…")
    plot_metric_comparison(df)
    plot_accuracy_bar(df)
    plot_confusion_matrices(y_test, records)
    plot_roc_curves(y_test, records)

    print("\n[INFO] Evaluation complete.")
    return df


if __name__ == "__main__":
    evaluate_all()

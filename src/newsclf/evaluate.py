"""Evaluation script for News Category Classifier baseline model."""

import argparse
import json
import os
import time
from typing import Any, Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from newsclf.data import LABEL_NAMES, prepare_datasets

DEFAULT_MODEL_PATH = os.path.join("models", "baseline_model.joblib")
DEFAULT_REPORTS_DIR = "reports"
DEFAULT_METRICS_PATH = os.path.join(DEFAULT_REPORTS_DIR, "metrics.json")
DEFAULT_CM_PATH = os.path.join(DEFAULT_REPORTS_DIR, "confusion_matrix.png")


def evaluate_model(
    model_path: str = DEFAULT_MODEL_PATH,
    reports_dir: str = DEFAULT_REPORTS_DIR,
    metrics_path: str = DEFAULT_METRICS_PATH,
    cm_path: str = DEFAULT_CM_PATH,
) -> Dict[str, Any]:
    """Evaluate trained model on the AG News test split.

    Saves:
        - metrics.json with overall and per-class metrics
        - confusion_matrix.png visual plot
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Please run training first."
        )

    print(f"Loading trained model from {model_path}...")
    loaded = joblib.load(model_path)
    if isinstance(loaded, dict) and "pipeline" in loaded:
        pipeline = loaded["pipeline"]
        class_names = loaded.get("labels", LABEL_NAMES)
    else:
        pipeline = loaded
        class_names = LABEL_NAMES

    print("Loading test split...")
    _, _, test_df = prepare_datasets(lowercase=True)
    X_test = test_df["text"]
    y_test = test_df["label"]
    test_count = len(test_df)
    print(f"Test samples: {test_count}")

    print("Running inference on test set...")
    t0 = time.time()
    y_pred = pipeline.predict(X_test)
    total_eval_time = time.time() - t0
    avg_latency_ms = (total_eval_time / test_count) * 1000

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    macro_precision = float(precision_score(y_test, y_pred, average="macro"))
    macro_recall = float(recall_score(y_test, y_pred, average="macro"))

    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    weighted_precision = float(precision_score(y_test, y_pred, average="weighted"))
    weighted_recall = float(recall_score(y_test, y_pred, average="weighted"))

    clf_report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
    )

    per_class_metrics = {}
    for name in class_names:
        if name in clf_report:
            per_class_metrics[name] = {
                "precision": float(clf_report[name]["precision"]),
                "recall": float(clf_report[name]["recall"]),
                "f1-score": float(clf_report[name]["f1-score"]),
                "support": int(clf_report[name]["support"]),
            }

    metrics = {
        "model_type": "TF-IDF + LogisticRegression",
        "num_test_samples": test_count,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "weighted_f1": round(weighted_f1, 4),
        "weighted_precision": round(weighted_precision, 4),
        "weighted_recall": round(weighted_recall, 4),
        "avg_latency_ms_per_sample": round(avg_latency_ms, 3),
        "per_class": per_class_metrics,
    }

    # Ensure reports directory exists
    os.makedirs(reports_dir, exist_ok=True)

    # Save metrics JSON
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Compute and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6), dpi=300)
    sns.set_theme(style="white")
    
    # Format annotations to show count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})"

    ax = sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=1,
        linecolor="#e0e0e0",
    )
    plt.title("AG News Category Classification - Confusion Matrix", fontsize=14, pad=15, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, labelpad=10)
    plt.xlabel("Predicted Label", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Print summary
    print("\n" + "=" * 55)
    print("           MODEL EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Overall Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Macro F1-Score   : {macro_f1:.4f}")
    print(f"Macro Precision  : {macro_precision:.4f}")
    print(f"Macro Recall     : {macro_recall:.4f}")
    print(f"Avg Latency/input: {avg_latency_ms:.2f} ms")
    print("-" * 55)
    print(f"{'Class':<12} {'Precision':<11} {'Recall':<11} {'F1-Score':<11} {'Support':<8}")
    print("-" * 55)
    for name, pcm in per_class_metrics.items():
        print(f"{name:<12} {pcm['precision']:<11.4f} {pcm['recall']:<11.4f} {pcm['f1-score']:<11.4f} {pcm['support']:<8}")
    print("=" * 55 + "\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline News Category Classifier")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to saved model")
    parser.add_argument("--reports-dir", type=str, default=DEFAULT_REPORTS_DIR, help="Directory to save reports")
    parser.add_argument("--metrics-name", type=str, default="metrics.json", help="Filename of metrics JSON")
    parser.add_argument("--cm-name", type=str, default="confusion_matrix.png", help="Filename of confusion matrix image")
    args = parser.parse_args()

    metrics_path = os.path.join(args.reports_dir, args.metrics_name)
    cm_path = os.path.join(args.reports_dir, args.cm_name)
    evaluate_model(
        model_path=args.model_path,
        reports_dir=args.reports_dir,
        metrics_path=metrics_path,
        cm_path=cm_path,
    )


if __name__ == "__main__":
    main()

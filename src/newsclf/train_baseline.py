"""Train baseline TF-IDF + Logistic Regression model on AG News."""

import argparse
import os
import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from newsclf.data import LABEL_NAMES, prepare_datasets

DEFAULT_MODEL_DIR = "models"
DEFAULT_MODEL_FILENAME = "baseline_model.joblib"


def build_baseline_pipeline(
    max_features: int = 50000,
    ngram_range: tuple = (1, 2),
    c_value: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """Build scikit-learn Pipeline with TF-IDF vectorizer and Logistic Regression."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                sublinear_tf=True,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=c_value,
                max_iter=max_iter,
                random_state=random_state,
                solver="lbfgs",
            ),
        ),
    ])


def train_baseline(
    model_dir: str = DEFAULT_MODEL_DIR,
    model_filename: str = DEFAULT_MODEL_FILENAME,
    max_features: int = 50000,
    ngram_range: tuple = (1, 2),
    c_value: float = 1.0,
    random_state: int = 42,
) -> str:
    """Train baseline pipeline on AG News training split and save to disk."""
    print("Loading and preparing AG News datasets (train/val/test)...")
    start_time = time.time()
    train_df, val_df, _ = prepare_datasets(
        val_size=0.1,
        random_state=random_state,
        lowercase=True,
    )
    print(f"Dataset ready. Train size: {len(train_df)}, Validation size: {len(val_df)}")

    print(f"Building TF-IDF ({ngram_range}, max_features={max_features}) + LogisticRegression pipeline...")
    pipeline = build_baseline_pipeline(
        max_features=max_features,
        ngram_range=ngram_range,
        c_value=c_value,
        random_state=random_state,
    )

    print("Fitting baseline pipeline on training data...")
    train_start = time.time()
    pipeline.fit(train_df["text"], train_df["label"])
    train_duration = time.time() - train_start
    print(f"Training completed in {train_duration:.2f} seconds.")

    # Validation score check
    val_acc = pipeline.score(val_df["text"], val_df["label"])
    print(f"Validation accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")

    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, model_filename)

    # Save model and metadata bundle
    artifact = {
        "pipeline": pipeline,
        "labels": LABEL_NAMES,
        "ngram_range": ngram_range,
        "max_features": max_features,
        "val_accuracy": val_acc,
        "train_samples": len(train_df),
        "train_duration_sec": train_duration,
    }
    joblib.dump(artifact, save_path)
    total_time = time.time() - start_time
    print(f"Model saved successfully to {save_path} (Total run time: {total_time:.2f}s)")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Train baseline TF-IDF + Logistic Regression model")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR, help="Directory to save the model")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_FILENAME, help="Filename of the saved model")
    parser.add_argument("--max-features", type=int, default=50000, help="Max features for TF-IDF")
    parser.add_argument("--c-value", type=float, default=1.0, help="Inverse regularization strength C")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_baseline(
        model_dir=args.model_dir,
        model_filename=args.model_name,
        max_features=args.max_features,
        c_value=args.c_value,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()

import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from flask import Blueprint, Response, jsonify, request, send_file

# Set up backend logging
logger = logging.getLogger("newsclf.backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Ensure src/newsclf is importable from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from newsclf.data import LABEL_NAMES
from newsclf.predict import DEFAULT_BASELINE_PATH, explain_prediction, load_model, predict, predict_batch

api_bp = Blueprint("api", __name__)

BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.joblib"
TRANSFORMER_MODEL_PATH = PROJECT_ROOT / "models" / "transformer_model"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "reports" / "confusion_matrix.png"


def is_transformer_ready() -> bool:
    """Check if fine-tuned transformer model weights actually exist on disk."""
    if not TRANSFORMER_MODEL_PATH.is_dir():
        return False
    return (
        (TRANSFORMER_MODEL_PATH / "config.json").exists()
        or (TRANSFORMER_MODEL_PATH / "model.safetensors").exists()
        or (TRANSFORMER_MODEL_PATH / "pytorch_model.bin").exists()
    )


def resolve_model_path(model_name: Optional[str] = "baseline") -> str:
    """Resolve and validate model path according to availability."""
    name = (model_name or "baseline").strip().lower()

    if name in ["baseline", "tf-idf", "tfidf", "logistic_regression", "logisticregression"]:
        if not BASELINE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Baseline model not found at '{BASELINE_MODEL_PATH}'. Please train the baseline model first."
            )
        return str(BASELINE_MODEL_PATH)

    if name in ["distilbert", "transformer"]:
        if not is_transformer_ready():
            raise FileNotFoundError(
                "DistilBERT model weights not found in 'models/transformer_model'. "
                "Only the baseline model is currently trained and available."
            )
        return str(TRANSFORMER_MODEL_PATH)

    raise ValueError(
        f"Unsupported model '{model_name}'. Available architectures: 'baseline', 'distilbert'."
    )


@api_bp.route("/", methods=["GET"])
def index():
    """Root endpoint confirming API is running."""
    return jsonify({
        "message": "News Category Classifier API is running",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "health": "/api/health",
            "models": "/api/models",
            "predict": "/api/predict",
            "batch_predict": "/api/batch-predict",
            "metrics": "/api/metrics",
            "confusion_matrix": "/api/confusion-matrix",
        },
    })


@api_bp.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@api_bp.route("/api/models", methods=["GET"])
def list_models():
    """List available model architectures and their readiness on disk."""
    transformer_available = is_transformer_ready()
    baseline_available = BASELINE_MODEL_PATH.exists()

    return jsonify({
        "models": [
            {
                "id": "baseline",
                "name": "Baseline (TF-IDF + Logistic Regression)",
                "available": baseline_available,
                "description": "Ultra-fast CPU inference (< 0.1 ms latency, 91.6% accuracy)",
                "default": True,
            },
            {
                "id": "distilbert",
                "name": "DistilBERT (Fine-tuned Transformer)",
                "available": transformer_available,
                "description": "Fine-tuned transformer model (requires training weights)",
                "default": False,
            },
        ]
    })


@api_bp.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict category for a single text snippet.

    Request format (JSON):
        {
            "text": "Messi scores twice in Champions League final",
            "model": "baseline"
        }

    Response format (JSON):
        {
            "label": "Sports",
            "confidence": 0.96,
            "top_k_probs": [
                {"label": "Sports", "probability": 0.96},
                {"label": "World", "probability": 0.02},
                ...
            ],
            "uncertain": false
        }
    """
    if not request.is_json:
        return jsonify({"error": "⚠ Invalid request body. Expected JSON object with Content-Type: application/json."}), 400

    try:
        payload = request.get_json(silent=False)
    except Exception as exc:
        logger.warning("Malformed JSON in request: %s", exc)
        return jsonify({"error": "⚠ Invalid JSON payload in request body."}), 400

    if not isinstance(payload, dict):
        return jsonify({"error": "⚠ Invalid JSON format: request body must be a JSON object."}), 400

    if "text" not in payload:
        return jsonify({"error": "⚠ Please enter some news text."}), 400

    text = payload.get("text")
    if text is None:
        return jsonify({"error": "⚠ Please enter some news text."}), 400

    text_str = str(text).strip()
    if not text_str:
        return jsonify({
            "label": "Unknown",
            "confidence": 0.0,
            "top_k_probs": [
                {"label": label, "probability": 0.0} for label in LABEL_NAMES
            ],
            "uncertain": True,
            "explanation": [],
            "warning": "⚠ Please enter some news text.",
        }), 200

    # Max length validation (50,000 characters)
    if len(text_str) > 50000:
        return jsonify({
            "error": "⚠ Input text is too long (maximum 50,000 characters). Please provide a shorter headline or excerpt."
        }), 400

    model_name = payload.get("model", "baseline")
    try:
        model_path = resolve_model_path(model_name)
    except (FileNotFoundError, ValueError) as err:
        logger.warning("Model resolution error: %s", err)
        return jsonify({"error": f"⚠ Invalid model selection: {str(err)}"}), 400

    try:
        raw_result = predict(text=text_str, model_path=model_path, explain=True)
    except Exception as exc:
        logger.exception("Prediction failed for input '%s...': %s", text_str[:100], exc)
        return jsonify({"error": "⚠ An internal prediction error occurred while processing the request. Please try again."}), 500

    confidence = float(raw_result.get("confidence", 0.0))
    raw_top_k = raw_result.get("top_k_probs", {})

    # Adapter: transform dict {label: prob} to list of objects [{"label": ..., "probability": ...}]
    top_k_probs = [
        {"label": lbl, "probability": round(float(prob), 4)}
        for lbl, prob in raw_top_k.items()
    ]

    # Ensure all 4 classes are represented in top_k_probs if any was omitted
    existing_labels = {item["label"] for item in top_k_probs}
    for lbl in LABEL_NAMES:
        if lbl not in existing_labels:
            top_k_probs.append({"label": lbl, "probability": 0.0})

    # Sort descending by probability
    top_k_probs.sort(key=lambda item: item["probability"], reverse=True)

    # Uncertainty rule: confidence < 0.60 OR extremely short text (< 5 chars or < 2 words)
    is_short = len(text_str) < 5 or len(text_str.split()) < 2
    uncertain = confidence < 0.60 or is_short

    explanation = raw_result.get("explanation")
    if explanation is None:
        explanation = explain_prediction(
            text=text_str,
            model_path=model_path,
            top_n=8,
            target_class=raw_result.get("label"),
        )

    response_data = {
        "label": raw_result.get("label", "Unknown"),
        "confidence": round(confidence, 4),
        "top_k_probs": top_k_probs,
        "uncertain": uncertain,
        "explanation": explanation or [],
    }

    if is_short:
        response_data["warning"] = "⚠ The text is too short for reliable classification."

    return jsonify(response_data)


@api_bp.route("/api/batch-predict", methods=["POST"])
def api_batch_predict():
    """Predict categories for a batch of texts uploaded in a CSV file.

    Accepts:
        - multipart/form-data with file=<CSV File>
        - Optional form field or query param 'text_column': Column name with headlines/articles
        - Optional form field or query param 'model': Model name ('baseline' / 'distilbert')
        - Optional form field or query param 'format': 'json' (default) or 'csv' (for download)
    """
    if "file" not in request.files:
        return jsonify({"error": "⚠ No file uploaded. Please select a CSV file with key 'file'."}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "⚠ Empty file or no file selected. Please select a CSV file to upload."}), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "⚠ Invalid file type. Please upload a valid CSV file (.csv)."}), 400

    # Multi-encoding support with graceful fallback
    df = None
    parse_err = None
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=encoding, skip_blank_lines=False)
            break
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            parse_err = exc
            continue

    if df is None:
        logger.warning("Failed to decode or parse CSV '%s': %s", file.filename, parse_err)
        if isinstance(parse_err, pd.errors.EmptyDataError):
            return jsonify({"error": "⚠ The uploaded CSV file contains no data rows."}), 400
        return jsonify({
            "error": "⚠ Unable to decode CSV file. Please ensure the file is a valid CSV encoded in UTF-8 format."
        }), 400

    if df.empty or len(df.columns) == 0:
        return jsonify({"error": "⚠ The uploaded CSV file contains no data rows."}), 400

    MAX_BATCH_ROWS = 10000
    if len(df) > MAX_BATCH_ROWS:
        return jsonify({
            "error": f"⚠ Uploaded CSV has {len(df):,} rows, which exceeds the maximum limit of {MAX_BATCH_ROWS:,} rows per batch. Please upload in smaller chunks."
        }), 400

    # Determine text column
    requested_col = request.form.get("text_column") or request.args.get("text_column")
    text_col = None

    if requested_col:
        if requested_col not in df.columns:
            return jsonify({
                "error": f"⚠ Please select a valid text column. Column '{requested_col}' not found in CSV. Available columns: {list(df.columns)}"
            }), 400
        text_col = requested_col
    else:
        # Auto-detect sensible text column name
        candidates = ["text", "headline", "title", "content", "article", "news", "description"]
        for cand in candidates:
            for col in df.columns:
                if col.lower() == cand:
                    text_col = col
                    break
            if text_col:
                break

        # Fallback to first object/string column if no named candidate found
        if not text_col:
            object_cols = [c for c in df.columns if df[c].dtype == object]
            if object_cols:
                text_col = object_cols[0]
            else:
                text_col = df.columns[0]

    # Validate that text column has at least some valid non-empty text rows
    cleaned_entries = df[text_col].dropna().astype(str).str.strip()
    cleaned_non_blank = cleaned_entries[cleaned_entries != ""]
    if cleaned_non_blank.empty:
        return jsonify({
            "error": f"⚠ Selected column '{text_col}' contains only empty or blank text rows. Please select a valid text column."
        }), 400

    model_name = request.form.get("model") or request.args.get("model") or "baseline"
    try:
        model_path = resolve_model_path(model_name)
    except (FileNotFoundError, ValueError) as err:
        logger.warning("Batch model resolution error: %s", err)
        return jsonify({"error": f"⚠ Invalid model selection: {str(err)}"}), 400

    # Extract string texts
    raw_texts = df[text_col].fillna("").astype(str).tolist()

    try:
        batch_results = predict_batch(raw_texts, model_path=model_path)
    except Exception as exc:
        logger.exception("Batch prediction execution failed: %s", exc)
        return jsonify({
            "error": "⚠ An internal error occurred during batch processing. Please check backend logs."
        }), 500

    predicted_labels = [res["label"] for res in batch_results]
    confidences = [res["confidence"] for res in batch_results]

    export_format = (
        request.form.get("format") or request.args.get("format") or "json"
    ).lower()

    if export_format == "csv":
        # Return CSV file attachment
        export_df = df.copy()
        export_df["predicted_category"] = predicted_labels
        export_df["confidence"] = confidences
        export_df["uncertain"] = [c < 0.60 for c in confidences]

        output = io.StringIO()
        export_df.to_csv(output, index=False)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=classified_news_predictions.csv"},
        )

    # Default: Return JSON with predictions list
    predictions = []
    for orig_text, label, conf in zip(raw_texts, predicted_labels, confidences):
        predictions.append({
            "text": orig_text,
            "predicted_category": label,
            "confidence": conf,
            "uncertain": conf < 0.60,
        })

    return jsonify({
        "total_records": len(predictions),
        "text_column": text_col,
        "available_columns": list(df.columns),
        "predictions": predictions,
    })


@api_bp.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Return model evaluation metrics from reports/metrics.json."""
    if not METRICS_PATH.exists():
        return jsonify({
            "error": "Metrics report 'reports/metrics.json' not found. "
            "Please run 'python -m newsclf.evaluate' to generate evaluation metrics."
        }), 404

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        return jsonify(metrics_data)
    except Exception as exc:
        return jsonify({"error": f"Failed to read metrics file: {str(exc)}"}), 500


@api_bp.route("/api/confusion-matrix", methods=["GET"])
def api_confusion_matrix():
    """Serve the existing confusion matrix image from reports/confusion_matrix.png."""
    if not CONFUSION_MATRIX_PATH.exists():
        return jsonify({
            "error": "Confusion matrix artifact 'reports/confusion_matrix.png' not found. "
            "Please run 'python -m newsclf.evaluate' to generate confusion matrix."
        }), 404

    return send_file(CONFUSION_MATRIX_PATH, mimetype="image/png")

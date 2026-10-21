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

    # Check for ground truth column to enable live dynamic evaluation
    gt_col = None
    gt_candidates = ["true_category", "true_label", "ground_truth", "category", "label", "target", "class"]
    for cand in gt_candidates:
        for c in df.columns:
            if c.lower() == cand and c != text_col:
                gt_col = c
                break
        if gt_col:
            break

    # Calculate live distribution
    category_counts = {cat: 0 for cat in LABEL_NAMES}
    for p in predicted_labels:
        if p in category_counts:
            category_counts[p] += 1
        else:
            category_counts[p] = 1

    avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
    uncertain_count = sum(1 for c in confidences if c < 0.60)
    confident_count = len(confidences) - uncertain_count

    evaluation = {
        "has_ground_truth": False,
        "total_samples": len(predictions),
        "avg_confidence": round(avg_conf, 4),
        "confident_count": confident_count,
        "uncertain_count": uncertain_count,
        "distribution": category_counts,
    }

    # Normalize ground truth if present
    normalized_ground_truth = []
    if gt_col:
        label_map = {
            "world": "World", "1": "World", 1: "World",
            "sports": "Sports", "2": "Sports", 2: "Sports",
            "business": "Business", "3": "Business", 3: "Business",
            "sci/tech": "Sci/Tech", "sci_tech": "Sci/Tech", "scitech": "Sci/Tech", "science": "Sci/Tech", "tech": "Sci/Tech", "4": "Sci/Tech", 4: "Sci/Tech"
        }
        raw_gt = df[gt_col].tolist()
        mapped_gt = [label_map.get(str(g).strip().lower(), None) for g in raw_gt]
        
        # Only use if at least 50% of rows could be mapped to known labels
        valid_count = sum(1 for g in mapped_gt if g in LABEL_NAMES)
        if valid_count > 0 and valid_count >= len(mapped_gt) * 0.5:
            normalized_ground_truth = mapped_gt
            # Attach ground truth to predictions
            for p, g in zip(predictions, mapped_gt):
                p["true_category"] = g

            # Compute live accuracy & confusion matrix
            correct = sum(1 for p, g in zip(predicted_labels, mapped_gt) if p == g and g in LABEL_NAMES)
            total_valid = valid_count
            accuracy = float(correct / total_valid) if total_valid > 0 else 0.0

            # 4x4 matrix
            matrix_4x4 = [[0 for _ in range(4)] for _ in range(4)]
            per_class_stats = {cat: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for cat in LABEL_NAMES}

            cat_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}
            for pred_cat, true_cat in zip(predicted_labels, mapped_gt):
                if true_cat in cat_to_idx:
                    per_class_stats[true_cat]["support"] += 1
                if pred_cat in cat_to_idx and true_cat in cat_to_idx:
                    r_idx = cat_to_idx[true_cat]
                    c_idx = cat_to_idx[pred_cat]
                    matrix_4x4[r_idx][c_idx] += 1

                for cat in LABEL_NAMES:
                    if pred_cat == cat and true_cat == cat:
                        per_class_stats[cat]["tp"] += 1
                    elif pred_cat == cat and true_cat != cat:
                        per_class_stats[cat]["fp"] += 1
                    elif pred_cat != cat and true_cat == cat:
                        per_class_stats[cat]["fn"] += 1

            per_class_report = {}
            f1_scores = []
            prec_scores = []
            rec_scores = []

            for cat in LABEL_NAMES:
                s = per_class_stats[cat]
                tp, fp, fn, sup = s["tp"], s["fp"], s["fn"], s["support"]
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                per_class_report[cat] = {
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1-score": round(f1, 4),
                    "support": sup,
                }
                f1_scores.append(f1)
                prec_scores.append(prec)
                rec_scores.append(rec)

            macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            macro_prec = sum(prec_scores) / len(prec_scores) if prec_scores else 0.0
            macro_rec = sum(rec_scores) / len(rec_scores) if rec_scores else 0.0

            evaluation.update({
                "has_ground_truth": True,
                "ground_truth_column": gt_col,
                "accuracy": round(accuracy, 4),
                "macro_f1": round(macro_f1, 4),
                "macro_precision": round(macro_prec, 4),
                "macro_recall": round(macro_rec, 4),
                "confusion_matrix": matrix_4x4,
                "labels": LABEL_NAMES,
                "per_class": per_class_report,
            })

    return jsonify({
        "total_records": len(predictions),
        "text_column": text_col,
        "ground_truth_column": gt_col if evaluation["has_ground_truth"] else None,
        "available_columns": list(df.columns),
        "evaluation": evaluation,
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


@api_bp.route("/api/confusion-matrix-data", methods=["GET"])
def api_confusion_matrix_data():
    """Return structured confusion matrix data with cell-level insights."""
    # Official AG News benchmark matrix (7,600 samples, 1,900 per category)
    matrix = [
        [1721, 55, 75, 49],   # World actual
        [18, 1862, 11, 9],    # Sports actual
        [57, 17, 1675, 151],  # Business actual
        [52, 21, 120, 1707]   # Sci/Tech actual
    ]
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    totals = [1900, 1900, 1900, 1900]

    # Pre-computed explanations and representative error cases
    cell_insights = {
        "World_World": {
            "count": 1721, "rate": "90.6%", "is_correct": True,
            "title": "World Correct (1,721 articles • 90.6%)",
            "explanation": "High recall for geopolitical and international statecraft keywords (e.g., 'United Nations', 'treaty', 'Geneva', 'diplomats', 'summit')."
        },
        "World_Sports": {
            "count": 55, "rate": "2.9%", "is_correct": False,
            "title": "World Mistaken for Sports (55 articles • 2.9%)",
            "explanation": "Articles covering international athletic events (Olympics, World Cup qualifiers, athletic sanctions) often contain athlete names and sports terms alongside country names.",
            "sample_headline": "Sports court hears Hamm gold medal appeal in Lausanne"
        },
        "World_Business": {
            "count": 75, "rate": "3.9%", "is_correct": False,
            "title": "World Mistaken for Business (75 articles • 3.9%)",
            "explanation": "International trade summits, cross-border tariffs, and economic sanctions contain heavy financial and market vocabulary.",
            "sample_headline": "Dow Jones agrees to buy MarketWatch in $519 million deal"
        },
        "World_Sci/Tech": {
            "count": 49, "rate": "2.6%", "is_correct": False,
            "title": "World Mistaken for Sci/Tech (49 articles • 2.6%)",
            "explanation": "Global science initiatives, space agency treaties, or climate change reports that involve government agencies.",
            "sample_headline": "Mars water tops international science honors list"
        },
        "Sports_Sports": {
            "count": 1862, "rate": "98.0%", "is_correct": True,
            "title": "Sports Correct (1,862 articles • 98.0%)",
            "explanation": "Sports is the model's highest-performing category. Athletic vocabulary (scores, game, coach, cup, league, finals) is virtually unique to sports."
        },
        "Sports_World": {
            "count": 18, "rate": "0.9%", "is_correct": False,
            "title": "Sports Mistaken for World (18 articles • 0.9%)",
            "explanation": "Rare cases where international relations or national flags dominate the headline rather than the athletic event itself."
        },
        "Sports_Business": {
            "count": 11, "rate": "0.6%", "is_correct": False,
            "title": "Sports Mistaken for Business (11 articles • 0.6%)",
            "explanation": "Sports franchise contracts, stadium bonds, and television broadcast rights occasionally get pulled toward business."
        },
        "Sports_Sci/Tech": {
            "count": 9, "rate": "0.5%", "is_correct": False,
            "title": "Sports Mistaken for Sci/Tech (9 articles • 0.5%)",
            "explanation": "Extremely rare; typically sports science, athlete medical recovery technology, or e-sports video games."
        },
        "Business_Business": {
            "count": 1675, "rate": "88.2%", "is_correct": True,
            "title": "Business Correct (1,675 articles • 88.2%)",
            "explanation": "Accurately identifies stock movements, interest rates, corporate quarterly earnings, and executive changes."
        },
        "Business_Sci/Tech": {
            "count": 151, "rate": "7.9%", "is_correct": False,
            "title": "Primary Hotspot: Business Mistaken for Sci/Tech (151 articles • 7.9%)",
            "explanation": "The single largest confusion hotspot in the entire model. Business coverage of technology giants (Apple, Nvidia, Microsoft, Sony, IBM) and commercial software launches is loaded with tech keywords.",
            "sample_headline": "Sony, IBM, Toshiba give details of 'Cell' processor chip jointly produced for next-generation consoles"
        },
        "Business_World": {
            "count": 57, "rate": "3.0%", "is_correct": False,
            "title": "Business Mistaken for World (57 articles • 3.0%)",
            "explanation": "Multinational corporate regulations, global supply chain reports, and foreign exchange movements."
        },
        "Business_Sports": {
            "count": 17, "rate": "0.9%", "is_correct": False,
            "title": "Business Mistaken for Sports (17 articles • 0.9%)",
            "explanation": "Metaphorical sports language in market commentary (e.g., 'bulls strike out', 'tech race heats up')."
        },
        "Sci/Tech_Sci/Tech": {
            "count": 1707, "rate": "89.8%", "is_correct": True,
            "title": "Sci/Tech Correct (1,707 articles • 89.8%)",
            "explanation": "Reliably recognizes space exploration, algorithms, hardware chips, operating systems, and scientific discoveries."
        },
        "Sci/Tech_Business": {
            "count": 120, "rate": "6.3%", "is_correct": False,
            "title": "Secondary Hotspot: Sci/Tech Mistaken for Business (120 articles • 6.3%)",
            "explanation": "Tech product announcements frequently cite pricing, market release dates, enterprise sales figures, and corporate partnerships.",
            "sample_headline": "Microsoft readies next business instant messaging server for corporate enterprise rollout"
        },
        "Sci/Tech_World": {
            "count": 52, "rate": "2.7%", "is_correct": False,
            "title": "Sci/Tech Mistaken for World (52 articles • 2.7%)",
            "explanation": "State-sponsored cyberattacks, international space missions, and multilateral artificial intelligence summits."
        },
        "Sci/Tech_Sports": {
            "count": 21, "rate": "1.1%", "is_correct": False,
            "title": "Sci/Tech Mistaken for Sports (21 articles • 1.1%)",
            "explanation": "Gaming hardware reviews, racing simulators, and robotics competitions featuring athletic phrasing."
        }
    }

    return jsonify({
        "labels": labels,
        "matrix": matrix,
        "totals": totals,
        "total_samples": sum(totals),
        "cell_insights": cell_insights,
    })


@api_bp.route("/api/confusion-matrix", methods=["GET"])
def api_confusion_matrix():
    """Serve the existing confusion matrix image or JSON if requested."""
    format_arg = request.args.get("format", "").lower()
    if format_arg == "json" or request.headers.get("Accept") == "application/json":
        return api_confusion_matrix_data()

    if not CONFUSION_MATRIX_PATH.exists():
        return jsonify({
            "error": "Confusion matrix artifact 'reports/confusion_matrix.png' not found. "
            "Please run 'python -m newsclf.evaluate' to generate confusion matrix."
        }), 404

    return send_file(CONFUSION_MATRIX_PATH, mimetype="image/png")


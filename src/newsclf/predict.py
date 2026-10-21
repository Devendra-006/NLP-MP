"""Inference and prediction module for News Category Classifier."""

import os
import threading
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from newsclf.data import ID2LABEL, LABEL_NAMES, clean_text

DEFAULT_BASELINE_PATH = os.path.join("models", "baseline_model.joblib")

_LOADED_MODELS: Dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()
_FEATURE_NAMES_CACHE: Dict[str, Any] = {}
_STOP_WORDS_SET = frozenset(ENGLISH_STOP_WORDS)


def is_transformer_model(model_path: str) -> bool:
    """Check if model path refers to a Hugging Face saved model directory."""
    return os.path.isdir(model_path) and (
        os.path.exists(os.path.join(model_path, "config.json"))
        or os.path.exists(os.path.join(model_path, "model.safetensors"))
        or os.path.exists(os.path.join(model_path, "pytorch_model.bin"))
    )


def load_model(model_path: Optional[str] = None) -> Any:
    """Load model pipeline from disk and cache in memory for high-performance inference."""
    if model_path is None:
        model_path = DEFAULT_BASELINE_PATH

    norm_path = os.path.abspath(model_path)
    if norm_path in _LOADED_MODELS:
        return _LOADED_MODELS[norm_path]

    with _MODEL_LOCK:
        # Double-check inside lock
        if norm_path in _LOADED_MODELS:
            return _LOADED_MODELS[norm_path]

        if not os.path.exists(norm_path):
            raise FileNotFoundError(
                f"Model file or directory not found at '{norm_path}'. Please train the model first."
            )

        if is_transformer_model(norm_path):
            from transformers import pipeline as hf_pipeline
            pipe = hf_pipeline(
                "text-classification",
                model=norm_path,
                tokenizer=norm_path,
                top_k=None,
            )
            _LOADED_MODELS[norm_path] = ("transformer", pipe)
            return _LOADED_MODELS[norm_path]

        loaded = joblib.load(norm_path)
        if isinstance(loaded, dict) and "pipeline" in loaded:
            pipeline = loaded["pipeline"]
        else:
            pipeline = loaded

        _LOADED_MODELS[norm_path] = ("baseline", pipeline)
        return _LOADED_MODELS[norm_path]


def explain_prediction(
    text: str,
    model_path: Optional[str] = None,
    top_n: int = 8,
    clean: bool = True,
    target_class: Optional[Union[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Calculate feature contributions for the predicted class.

    For TF-IDF + Logistic Regression:
        contribution = TF-IDF feature value * classifier coefficient for target class

    Args:
        text: Input headline or article body text.
        model_path: Optional path to joblib baseline model file.
        top_n: Number of top contributing features to return (default: 8).
        clean: Whether to apply text cleaning before vectorization (default: True).
        target_class: Optional class label or index to explain (default: predicted class).

    Returns:
        List of dicts: [{'feature': '...', 'contribution': float}, ...]
    """
    if text is None or not str(text).strip():
        return []

    model_type, model_obj = load_model(model_path)
    if model_type != "baseline":
        return []

    if (
        not hasattr(model_obj, "named_steps")
        or "tfidf" not in model_obj.named_steps
        or "clf" not in model_obj.named_steps
    ):
        return []

    tfidf = model_obj.named_steps["tfidf"]
    clf = model_obj.named_steps["clf"]

    processed = clean_text(text, lowercase=True) if clean else str(text)
    vec = tfidf.transform([processed])
    if vec.nnz == 0:
        return []

    classes = getattr(clf, "classes_", np.arange(len(LABEL_NAMES)))
    if target_class is None:
        probs = clf.predict_proba(vec)[0]
        class_idx = int(np.argmax(probs))
    elif isinstance(target_class, str) and target_class in LABEL_NAMES:
        class_idx = LABEL_NAMES.index(target_class)
    elif isinstance(target_class, int) and 0 <= target_class < len(classes):
        class_idx = target_class
    else:
        probs = clf.predict_proba(vec)[0]
        class_idx = int(np.argmax(probs))

    coefs = clf.coef_[class_idx]
    norm_path = os.path.abspath(model_path or DEFAULT_BASELINE_PATH)
    if norm_path not in _FEATURE_NAMES_CACHE:
        _FEATURE_NAMES_CACHE[norm_path] = tfidf.get_feature_names_out()
    feature_names = _FEATURE_NAMES_CACHE[norm_path]

    contributions = []
    for idx, val in zip(vec.indices, vec.data):
        feat = feature_names[idx]
        words = feat.split()
        # Filter meaningless stopwords or n-grams starting/ending with a stopword
        if (
            feat in _STOP_WORDS_SET
            or words[0] in _STOP_WORDS_SET
            or words[-1] in _STOP_WORDS_SET
        ):
            continue

        contrib = float(val * coefs[idx])
        contributions.append({
            "feature": feat,
            "contribution": round(contrib, 4),
        })

    # Sort descending by contribution
    contributions.sort(key=lambda item: item["contribution"], reverse=True)

    positive_contribs = [c for c in contributions if c["contribution"] > 0]
    if positive_contribs:
        return positive_contribs[:top_n]
    return contributions[:top_n]


def predict(
    text: str,
    model_path: Optional[str] = None,
    top_k: int = 4,
    clean: bool = True,
    explain: bool = False,
    top_n_features: int = 8,
) -> Dict[str, Any]:
    """Predict news category for a single text input.

    Args:
        text: Input headline or article body text.
        model_path: Optional path to joblib baseline or transformer model directory.
        top_k: Number of top class probabilities to return (default: 4).
        clean: Whether to apply text cleaning before vectorization (default: True).
        explain: Whether to include feature contribution explanation (default: False).
        top_n_features: Number of top contributing features for explanation (default: 8).

    Returns:
        Dict with keys:
            - 'label': predicted class name (str)
            - 'confidence': confidence score between 0.0 and 1.0 (float)
            - 'top_k_probs': dict mapping class names to probability scores (Dict[str, float])
            - 'explanation': (optional) list of top contributing features if explain=True
    """
    if text is None or not str(text).strip():
        res = {
            "label": "Unknown",
            "confidence": 0.0,
            "top_k_probs": {label: 0.0 for label in LABEL_NAMES[:top_k]},
        }
        if explain:
            res["explanation"] = []
        return res

    model_type, model_obj = load_model(model_path)

    if model_type == "transformer":
        # Transformer: preserve case during cleaning
        processed_text = clean_text(text, lowercase=False) if clean else str(text)
        # HF pipeline with top_k=None returns list of dicts: [{'label': '...', 'score': ...}, ...]
        raw_outputs = model_obj(processed_text)[0]
        class_probs: Dict[str, float] = {}
        for item in raw_outputs:
            raw_label = item["label"]
            # Handle possible LABEL_0 or class names
            if raw_label.startswith("LABEL_") and raw_label[6:].isdigit():
                idx = int(raw_label[6:])
                name = ID2LABEL.get(idx, raw_label)
            else:
                name = raw_label
            class_probs[name] = float(item["score"])
    else:
        # Baseline TF-IDF + Logistic Regression
        processed_text = clean_text(text, lowercase=True) if clean else str(text)
        probs = model_obj.predict_proba([processed_text])[0]
        classes = getattr(model_obj, "classes_", np.arange(len(probs)))

        class_probs = {}
        for idx, prob in zip(classes, probs):
            label_name = LABEL_NAMES[idx] if idx < len(LABEL_NAMES) else f"Class_{idx}"
            class_probs[label_name] = float(prob)

    # Sort descending by probability
    sorted_classes = sorted(class_probs.items(), key=lambda item: item[1], reverse=True)
    best_label, best_conf = sorted_classes[0]

    top_k_probs = {k: round(v, 4) for k, v in sorted_classes[:top_k]}

    is_short = len(str(text).strip()) < 5 or len(str(text).split()) < 2
    is_uncertain = bool(best_conf < 0.60 or is_short)

    res = {
        "label": best_label,
        "confidence": round(float(best_conf), 4),
        "top_k_probs": top_k_probs,
        "uncertain": is_uncertain,
    }

    if explain:
        res["explanation"] = explain_prediction(
            text=text,
            model_path=model_path,
            top_n=top_n_features,
            clean=clean,
            target_class=best_label,
        )

    return res


def predict_batch(
    texts: List[str],
    model_path: Optional[str] = None,
    clean: bool = True,
) -> List[Dict[str, Any]]:
    """Predict news categories for a batch of input texts.

    Args:
        texts: List of headline/article text strings.
        model_path: Optional path to model file or directory.
        clean: Whether to clean texts before prediction.

    Returns:
        List of prediction dictionaries.
    """
    if not texts:
        return []

    model_type, model_obj = load_model(model_path)

    if model_type == "transformer":
        results = [predict(t, model_path=model_path, clean=clean) for t in texts]
        return results

    # Distinguish empty/whitespace texts from non-empty texts
    results: List[Optional[Dict[str, Any]]] = [None] * len(texts)
    non_empty_indices = [i for i, t in enumerate(texts) if t is not None and str(t).strip()]
    empty_indices = [i for i, t in enumerate(texts) if t is None or not str(t).strip()]

    for i in empty_indices:
        results[i] = {
            "label": "Unknown",
            "confidence": 0.0,
            "top_k_probs": {label: 0.0 for label in LABEL_NAMES},
        }

    if non_empty_indices:
        non_empty_texts = [texts[i] for i in non_empty_indices]
        processed = [clean_text(t, lowercase=True) if clean else str(t) for t in non_empty_texts]
        probs_matrix = model_obj.predict_proba(processed)
        classes = getattr(model_obj, "classes_", np.arange(probs_matrix.shape[1]))

        class_label_list = [LABEL_NAMES[c] if c < len(LABEL_NAMES) else f"Class_{c}" for c in classes]
        best_indices = np.argmax(probs_matrix, axis=1)
        best_confs = probs_matrix[np.arange(len(non_empty_indices)), best_indices]

        for orig_idx, probs, best_idx, best_conf in zip(
            non_empty_indices, probs_matrix, best_indices, best_confs
        ):
            sorted_order = np.argsort(probs)[::-1]
            top_k_probs = {class_label_list[k]: round(float(probs[k]), 4) for k in sorted_order}
            results[orig_idx] = {
                "label": class_label_list[best_idx],
                "confidence": round(float(best_conf), 4),
                "top_k_probs": top_k_probs,
            }

    return [r for r in results if r is not None]


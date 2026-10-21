# 📰 News Category Classifier

A fast, production-ready Python package and interactive Streamlit web application that classifies English news headlines and articles into four categories (**World**, **Sports**, **Business**, **Sci/Tech**) with calibrated confidence scores and class probability distributions.

---

## 🎯 Features & Goals

- **High-Accuracy Baseline**: TF-IDF (1–2 n-grams, max 50k features) + Logistic Regression achieving **91.64% test accuracy** (well above the 88% PRD target).
- **Ultra-Low Latency**: Sub-millisecond CPU inference (**~0.05 ms per prediction**), far beating the 50 ms non-functional requirement.
- **Reproducible Pipeline**: Stratified train/validation split (10% of train, fixed seed `42`), automated metrics reporting (`reports/metrics.json`), and high-resolution confusion matrix generation (`reports/confusion_matrix.png`).
- **Interactive Streamlit Web UI**: Real-time prediction, probability bar chart, quick presets, model selector (Baseline / DistilBERT), low-confidence alerts (< 60%), and batch CSV upload/download.
- **Developer-Friendly API & CLI**: Full `predict(text)` Python API, batch prediction, and terminal CLI (`python -m newsclf predict "..."`).
- **Explainable AI (XAI) / Why This Prediction?**: Feature contribution breakdown for each prediction ($W_{k, j} \times \text{TF-IDF}_j$), explaining top influential words and phrases with stopword filtering.
- **Transformer Support**: Training pipeline for fine-tuning `distilbert-base-uncased` with Hugging Face Trainer.

---

## 📊 Benchmark & Evaluation Results

Evaluated on the full official **AG News** test set (7,600 samples):

| Metric | Baseline (TF-IDF + LogReg) | Target Threshold |
| :--- | :--- | :--- |
| **Overall Accuracy** | **91.64%** (0.9164) | $\ge$ 88.0% |
| **Macro F1-Score** | **0.9163** | $\ge$ 0.880 |
| **Macro Precision** | **0.9163** | — |
| **Macro Recall** | **0.9164** | — |
| **Avg Latency / Sample** | **0.05 ms** (CPU) | $<$ 50 ms |

### Per-Class Metrics Breakdown

| Category | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| 🌍 **World** | 93.13% | 90.58% | **0.9184** | 1,900 |
| ⚽ **Sports** | 95.24% | 98.00% | **0.9660** | 1,900 |
| 📈 **Business** | 89.05% | 88.16% | **0.8860** | 1,900 |
| 🔬 **Sci/Tech** | 89.09% | 89.84% | **0.8947** | 1,900 |

*Generated artifacts: [metrics.json](reports/metrics.json) and [confusion_matrix.png](reports/confusion_matrix.png).*

---

## 🗂️ Project Structure

```
News_Category_Classifier/
├── frontend/                  # Modern Web Dashboard (HTML5, CSS3, Vanilla JS)
│   ├── index.html
│   ├── style.css
│   └── script.js
├── backend/                   # Flask REST API Service
│   ├── __init__.py
│   ├── app.py                 # Flask server entry point & CORS configuration
│   └── api.py                 # REST API endpoints & prediction adapter
├── src/newsclf/               # Core ML package
│   ├── __init__.py
│   ├── __main__.py            # Entry point for python -m newsclf
│   ├── data.py                # Load, clean, and stratified split
│   ├── train_baseline.py      # TF-IDF + Logistic Regression training
│   ├── train_transformer.py   # DistilBERT fine-tuning script
│   ├── evaluate.py            # Evaluation & confusion matrix generator
│   ├── predict.py             # predict() & batch inference
│   └── cli.py                 # Command-line interface
├── app/
│   └── streamlit_app.py       # Modern Streamlit web application
├── models/                    # Serialized models
│   └── baseline_model.joblib
├── data/                      # Cached dataset
├── reports/                   # Evaluation artifacts
│   ├── metrics.json
│   └── confusion_matrix.png
├── tests/
│   ├── test_data.py           # Unit tests for text cleaning and data
│   ├── test_predict.py        # Unit tests for inference & acceptance criteria
│   └── test_backend.py        # Unit tests for Flask REST API endpoints
├── requirements.txt           # Core Python dependencies
├── pyproject.toml             # Packaging specification
└── README.md                  # Project documentation
```

---

## 🚀 Quickstart & Setup

### 1. Clone & Set Up Virtual Environment

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 🛠️ Usage Instructions

### 1. Train the Baseline Model

Trains the TF-IDF + Logistic Regression pipeline on the stratified AG News training set (108k train, 12k val) and saves the serialized bundle to `models/baseline_model.joblib`:

```bash
python -m newsclf.train_baseline
```

### 2. Evaluate Model & Generate Reports

Evaluates the saved model against the 7,600 test examples, prints classification metrics, and outputs `reports/metrics.json` and `reports/confusion_matrix.png`:

```bash
python -m newsclf.evaluate
```

### 3. Run the Web Application (Modern Frontend + Flask REST API)

The modern decoupled web application provides a full-featured UI powered by a separate Flask REST API backend:

- **Frontend:** HTML5, CSS3 (vanilla custom design system, glassmorphism, responsive cards), and Vanilla JavaScript (Fetch API, zero framework bloat).
- **Backend:** Flask REST API with CORS support, request validation, and clean adapters to the ML inference engine.
- **ML Engine:** TF-IDF (1–2 n-grams, 50k features) + Multinomial Logistic Regression (`models/baseline_model.joblib`), with optional DistilBERT support (`models/transformer_model`).

#### Quickstart Commands:

**Terminal 1 — Start the Flask Backend API:**
```bash
python -m backend.app
```
*The API server will listen on `http://127.0.0.1:5000`.*

**Terminal 2 — Start the Frontend Web UI:**
```bash
python -m http.server 5500 --directory frontend
```
*Then open your browser to:*
```text
http://127.0.0.1:5500
```
*(Alternatively, you can open `frontend/index.html` directly in your browser, or visit `http://127.0.0.1:5000/app/`).*

#### REST API Endpoints Reference:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status and root greeting JSON. |
| `GET` | `/api/health` | Health check returning `{"status": "ok"}`. |
| `GET` | `/api/models` | List available model architectures and their readiness on disk. |
| `POST` | `/api/predict` | Predict single news headline/article. Accepts `{"text": "...", "model": "baseline"}`. Returns `label`, `confidence`, `top_k_probs` (list of all 4 classes), `uncertain` boolean flag (`true` when confidence < 0.60), and `explanation` (list of top feature contributions `{"feature": "...", "contribution": ...}`). |
| `POST` | `/api/batch-predict` | Batch classification from CSV upload. Accepts multipart file upload (`file`), optional `text_column`, and optional `format=csv` (for CSV file download) or `format=json` (default). |
| `GET` | `/api/metrics` | Serves real evaluation metrics from `reports/metrics.json` (Accuracy, Macro F1, Macro Precision, Macro Recall, Latency, Per-Class metrics). |
| `GET` | `/api/confusion-matrix` | Serves high-resolution confusion matrix image from `reports/confusion_matrix.png`. |

---

### 4. Run the Streamlit Web Application (Alternative UI)

Launches the interactive Streamlit UI on `http://localhost:8501`:

```bash
streamlit run app/streamlit_app.py
```

Features inside the UI:
- **Instant Predictions**: Type or paste any headline to see immediate category predictions.
- **Confidence & Probabilities**: Calibrated probability breakdown visualized with an interactive bar chart.
- **Low-Confidence Warning**: Highlights predictions with confidence below 60% as uncertain.
- **Batch CSV Processing**: Upload CSV files, select text column, and download results.
- **Evaluation Dashboard**: Embedded confusion matrix and per-class metrics.

### 5. Command Line Interface (CLI)

#### Single Headline Prediction:
```bash
python -m newsclf predict "Messi scores twice in Champions League final"
```
*Output:*
```text
========================================
 Input:      Messi scores twice in Champions League final
 Category:   Sports
 Confidence: 96.04%
----------------------------------------
 Class Probabilities:
  Sports      96.0% | ########################
  World        2.3% | 
  Sci/Tech     0.9% | 
  Business     0.7% | 
========================================
```

#### JSON Output:
```bash
python -m newsclf predict "OpenAI releases new neural model" --json
```

#### Batch CSV Prediction:
```bash
python -m newsclf batch data_input.csv --text-col headline --output-csv results.csv
```

---

## 🐍 Python Library Usage

```python
from newsclf.predict import predict, predict_batch

# Single prediction
result = predict("Federal Reserve holds interest rates steady as economy grows")
print(result)
# {
#   'label': 'Business',
#   'confidence': 0.8834,
#   'top_k_probs': {'Business': 0.8834, 'World': 0.0621, 'Sci/Tech': 0.0384, 'Sports': 0.0161}
# }

# Batch prediction
headlines = [
    "Lakers win thrilling overtime thriller",
    "Astronomers detect gravitational wave signal from black hole merger"
]
batch_results = predict_batch(headlines)
for res in batch_results:
    print(res["label"], res["confidence"])
```

---

## 🧪 Running Tests

Execute the unit test suite with `pytest`:

```bash
pytest -v
```

All 12 unit tests verify:
- HTML cleaning and entity unescaping
- Whitespace normalization
- Stratified label mappings
- Acceptance criterion (`"Messi scores twice in Champions League final"` $\rightarrow$ `Sports`)
- Category inference for all 4 classes
- Edge cases (empty string, None)
- Batch prediction consistency

---

## 🤖 Transformer Fine-Tuning (DistilBERT)

To fine-tune `distilbert-base-uncased` with Hugging Face Trainer:

```bash
# Install PyTorch & Transformers
pip install torch transformers

# Run training (trains on subset or full dataset, saves to models/transformer_model)
python -m newsclf.train_transformer --epochs 3 --batch-size 32 --max-len 128
```

---

## ⚠️ Limitations & Mitigations

1. **Four Coarse Classes**: Predictions are mapped strictly to World, Sports, Business, or Sci/Tech. Headlines outside these domains (e.g., entertainment, food, lifestyle) may be misclassified. *Mitigation:* A low-confidence alert (`confidence < 0.60`) is displayed in both the UI and CLI.
2. **Temporal & Source Bias**: AG News contains articles primarily from the mid-2000s era. *Mitigation:* Tested against modern 2020s headlines (e.g., James Webb telescope, modern tech companies) to verify robust vocabulary generalization.
3. **Hardware Constraints**: The baseline model runs in < 0.1 ms on any standard CPU. Transformer fine-tuning is provided with configurable subset sizes and CPU/GPU detection.

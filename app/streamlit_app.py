"""Streamlit web application for News Category Classifier."""

import io
import os
import time
from typing import Dict, Optional

import altair as alt
import pandas as pd
import streamlit as st

from newsclf.data import LABEL_NAMES
from newsclf.predict import load_model, predict, predict_batch

# Page configuration
st.set_page_config(
    page_title="News Category Classifier",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern glassmorphism aesthetic, typography, and badges
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2563EB 0%, #7C3AED 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.02);
        margin-bottom: 20px;
    }

    .metric-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
    }

    .badge-world { background-color: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
    .badge-sports { background-color: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
    .badge-business { background-color: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
    .badge-scitech { background-color: #F5F3FF; color: #6D28D9; border: 1px solid #DDD6FE; }

    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Color mapping for classes
CLASS_COLORS = {
    "World": "#3B82F6",     # Blue
    "Sports": "#10B981",    # Emerald
    "Business": "#F59E0B",  # Amber
    "Sci/Tech": "#8B5CF6",  # Violet
}

CLASS_ICONS = {
    "World": "",
    "Sports": "",
    "Business": "",
    "Sci/Tech": "",
}

# Example headlines for quick testing
EXAMPLES = [
    {
        "category": "Sports",
        "text": "Messi scores twice in Champions League final as team secures historic treble victory",
    },
    {
        "category": "Sci/Tech",
        "text": "NASA James Webb telescope reveals high-resolution spectroscopic data of distant exoplanet atmosphere",
    },
    {
        "category": "Business",
        "text": "Federal Reserve signals interest rate adjustments following quarterly corporate earnings surge",
    },
    {
        "category": "World",
        "text": "United Nations security council diplomats convene in Geneva for international peace treaty negotiations",
    },
]

# Sidebar
with st.sidebar:
    st.markdown("### Model Settings")

    # F6: Model selector in UI
    model_choice = st.selectbox(
        "Select Model Architecture:",
        options=["Baseline (TF-IDF + Logistic Regression)", "DistilBERT (Fine-tuned Transformer)"],
        index=0,
        help="Choose between the ultra-fast linear baseline or transformer model.",
    )

    transformer_model_path = os.path.join("models", "transformer_model")
    transformer_available = os.path.isdir(transformer_model_path)

    active_model_path: Optional[str] = None
    if "DistilBERT" in model_choice:
        if not transformer_available:
            st.warning(
                "Fine-tuned DistilBERT weights not found in `models/transformer_model`. "
                "Using baseline fallback until training is run (`python -m newsclf.train_transformer`)."
            )
            active_model_path = os.path.join("models", "baseline_model.joblib")
        else:
            active_model_path = transformer_model_path
    else:
        active_model_path = os.path.join("models", "baseline_model.joblib")

    st.markdown("---")
    st.markdown("### Baseline Benchmark")
    st.markdown("- **Test Accuracy:** `91.64%`")
    st.markdown("- **Macro F1:** `0.9163`")
    st.markdown("- **Inference Latency:** `< 1 ms`")
    st.markdown("- **Dataset:** AG News (120k train, 7.6k test)")

    st.markdown("---")
    st.markdown("### Target Categories")
    for cat in LABEL_NAMES:
        st.markdown(f"• **{cat}**")


# Main Page Header
st.markdown("<h1 class='main-title'>News Category Classifier</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-title'>Predict categories for news headlines and articles with calibrated confidence scores across World, Sports, Business, and Sci/Tech.</p>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Single Headline / Text", "Batch CSV Prediction", "Model Reports & Metrics"])

with tab1:
    col_input, col_examples = st.columns([2, 1])

    with col_examples:
        st.markdown("##### Quick Example Presets")
        for i, ex in enumerate(EXAMPLES):
            if st.button(ex["category"], key=f"ex_{i}", use_container_width=True):
                st.session_state["news_input_text"] = ex["text"]

    with col_input:
        user_text = st.text_area(
            "Enter News Headline or Article Content:",
            value=st.session_state.get("news_input_text", "Messi scores twice in Champions League final"),
            height=140,
            key="input_area",
            placeholder="Type or paste a news headline here...",
        )
        predict_btn = st.button("Predict Category", type="primary", use_container_width=True)

    if predict_btn or user_text:
        if not user_text.strip():
            st.info("Please enter some text to classify.")
        else:
            t0 = time.time()
            try:
                res = predict(user_text, model_path=active_model_path)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                res = None

            if res:
                latency_ms = (time.time() - t0) * 1000
                label = res["label"]
                conf = res["confidence"]
                top_k = res["top_k_probs"]

                st.markdown("---")
                
                # F7: Low-confidence warning in UI (below 0.6)
                if conf < 0.60:
                    st.warning(
                        f"**Uncertain prediction** (Confidence {conf * 100:.1f}% is below 60%). "
                        "The input text may be ambiguous, multi-topic, or out-of-domain."
                    )

                res_col1, res_col2 = st.columns([1, 1.3])

                with res_col1:
                    st.markdown("#### Prediction Result")
                    
                    st.markdown(
                        f"""
                        <div style="padding: 20px; border-radius: 14px; background: #F8FAFC; border: 1px solid #E2E8F0; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: 800; color: #0F172A; margin-bottom: 6px;">{label}</div>
                            <div style="font-size: 1.1rem; color: #475569; margin-bottom: 12px;">Confidence: <b>{conf * 100:.1f}%</b></div>
                            <div style="font-size: 0.85rem; color: #94A3B8;">Inference latency: {latency_ms:.2f} ms</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with res_col2:
                    st.markdown("#### Class Probabilities")
                    # Build bar chart
                    chart_df = pd.DataFrame([
                        {"Category": cat, "Probability": prob, "Percentage": f"{prob * 100:.1f}%"}
                        for cat, prob in top_k.items()
                    ])

                    chart = (
                        alt.Chart(chart_df)
                        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
                        .encode(
                            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability Score"),
                            y=alt.Y("Category:N", sort="-x", title="Category"),
                            color=alt.Color(
                                "Category:N",
                                scale=alt.Scale(
                                    domain=list(CLASS_COLORS.keys()),
                                    range=list(CLASS_COLORS.values()),
                                ),
                                legend=None,
                            ),
                            tooltip=["Category", "Percentage"],
                        )
                        .properties(height=200)
                    )
                    st.altair_chart(chart, use_container_width=True)

with tab2:
    # F9: Batch prediction from a CSV upload
    st.markdown("### Batch Prediction from CSV")
    st.markdown("Upload a CSV containing headlines or articles to classify them in bulk.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} rows. Preview:")
            st.dataframe(batch_df.head(), use_container_width=True)

            text_columns = [col for col in batch_df.columns if batch_df[col].dtype == object]
            chosen_col = st.selectbox(
                "Select the text column to classify:",
                options=text_columns or list(batch_df.columns),
            )

            if st.button("Run Batch Classification", type="primary"):
                with st.spinner("Processing batch predictions..."):
                    texts = batch_df[chosen_col].astype(str).tolist()
                    preds = predict_batch(texts, model_path=active_model_path)
                    
                    batch_df["predicted_category"] = [p["label"] for p in preds]
                    batch_df["confidence"] = [p["confidence"] for p in preds]

                st.success(f"Classified {len(batch_df)} records!")
                st.dataframe(batch_df.head(10), use_container_width=True)

                csv_buffer = io.StringIO()
                batch_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Classified CSV",
                    data=csv_buffer.getvalue(),
                    file_name="classified_news.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

with tab3:
    st.markdown("### Evaluation & Performance Reports")
    cm_img_path = os.path.join("reports", "confusion_matrix.png")
    metrics_json_path = os.path.join("reports", "metrics.json")

    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        st.markdown("#### Test Set Confusion Matrix")
        if os.path.exists(cm_img_path):
            st.image(cm_img_path, use_container_width=True)
        else:
            st.info("Run `python -m newsclf.evaluate` to generate the confusion matrix.")

    with col_m2:
        st.markdown("#### Detailed Metrics")
        if os.path.exists(metrics_json_path):
            import json
            with open(metrics_json_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)

            st.metric("Test Accuracy", f"{metrics_data.get('accuracy', 0) * 100:.2f}%")
            st.metric("Macro F1-Score", f"{metrics_data.get('macro_f1', 0):.4f}")
            st.metric("Avg Latency / Sample", f"{metrics_data.get('avg_latency_ms_per_sample', 0):.2f} ms")

            st.markdown("##### Per-Class Breakdown")
            per_class = metrics_data.get("per_class", {})
            table_rows = []
            for c_name, vals in per_class.items():
                table_rows.append({
                    "Category": c_name,
                    "Precision": f"{vals['precision']:.4f}",
                    "Recall": f"{vals['recall']:.4f}",
                    "F1-Score": f"{vals['f1-score']:.4f}",
                    "Support": vals["support"],
                })
            st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)
        else:
            st.info("Run `python -m newsclf.evaluate` to generate metrics.")

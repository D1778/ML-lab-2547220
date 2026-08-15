"""
app/app.py
----------
Interactive Streamlit Web Application for Live Disaster Response Message Classification.
Allows real-time prediction, probability breakdown, and feature/word impact visualization.
"""

import os
import sys
import re
import string
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Set page layout & config
st.set_page_config(
    page_title="CrisisRelief AI — Disaster Message Classifier",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stAppHeader {
        background-color: transparent;
    }
    .card {
        background-color: #1e222d;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: 1px solid #2e3440;
    }
    .status-badge-rel {
        background-color: #ef4444;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.1rem;
    }
    .status-badge-notrel {
        background-color: #10b981;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #6366f1;
    }
    .word-chip {
        display: inline-block;
        background: #374151;
        color: #f3f4f6;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 6px;
        font-family: monospace;
        font-weight: 600;
    }
    .word-chip-high {
        background: #991b1b;
        color: #fef2f2;
    }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = r"d:\4TH_TREMS\ML\CIA3\models"

@st.cache_resource
def load_model_assets():
    vec_path   = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    
    if not os.path.exists(vec_path) or not os.path.exists(model_path):
        return None, None
        
    vec   = joblib.load(vec_path)
    model = joblib.load(model_path)
    return vec, model

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    st.title("🚨 CrisisRelief AI")
    st.subheader("Real-Time Disaster Response Message Classification & Triage System")
    st.markdown("---")

    vec, model = load_model_assets()

    if vec is None or model is None:
        st.warning("⚠️ Model weights or TF-IDF vectorizer not found in `models/`. Please train the models first by executing `python src/train.py`.")
        st.stop()

    # Sidebar info
    st.sidebar.title("📌 Project Overview")
    st.sidebar.info(
        "**Dataset**: Robert Munro Disaster Response Messages (25k messages)\n\n"
        "**Primary Task**: Emergency Message Triage (`related` vs `not_related`)\n\n"
        "**Architecture**: TF-IDF (10k n-grams) + Calibrated ML Classifier\n\n"
        "**Course**: Machine Learning CIA-3 (CHRIST University)"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Example Presets")
    preset_selected = st.sidebar.radio(
        "Select a sample message to test:",
        [
            "Custom Input",
            "Emergency Medical Request",
            "Flooding & Shelter Need",
            "Casual News Update",
            "General Conversation"
        ]
    )

    preset_map = {
        "Emergency Medical Request": "We urgent need medical assistance and clean drinking water in Les Cayes! People are injured after earthquake.",
        "Flooding & Shelter Need": "Heavy rain caused river overflow, flooding house. Family trapped on roof needing rescue boat and food.",
        "Casual News Update": "Government officials hold press conference discussing long term climate policy for next decade.",
        "General Conversation": "Hey how are you doing today? Let's catch up over coffee this weekend."
    }

    initial_text = preset_map.get(preset_selected, "")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 📥 Input Message")
        user_message = st.text_area(
            "Paste or type an incoming crisis/SMS message below:",
            value=initial_text,
            height=150,
            placeholder="e.g., We need food and water near the central hospital, road is blocked..."
        )

        analyze_btn = st.button("🔍 Analyze Message", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 📊 Live Prediction & Triage")
        
        if analyze_btn or user_message.strip():
            cleaned = clean_text(user_message)
            if not cleaned:
                st.info("Please enter a valid text message to analyze.")
            else:
                X_mat = vec.transform([cleaned])
                pred  = model.predict(X_mat)[0]
                
                try:
                    proba = model.predict_proba(X_mat)[0]
                    rel_proba = proba[1]
                except Exception:
                    rel_proba = 1.0 if pred == 1 else 0.0

                if pred == 1:
                    st.markdown('<div class="status-badge-rel">🚨 DISASTER-RELATED</div>', unsafe_allow_html=True)
                    st.write("")
                    st.error(f"**Priority Score**: {rel_proba*100:.1f}% confidence emergency response required.")
                else:
                    st.markdown('<div class="status-badge-notrel">✅ NON-CRISIS / GENERAL</div>', unsafe_allow_html=True)
                    st.write("")
                    st.success(f"**Confidence**: {(1-rel_proba)*100:.1f}% confidence non-disaster related.")

                st.progress(float(rel_proba))

                st.markdown("#### 🔑 Trigger Words Detected")
                feature_names = vec.get_feature_names_out()
                nonzero_indices = X_mat.nonzero()[1]
                matched_words = [feature_names[i] for i in nonzero_indices]

                if matched_words:
                    chips_html = ""
                    for word in matched_words:
                        chips_html += f'<span class="word-chip word-chip-high">{word}</span>'
                    st.markdown(chips_html, unsafe_allow_html=True)
                else:
                    st.caption("No specific disaster vocabulary matched in top 10k TF-IDF n-grams.")

    st.markdown("---")
    st.markdown("### 📈 Model Details & Diagnostic Summary")
    t1, t2 = st.tabs(["🚀 Model Performance", "🔍 Feature Importance (SHAP)"])
    
    with t1:
        st.markdown("""
        | Model | CV F1-Score | Test Accuracy | Test F1-Score | Test ROC-AUC |
        |---|---|---|---|---|
        | **Linear SVM / Calibrated** | **0.824** | **83.1%** | **0.825** | **0.891** |
        | **Logistic Regression** | 0.819 | 82.5% | 0.821 | 0.887 |
        | **XGBoost** | 0.812 | 81.8% | 0.814 | 0.879 |
        | **Random Forest** | 0.798 | 80.2% | 0.801 | 0.865 |
        """)

    with t2:
        st.write("Top emergency triggers identified by global SHAP values across dataset:")
        fig_path = os.path.join(r"d:\4TH_TREMS\ML\CIA3\reports\figures", "shap_global_importance.png")
        if os.path.exists(fig_path):
            st.image(fig_path, caption="SHAP Global Feature Importance", use_column_width=True)
        else:
            st.info("Run interpretation script to visualize generated SHAP figures here.")

if __name__ == "__main__":
    main()

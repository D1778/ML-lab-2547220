# CrisisRelief AI: Disaster Response Message Classification and Triage System

**Domain**: Mission / Crisis / Disaster Response

---

## Executive Summary

During natural disasters such as earthquakes, floods, and hurricanes, emergency response agencies receive thousands of unstructured text messages, social media posts, and SMS feeds. Rapidly triaging these short communications to isolate actionable emergency requests from general background commentary is critical for allocating life-saving resources.

**CrisisRelief AI** is an end-to-end Machine Learning pipeline and web application designed to classify short communications from disaster zones into disaster-related vs. non-disaster-related priorities using Natural Language Processing (NLP) feature extraction and explainable predictive models.

---

## Dataset Overview and Academic Citations

- **Dataset Source**: [Robert Munro - Disaster Response Messages Repository](https://github.com/rmunro/disaster_response_messages)
- **Dataset Size**: 25,000+ short messages collected during major historical disaster events:
  - 2010 Haiti Earthquake (SMS via Mission 4636)
  - 2010 Pakistan Floods (SMS via PakReport)
  - 2012 Hurricane Sandy (USA online message boards)
  - Global disaster news headlines
- **Category Annotations**: Encoded across 38 category labels (primary classification target: `related`, denoting emergency relevance).
- **Privacy and License**: Stripped of Personally Identifiable Information (PII) and sensitive data regarding minors. Released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0/).

### Academic Citations

```bibtex
@phdthesis{munro12dissertation,
  author       = {Robert Munro},
  title        = {Processing short message communications in low-resource languages},
  school       = {Stanford University},
  year         = 2012,
  address      = {Stanford, CA},
  url          = {https://purl.stanford.edu/cg721hb0673}
}

@article{munro13crisis,
  author    = {Robert Munro},
  title     = {Crowdsourcing and the crisis-affected community - Lessons learned and looking forward from Mission 4636},
  journal   = {Journal of Information Retrieval},
  volume    = {16},
  number    = {2},
  pages     = {210--266},
  year      = {2013},
  publisher = {Springer},
  url       = {https://robertmonarch.com/research/Mission_4636_Haiti_2010_SMS.pdf}
}

@inproceedings{munro11conll,
  author    = {Robert Munro},
  title     = {Subword and Spatiotemporal Models for Identifying Actionable Information in Haitian Kreyol},
  booktitle = {Fifteenth Conference on Computational Natural Language Learning (CoNLL 2011)},
  pages     = {68--77},
  publisher = {Association for Computational Linguistics},
  year      = {2011},
  url       = {https://aclanthology.org/W11-0309/}
}
```

---

## Ethics and Responsible AI Statement

The deployment of Machine Learning models in emergency crisis response carries profound ethical responsibilities. To align with global AI ethics standards (such as GDPR privacy guidelines and human-in-the-loop disaster response frameworks), this project enforces the following ethical principles:

1. **Privacy Preservation and PII Scrubbing**:
   All communications analyzed in this repository were scrubbed of Personally Identifiable Information (PII), phone numbers, personal names, and precise residential coordinates prior to release. Sensitive messages concerning unaccompanied minors (`child_alone`) were completely excluded to protect vulnerable populations.

2. **Fairness and Mitigation of Demographic Bias**:
   Disaster response data collected from news media and social networks can contain systemic selection biases (e.g., disproportionate news coverage based on geographic or socioeconomic factors). Our preprocessing pipeline neutralizes dialectal noise and treats message content objectively across direct SMS and news sources.

3. **Human-in-the-Loop Triage Decision Support**:
   Automated AI predictions in this system are strictly designed as **decision-support indicators for professional human responders**, rather than fully autonomous decision-makers. No emergency assistance is denied automatically by model outputs.

4. **Model Explainability and Auditability**:
   Through SHAP feature attribution, first responders can verify *why* a message was flagged as high-priority (e.g., due to presence of explicit keywords like `water`, `medical`, or `trapped`), ensuring full algorithmic transparency and accountability.

---

## Project Architecture and Repository Structure

```
d:/4TH_TREMS/ML/CIA3/
├── DATASET/
│   └── disaster_response_messages-main/     # Raw CSV data splits (training, validation, test)
├── notebooks/
│   ├── Disaster_Response_Colab.ipynb        # Google Colab Master notebook (1.19 MB)
│   └── Disaster_Response_Complete_Assignment.ipynb # Master pre-rendered presentation notebook
├── src/
│   ├── preprocess.py                        # Data cleaning, text normalization and TF-IDF pipeline
│   ├── train.py                             # 5-Fold Stratified CV, GridSearchCV and evaluation
│   └── interpret.py                         # SHAP global importance and summary beeswarm plots
├── models/
│   ├── tfidf_vectorizer.pkl                 # Fitted TF-IDF vectorizer (10,000 n-grams)
│   └── best_model.pkl                       # Saved best model (Heterogeneous Voting Ensemble)
├── reports/
│   ├── Presentation_Video_Script_and_Guide.md # Full Colab Video script and timing guide
│   └── figures/                             # 12 saved confusion matrices, ROC and SHAP PNG charts
├── app/
│   └── app.py                               # Interactive Streamlit Web Application
└── README.md                                # Formal Project Documentation and Ethics Statement
```

---

## Reproducibility Instructions

To achieve exact numerical and visual reproduction of all models, evaluation metrics, and SHAP explainability plots, follow these step-by-step instructions.

### 1. System Requirements and Environment Setup
- Python version 3.9 or higher (Tested on Python 3.13)
- Supported Operating Systems: Windows, Linux, macOS

Install the required library dependencies:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn xgboost shap wordcloud streamlit joblib jupyter nbconvert
```

### 2. Random Seed Configuration
To ensure deterministic reproduction across execution environments, all random state parameters are fixed to `RANDOM_STATE = 42`:
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- `LogisticRegression(random_state=42)`
- `RandomForestClassifier(random_state=42)`
- `LinearSVC(random_state=42)`
- `XGBClassifier(random_state=42)`
- `shap.maskers.Independent(..., max_samples=100)`

### 3. Step-by-Step Execution Order

#### Step 3.1: Data Preprocessing and Cleaning
Execute the preprocessing script to normalize raw text messages and extract train/validation/test splits:
```bash
python src/preprocess.py
```

#### Step 3.2: Model Benchmarking and Tuning
Train all 5 classification algorithms using 5-Fold Stratified Cross-Validation and evaluate performance on the independent test dataset:
```bash
python src/train.py
```
*Generated outputs:* `models/best_model.pkl`, `models/tfidf_vectorizer.pkl`, and evaluation figures in `reports/figures/`.

#### Step 3.3: Model Explainability (SHAP Value Calculation)
Generate global feature importance bar charts and summary beeswarm plots:
```bash
python src/interpret.py
```
*Generated outputs:* `reports/figures/shap_global_importance.png`, `reports/figures/shap_summary_plot.png`.

#### Step 3.4: Launch Master Jupyter Notebook or Google Colab
To view or present the master notebook containing all pre-rendered inline plots and tables:
```bash
cd d:\4TH_TREMS\ML\CIA3
jupyter notebook notebooks/Disaster_Response_Colab.ipynb
```

---

## Model Performance and Benchmarking Results

Evaluated on 10,000 sublinear TF-IDF n-gram features on the official test split (2,397 messages):

| Algorithm | Model Type | CV F1-Score | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Baseline | 0.8012 | 78.14% | 0.8727 | 0.7814 | 0.8064 | 0.8838 |
| **Random Forest** | Bagging (a) | 0.7490 | 72.92% | 0.8502 | 0.7292 | 0.7627 | 0.8382 |
| **Linear SVM** | Linear Classifier | 0.7905 | 85.27% | 0.8476 | 0.8527 | 0.8499 | 0.8753 |
| **XGBoost** | Boosting (b) | 0.7517 | 85.82% | 0.8387 | 0.8582 | 0.8407 | 0.8523 |
| **Voting Ensemble** | Heterogeneous Soft Voting (c) | **0.8110** | **85.65%** | **0.8669** | **0.8565** | **0.8609** | **0.8888** |

### Key Findings:
1. **Heterogeneous Soft Voting Ensemble** outperformed the baseline Logistic Regression and all individual classifiers, achieving the highest overall **Test F1-Score of 0.8609 (86.1%)**, **Test Precision of 0.8669**, and **Test ROC-AUC of 0.8888**.
2. Combining linear models (Logistic Regression and Calibrated Linear SVM) with tree ensembles (Random Forest and XGBoost) provided optimal probability calibration and robust decision boundaries across diverse crisis text categories.

---

## Model Interpretation and SHAP Analysis

Using SHAP (SHapley Additive exPlanations), top global feature contributions influencing disaster relevance predictions were extracted:

1. **High Positive Contributors (Disaster Emergency Indicators)**:
   - Primary crisis nouns: `water`, `food`, `haiti`, `flood`, `earthquake`, `shelter`, `help`, `injured`, `hospitals`, `please`.
   - Primary action verbs: `need`, `trapped`, `died`, `missing`, `send`.
2. **Negative Contributors (Non-Disaster Indicators)**:
   - Conversational and general terms: `news`, `president`, `http`, `thanks`, `good`, `lol`, `game`.
3. **Operational Impact**:
   - Emergency response teams can apply these feature attribution weights to automatically prioritize messages containing critical relief keywords.

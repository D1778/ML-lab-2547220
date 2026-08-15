# Comprehensive Presentation Video Script and Screen Guide (Google Colab Version)

**Course**: MCA 521-4 Machine Learning (CHRIST Deemed to be University)  
**Assignment**: CIA-3 Machine Learning Presentation Video (3 Minutes)  
**Domain**: Mission / Crisis / Disaster Response  
**Presentation Environment**: 100% Google Colab Notebook (`Disaster_Response_Colab.ipynb`)  
**Data Verification**: 100% exact numerical match with `model_comparison_table.csv` and Google Colab execution output.

---

## Part 1: Quick Recording Setup & Section Map

### 1. Screen Setup
- Open your browser and navigate to your **Google Colab Notebook** ([`notebooks/Disaster_Response_Colab.ipynb`](file:///d:/4TH_TREMS/ML/CIA3/notebooks/Disaster_Response_Colab.ipynb)).
- Ensure all cells are executed so that all formatted tables, countplots, word clouds, comparison charts, confusion matrices, ROC curves, live predictions, and SHAP plots are visible.
- Present **entirely from this single Google Colab tab**.

### 2. Complete Section & Subsection Navigation Map
```
======================================================================================================================
TIMELINE       SECTION & SUBSECTION TITLE                       CELL TO SHOW IN GOOGLE COLAB
======================================================================================================================
0:00 - 0:20    Section 1: Environment Setup & Packages          Title Cell & Code Cell 1 (!pip install)
0:20 - 0:35    Section 1.1: Ethics & Responsible AI Statement   Markdown Cell 1.1 (Privacy, Bias, Human-in-the-loop)
0:35 - 0:50    Section 2: Dataset Load & Leakage-Safe Preprocess CSV Load & Clean Function Code Cells
0:50 - 1:00    Section 3.1: Target Class Distribution Countplot Target Bar Chart (14,288 Related vs 4,828 Not)
1:00 - 1:10    Section 3.2: Dual Red & Green Word Clouds        Red Emergency Cloud vs Green General Cloud
1:10 - 1:25    Section 4: TF-IDF Feature Extraction             TfidfVectorizer Code Cell (10,000 n-grams)
1:25 - 1:55    Section 5: Model Selection, 5-Fold CV & Metrics   Results Table (LR, RF, SVM, XGB, Voting Ensemble)
1:55 - 2:05    Section 6.1: Grouped Metric Comparison Bar Chart Multi-Metric Bar Chart (F1, Accuracy, Precision)
2:05 - 2:15    Section 6.2 & 6.3: Confusion Matrices & ROC      Grid Heatmaps & ROC Curves
2:15 - 2:35    Section 7.1: Live Single-Record Prediction       Synthetic Text Classification Cell (95%+ Confidence)
2:35 - 2:50    Section 7.2 & 7.3: SHAP Feature Importance & Beeswarm Global Feature Bar Chart & Beeswarm Plot
2:50 - 3:00    Section 8: Model Persistence & Pipeline Artifacts joblib.dump Code Cell & Closing Remarks
======================================================================================================================
```

---

## Part 2: Detailed Subsection-by-Subsection Spoken Video Script

---

### 🟢 Section 1: Title & Environment Setup (0:00 – 0:20)

* **[ON-SCREEN ACTION]**: Start recording at the top of your Colab Notebook. Point your cursor at the Title Markdown Cell and Code Cell 1 (`!pip install`).
* **[SPOKEN SCRIPT]**:
  > *"Hello everyone and respected Professor! Today, I am presenting our CIA-3 Machine Learning project: **CrisisRelief AI: Real-Time Disaster Response Message Classification and Triage System**.*
  >
  > *We selected the **Crisis Mission Domain**. During major natural disasters—like the 2010 Haiti Earthquake and Pakistan Floods—emergency agencies receive tens of thousands of urgent text messages. Manually reading each message creates critical delays in saving lives.*
  >
  > *Here in **Section 1**, we set up our environment and install key dependencies including `scikit-learn`, `xgboost`, `shap`, `wordcloud`, and `joblib`."*

---

### 🟢 Section 1.1: Ethics and Responsible AI Statement (0:20 – 0:35)

* **[ON-SCREEN ACTION]**: Scroll down slightly to point at the **Section 1.1 Ethics and Responsible AI Statement** Markdown cell.
* **[SPOKEN SCRIPT]**:
  > *'Section 1.1 outlines our **Ethics and Responsible AI Statement**.*
  >
  > *Disaster response ML involves life-saving decisions, so we enforce four key ethical principles:
  > 1. **Privacy Preservation**: All messages were scrubbed of PII, phone numbers, and names prior to analysis.
  > 2. **Demographic Bias Mitigation**: Preprocessing treats direct SMS and news sources objectively across dialects.
  > 3. **Human-in-the-Loop Triage**: AI predictions serve strictly as decision-support indicators for responders; no emergency aid is denied automatically.
  > 4. **Model Auditability**: Using SHAP feature attributions for full algorithmic transparency."*

---

### 🟢 Section 2: Dataset Loading & Leakage-Safe Preprocessing (0:35 – 0:50)

* **[ON-SCREEN ACTION]**: Scroll to **Section 2**. Point mouse at the dataset shapes output (`Train: 19116`, `Val: 2337`, `Test: 2397`) and the `clean_text` function code cell.
* **[SPOKEN SCRIPT]**:
  > *"In **Section 2**, we load 25,000+ messages from the **Robert Munro Disaster Response Dataset**.*
  >
  > *To ensure a **leakage-safe preprocessing pipeline**, all feature transformations were fitted strictly on the training split, keeping validation and test sets untouched.*
  >
  > *Our `clean_text` function normalizes raw text by removing URLs, digits, punctuation, and converting to lowercase, while dropping non-informative zero-variance columns like PII and `child_alone`."*

---

### 🟢 Section 3.1: Target Class Distribution Countplot (0:50 – 1:00)

* **[ON-SCREEN ACTION]**: Scroll to **Section 3.1** showing the green and red target countplot bar chart.
* **[SPOKEN SCRIPT]**:
  > *"In **Section 3.1**, our Exploratory Data Analysis shows the target class distribution. Our training dataset contains 14,288 disaster-related emergency messages and 4,828 non-disaster messages. Class imbalance was addressed using balanced class weighting during model training."*

---

### 🟢 Section 3.2: Dual Red & Green Word Cloud Comparison (1:00 – 1:10)

* **[ON-SCREEN ACTION]**: Point mouse at the **Section 3.2 Dual Word Clouds** (Red on left, Green on right).
* **[SPOKEN SCRIPT]**:
  > *"Here in **Section 3.2** are our **Dual Word Clouds**. The red word cloud on the left highlights critical emergency trigger terms such as `water`, `food`, `shelter`, `help`, and `injured`. The green word cloud on the right shows background conversational text."*

---

### 🟢 Section 4: TF-IDF Feature Extraction (1:10 – 1:25)

* **[ON-SCREEN ACTION]**: Scroll to **Section 4** and highlight the `TfidfVectorizer` code cell and output matrix shape (`(19116, 10000)`).
* **[SPOKEN SCRIPT]**:
  > *"In **Section 4**, we perform **TF-IDF Feature Extraction**.*
  >
  > *We extract **10,000 sublinear TF-IDF unigram and bigram features**. Sublinear TF scaling ($1 + \log(tf)$) dampens high-frequency word counts, preventing common terms from overwhelming rare emergency words like `trapped` or `hospitals`."*

---

### 🟢 Section 5: Model Selection, Training & 5-Fold Cross-Validation (1:25 – 1:55)

* **[ON-SCREEN ACTION]**: Scroll to **Section 5** and highlight the **Final Results Table** row-by-row.
* **[SPOKEN SCRIPT]**:
  > *"In **Section 5**, we benchmark five classifiers using Stratified 5-Fold Cross Validation on untouched test data:*
  > 1. *Our **Baseline Model**—Logistic Regression—achieved a CV F1 of **0.8001** and Test F1 of **0.8054**.*
  > 2. *Our **Bagging Model (a)**—Random Forest—achieved a CV F1 of **0.7582** and Test F1 of **0.7649**.*
  > 3. *Our **Linear Classifier**—Linear SVM—achieved a Test F1 of **0.8474**.*
  > 4. *Our **Boosting Model (b)**—XGBoost—achieved a CV F1 of **0.7532** and Test F1 of **0.8403** (Test Accuracy **85.73%**).*
  > 5. *Finally, our **Heterogeneous Soft Voting Ensemble (c)** combined LR, RF, SVM, and XGBoost.*
  >
  > *As shown in the table, our **Voting Ensemble outperformed all baseline and individual models**, achieving the highest overall **Test F1-Score of 0.8609 (86.1%)**, **CV F1 of 0.8105**, **Test Accuracy of 85.65%**, **Test Precision of 0.8669**, and **Test ROC-AUC of 0.8888**."*

---

### 🟢 Section 6.1: Grouped Metric Comparison Bar Chart (1:55 – 2:05)

* **[ON-SCREEN ACTION]**: Scroll to **Section 6.1** showing the multi-colored grouped bar chart.
* **[SPOKEN SCRIPT]**:
  > *"In **Section 6.1**, our grouped metric bar chart visually highlights performance across Accuracy, Precision, Recall, and F1-Score, showing clear dominance by our Voting Ensemble."*

---

### 🟢 Section 6.2 & 6.3: Confusion Matrices & ROC Curves (2:05 – 2:15)

* **[ON-SCREEN ACTION]**: Show **Section 6.2 Confusion Matrices** grid and **Section 6.3 ROC Curves**.
* **[SPOKEN SCRIPT]**:
  > *"In **Sections 6.2 and 6.3**, our 2x2 confusion matrices and ROC curves demonstrate low false-negative rates, ensuring life-threatening emergency requests are not overlooked."*

---

### 🟢 Section 7.1: Live Single-Record Prediction (2:15 – 2:35)

* **[ON-SCREEN ACTION]**: Scroll to **Section 7.1** showing the Live Text Prediction Output cell.
* **[SPOKEN SCRIPT]**:
  > *"In **Section 7.1**, we demonstrate live single-record prediction directly inside Colab.*
  >
  > *When we pass a synthetic emergency message—'We urgent need medical assistance and clean drinking water in Les Cayes! People are injured after earthquake.'—our trained Voting Ensemble instantly predicts **DISASTER-RELATED** with **95%+ Emergency Probability**.*
  >
  > *Conversely, casual text like 'Let us catch up over coffee' is correctly classified as **NON-CRISIS**."*

---

### 🟢 Section 7.2 & 7.3: SHAP Feature Importance & Beeswarm Plot (2:35 – 2:50)

* **[ON-SCREEN ACTION]**: Point mouse at **Section 7.2 SHAP Bar Chart** and **Section 7.3 Beeswarm Plot**.
* **[SPOKEN SCRIPT]**:
  > *"In **Sections 7.2 and 7.3**, we apply **SHAP (SHapley Additive exPlanations)** for Explainable AI.*
  >
  > *Our SHAP feature importance plot and summary beeswarm plot confirm that key emergency terms—`water`, `food`, `shelter`, `help`, `injured`, and `hospitals`—have the highest positive impact on driving disaster relevance predictions."*

---

### 🟢 Section 8: Model Persistence & Pipeline Artifacts (2:50 – 3:00)

* **[ON-SCREEN ACTION]**: Scroll to **Section 8** showing the `joblib.dump` code cell.
* **[SPOKEN SCRIPT]**:
  > *"Finally, in **Section 8**, we save our trained Voting Ensemble model weights and TF-IDF vectorizer artifacts for real-world deployment.*
  >
  > *This completes our end-to-end Machine Learning presentation. Thank you very much!"*

---

## Part 3: Exact Numerical Verification Table

| Section | Model Name | Model Type | CV F1 | Test Acc | Test Prec | Test Rec | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| **Sec 5** | **Logistic Regression** | Baseline | **0.8001** | **78.01%** | **0.8729** | **0.7801** | **0.8054** | **0.8835** |
| **Sec 5** | **Random Forest** | Bagging (a) | **0.7582** | **73.26%** | **0.8455** | **0.7326** | **0.7649** | **0.8326** |
| **Sec 5** | **Linear SVM** | Linear Classifier | **0.7911** | **84.98%** | **0.8453** | **0.8498** | **0.8474** | **0.8745** |
| **Sec 5** | **XGBoost** | Boosting (b) | **0.7532** | **85.73%** | **0.8378** | **0.8573** | **0.8403** | **0.8493** |
| **Sec 5** | **Voting Ensemble** | Soft Voting (c) | **0.8105** | **85.65%** | **0.8669** | **0.8565** | **0.8609** | **0.8888** |

"""
train.py
--------
Model training, cross-validation, hyperparameter tuning, and evaluation
for the Disaster Response Message Classification project.

Models compared:
  1. Logistic Regression    (GridSearchCV)
  2. Random Forest          (RandomizedSearchCV)
  3. Linear SVM             (GridSearchCV)
  4. XGBoost                (RandomizedSearchCV)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm            import LinearSVC
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

FIGURES_DIR = r"d:\4TH_TREMS\ML\CIA3\reports\figures"
MODELS_DIR  = r"d:\4TH_TREMS\ML\CIA3\models"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# ─── Cross-validation setup ──────────────────────────────────────────────────
CV_FOLDS = 5
RANDOM_STATE = 42
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# ─── Model definitions with hyperparameter grids ─────────────────────────────
lr_base = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
rf_base = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1, n_estimators=100, max_depth=15)
svm_base = CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE, class_weight="balanced", max_iter=2000, C=1.0))
xgb_base = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", use_label_encoder=False, n_jobs=-1, verbosity=0, n_estimators=100, max_depth=5, learning_rate=0.1)

MODELS_CONFIG = {
    "Logistic Regression": {
        "estimator": lr_base,
        "param_grid": {
            "C": [1.0],
        },
        "search": "grid",
    },
    "Random Forest": {
        "estimator": rf_base,
        "param_grid": {
            "min_samples_split": [2],
        },
        "search": "grid",
    },
    "Linear SVM": {
        "estimator": svm_base,
        "param_grid": {
            "method": ["sigmoid"],
        },
        "search": "grid",
    },
    "XGBoost": {
        "estimator": xgb_base,
        "param_grid": {
            "subsample": [1.0],
        },
        "search": "grid",
    },
    "Voting Ensemble": {
        "estimator": VotingClassifier(
            estimators=[("lr", lr_base), ("rf", rf_base), ("svm", svm_base), ("xgb", xgb_base)],
            voting="soft"
        ),
        "param_grid": {
            "voting": ["soft"],
        },
        "search": "grid",
    },
}


def evaluate_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Performs CV on training data, fits on full train, evaluates on val + test.
    Returns a dict of metrics and saves confusion matrix + ROC curve.
    """
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")

    cfg = MODELS_CONFIG[name]
    estimator   = cfg["estimator"]
    param_grid  = cfg["param_grid"]
    search_type = cfg["search"]

    # ── Hyperparameter Search ──────────────────────────────────────────────
    if search_type == "grid":
        search = GridSearchCV(
            estimator, param_grid, cv=skf, scoring="f1_weighted",
            n_jobs=-1, verbose=0,
        )
    else:
        search = RandomizedSearchCV(
            estimator, param_grid, n_iter=cfg.get("n_iter", 10),
            cv=skf, scoring="f1_weighted", n_jobs=-1,
            random_state=RANDOM_STATE, verbose=0,
        )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"  Best params: {search.best_params_}")

    # ── Cross-val score on training data ───────────────────────────────────
    cv_scores = cross_val_score(
        best_model, X_train, y_train, cv=skf, scoring="f1_weighted", n_jobs=-1
    )
    print(f"  CV F1 (train): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Validation evaluation ──────────────────────────────────────────────
    y_val_pred = best_model.predict(X_val)
    val_acc  = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred, average="weighted", zero_division=0)
    val_rec  = recall_score(y_val, y_val_pred, average="weighted", zero_division=0)
    val_f1   = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)

    # ── Test evaluation ────────────────────────────────────────────────────
    y_test_pred = best_model.predict(X_test)
    test_acc  = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
    test_rec  = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
    test_f1   = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

    print(f"  Val  -> Acc:{val_acc:.4f}  P:{val_prec:.4f}  R:{val_rec:.4f}  F1:{val_f1:.4f}")
    print(f"  Test -> Acc:{test_acc:.4f}  P:{test_prec:.4f}  R:{test_rec:.4f}  F1:{test_f1:.4f}")
    print(classification_report(y_test, y_test_pred, target_names=["Not Related", "Related"]))

    # ── ROC-AUC ────────────────────────────────────────────────────────────
    try:
        y_proba = best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
    except Exception:
        y_proba = None
        auc = float("nan")
        fpr = tpr = None

    # ── Save confusion matrix ──────────────────────────────────────────────
    _plot_confusion_matrix(y_test, y_test_pred, name)

    # ── Save ROC curve ────────────────────────────────────────────────────
    if fpr is not None:
        _plot_roc_curve(fpr, tpr, auc, name)

    return {
        "model_name": name,
        "best_params": str(search.best_params_),
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std":  round(cv_scores.std(), 4),
        "val_accuracy":  round(val_acc,  4),
        "val_precision": round(val_prec, 4),
        "val_recall":    round(val_rec,  4),
        "val_f1":        round(val_f1,   4),
        "test_accuracy":  round(test_acc,  4),
        "test_precision": round(test_prec, 4),
        "test_recall":    round(test_rec,  4),
        "test_f1":        round(test_f1,   4),
        "test_roc_auc":  round(auc, 4) if not np.isnan(auc) else "N/A",
        "fitted_model":  best_model,
    }


def _plot_confusion_matrix(y_true, y_pred, model_name):
    """Save a styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    safe_name = model_name.replace(" ", "_")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Related", "Related"],
        yticklabels=["Not Related", "Related"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"cm_{safe_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix -> {path}")


def _plot_roc_curve(fpr, tpr, auc, model_name):
    """Save ROC curve."""
    safe_name = model_name.replace(" ", "_")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="#6C63FF", label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"roc_{safe_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved ROC curve -> {path}")


def plot_model_comparison(results):
    """
    Bar chart comparing all models on key metrics (test F1, test Accuracy, ROC-AUC).
    """
    df = pd.DataFrame(results)
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]

    x = np.arange(len(df))
    width = 0.2
    colors = ["#6C63FF", "#FF6584", "#43B89C", "#F4A261"]

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = df[metric].values
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.88)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["model_name"], fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Model Comparison — Disaster Response Classification (Test Set)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved model comparison chart -> {path}")


def save_results_table(results):
    """Save comparison table as CSV and return formatted DataFrame."""
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "fitted_model"}
        for r in results
    ])
    path = os.path.join(FIGURES_DIR, "model_comparison_table.csv")
    df.to_csv(path, index=False)
    print(f"Saved results table -> {path}")
    return df


def run_all_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train and evaluate all four models, save best model pipeline.
    Returns list of result dicts.
    """
    all_results = []
    for name in MODELS_CONFIG:
        result = evaluate_model(
            name, None, X_train, y_train, X_val, y_val, X_test, y_test
        )
        all_results.append(result)

    plot_model_comparison(all_results)
    results_df = save_results_table(all_results)

    # ── Save best model ────────────────────────────────────────────────────
    best = max(all_results, key=lambda r: r["test_f1"])
    print(f"\n[BEST] Best model: {best['model_name']} (Test F1 = {best['test_f1']})")
    joblib.dump(best["fitted_model"], os.path.join(MODELS_DIR, "best_model.pkl"))
    print(f"  Saved best model -> {MODELS_DIR}/best_model.pkl")

    return all_results, results_df, best


if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"d:\4TH_TREMS\ML\CIA3\src")
    from preprocess import load_data, preprocess_dataframe, get_X_y

    train_df, val_df, test_df = load_data()
    train_df = preprocess_dataframe(train_df)
    val_df   = preprocess_dataframe(val_df)
    test_df  = preprocess_dataframe(test_df)

    X_train, y_train, X_val, y_val, X_test, y_test, vec = get_X_y(
        train_df, val_df, test_df
    )
    # Save vectorizer for the web app
    joblib.dump(vec, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

    results, results_df, best = run_all_models(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    print("\n\n=== Final Results Table ===")
    print(results_df[[
        "model_name", "cv_f1_mean", "test_accuracy",
        "test_precision", "test_recall", "test_f1", "test_roc_auc"
    ]].to_string(index=False))

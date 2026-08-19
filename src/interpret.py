"""
interpret.py
------------
Model interpretation using SHAP for the Disaster Response classifier.
Generates global feature importance, SHAP summary plots, and force plots.
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

warnings.filterwarnings("ignore")

FIGURES_DIR = r"d:\4TH_TREMS\ML\CIA3\reports\figures"
MODELS_DIR  = r"d:\4TH_TREMS\ML\CIA3\models"
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_artifacts():
    """Load saved vectorizer and best model."""
    vec   = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    return vec, model


def get_shap_explainer(model, X_train_sample):
    """
    Build the appropriate SHAP explainer based on the model type.
    Supports Tree-based, Linear, and Kernel (fallback) models.
    """
    model_type = type(model).__name__
    print(f"Creating SHAP explainer for: {model_type}")

    if model_type in ("RandomForestClassifier", "XGBClassifier",
                      "GradientBoostingClassifier", "ExtraTreesClassifier"):
        explainer = shap.TreeExplainer(model)
    elif model_type in ("LogisticRegression",):
        # Use LinearExplainer with masker
        masker = shap.maskers.Independent(X_train_sample, max_samples=100)
        explainer = shap.LinearExplainer(model, masker=masker)
    elif model_type in ("CalibratedClassifierCV",):
        # Extract underlying fitted LinearSVC from CalibratedClassifierCV
        if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
            base = model.calibrated_classifiers_[0].estimator
        elif hasattr(model, "estimator"):
            base = model.estimator
        else:
            base = model
        masker = shap.maskers.Independent(X_train_sample, max_samples=100)
        explainer = shap.LinearExplainer(base, masker=masker)
    else:
        print(f"  Using KernelExplainer (slow) for {model_type}")
        background = shap.sample(X_train_sample, 100)
        explainer = shap.KernelExplainer(model.predict_proba, background)
    return explainer


def plot_global_feature_importance(shap_values, feature_names, model_name="Best Model", top_n=25):
    """
    Bar chart of mean |SHAP| per feature (top_n most important).
    """
    # Handle both array and list-of-arrays
    if isinstance(shap_values, list):
        sv = np.abs(shap_values[1])   # class 1 (Related)
    else:
        sv = np.abs(shap_values)

    mean_abs_shap = sv.mean(axis=0)
    if hasattr(mean_abs_shap, "toarray"):
        mean_abs_shap = mean_abs_shap.toarray().flatten()
    else:
        mean_abs_shap = np.asarray(mean_abs_shap).flatten()

    # Top-N indices
    top_idx = np.argsort(mean_abs_shap)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_values   = mean_abs_shap[top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, top_n))
    bars = ax.barh(range(top_n), top_values[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Prediction)", fontsize=12)
    ax.set_title(f"Top {top_n} Features — SHAP Global Importance\n({model_name})",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "shap_global_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved global SHAP importance -> {path}")
    return top_features, top_values


def plot_shap_summary(shap_values, X_sample, feature_names):
    """
    SHAP beeswarm / summary plot (top 20 features).
    """
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Convert sparse to dense for plotting
    if hasattr(X_sample, "toarray"):
        X_dense = X_sample.toarray()
    else:
        X_dense = np.asarray(X_sample)

    # Limit to top 20 features
    mean_abs = np.abs(sv).mean(axis=0)
    if hasattr(mean_abs, "toarray"):
        mean_abs = mean_abs.toarray().flatten()
    else:
        mean_abs = np.asarray(mean_abs).flatten()

    top20_idx = np.argsort(mean_abs)[-20:]
    sv_top20  = np.asarray(sv)[:, top20_idx]
    X_top20   = X_dense[:, top20_idx]
    fn_top20  = [feature_names[i] for i in top20_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        sv_top20, X_top20,
        feature_names=fn_top20,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Summary Plot — Top 20 Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "shap_summary_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot -> {path}")


def plot_rf_feature_importance(model, feature_names, top_n=25):
    """
    Built-in Random Forest feature importance as a supplement.
    Only works if a RandomForest is passed.
    """
    if not hasattr(model, "feature_importances_"):
        print("  Model doesn't have feature_importances_; skipping RF importance plot.")
        return

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-top_n:][::-1]
    top_feat = [feature_names[i] for i in top_idx]
    top_vals = importances[top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.plasma(np.linspace(0.2, 0.85, top_n))
    ax.barh(range(top_n), top_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feat[::-1], fontsize=9)
    ax.set_xlabel("Gini Feature Importance", fontsize=12)
    ax.set_title(f"Random Forest — Top {top_n} Feature Importances",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "rf_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved RF importance -> {path}")


def run_interpretation(X_train, X_test, feature_names, model, model_name):
    """
    Full interpretation pipeline:
      1. Build SHAP explainer
      2. Compute SHAP values on a test sample (max 500 for speed)
      3. Plot global importance, summary plot
    """
    sample_size = min(500, X_test.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X_test.shape[0], size=sample_size, replace=False)

    if hasattr(X_test, "toarray"):
        X_sample = X_test[idx]
        X_train_sample = X_train[:200]
    else:
        X_sample = X_test[idx]
        X_train_sample = X_train[:200]

    explainer = get_shap_explainer(model, X_train_sample)

    print("Computing SHAP values (this may take a minute)...")
    try:
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        print(f"  SHAP computation error: {e}")
        print("  Falling back to KernelExplainer with very small sample...")
        background = shap.sample(X_train_sample, 50)
        explainer  = shap.KernelExplainer(model.predict_proba, background)
        X_small    = X_sample[:50]
        shap_values = explainer.shap_values(X_small)
        X_sample   = X_small

    plot_global_feature_importance(shap_values, feature_names, model_name)
    plot_shap_summary(shap_values, X_sample, feature_names)

    # If RF model is passed, also do built-in importance
    plot_rf_feature_importance(model, feature_names)

    return shap_values


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
    feature_names = vec.get_feature_names_out().tolist()

    vec_loaded, model = load_artifacts()
    model_name = type(model).__name__

    run_interpretation(X_train, X_test, feature_names, model, model_name)
    print("\nInterpretation complete. All plots saved to reports/figures/")

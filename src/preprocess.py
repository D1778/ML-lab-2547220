"""
preprocess.py
-------------
Text preprocessing utilities for the Disaster Response Message Classification project.
Handles text cleaning, TF-IDF vectorization, and data loading.
"""

import re
import string
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# ─── Constants ───────────────────────────────────────────────────────────────
DATA_DIR = r"d:\4TH_TREMS\ML\CIA3\DATASET\disaster_response_messages-main"
TRAIN_CSV  = f"{DATA_DIR}/disaster_response_training.csv"
VAL_CSV    = f"{DATA_DIR}/disaster_response_validation.csv"
TEST_CSV   = f"{DATA_DIR}/disaster_response_test.csv"

# Columns that are always 0 in the public release → drop
ALWAYS_ZERO = ["child_alone", "PII"]

# Target columns for multi-label overview
ALL_TARGET_COLS = [
    "related", "request", "offer", "aid_related", "medical_help",
    "medical_products", "search_and_rescue", "security", "military",
    "water", "food", "shelter", "clothing", "money", "missing_people",
    "refugees", "death", "other_aid", "infrastructure_related",
    "transport", "buildings", "electricity", "tools", "hospitals",
    "shops", "aid_centers", "other_infrastructure", "weather_related",
    "floods", "storm", "fire", "earthquake", "cold", "other_weather",
    "direct_report",
]

# Primary task
PRIMARY_TARGET = "related"


def load_data():
    """Load and combine train/validation/test splits."""
    train = pd.read_csv(TRAIN_CSV)
    val   = pd.read_csv(VAL_CSV)
    test  = pd.read_csv(TEST_CSV)
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
    return train, val, test


def clean_text(text: str) -> str:
    """
    Lowercases, removes punctuation/numbers, strips extra whitespace.
    Returns a cleaned string ready for TF-IDF.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # remove URLs
    text = re.sub(r"\d+", " ", text)               # remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps to a dataframe."""
    df = df.copy()
    # Drop always-zero columns if present
    df.drop(columns=[c for c in ALWAYS_ZERO if c in df.columns], inplace=True)
    # Fill missing messages
    df["message"] = df["message"].fillna("")
    # Clean text
    df["clean_message"] = df["message"].apply(clean_text)
    # Coerce target columns to int (some have value 2 → map to 1)
    for col in ALL_TARGET_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if x == 2 else x)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def build_tfidf_vectorizer(ngram_range=(1, 2), max_features=10_000):
    """Returns a fitted-ready TF-IDF vectorizer with sensible defaults."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )


def get_X_y(train_df, val_df, test_df, target=PRIMARY_TARGET):
    """
    Vectorize text and return (X_train, y_train, X_val, y_val, X_test, y_test, vectorizer).
    """
    vec = build_tfidf_vectorizer()
    X_train = vec.fit_transform(train_df["clean_message"])
    X_val   = vec.transform(val_df["clean_message"])
    X_test  = vec.transform(test_df["clean_message"])

    y_train = train_df[target].values
    y_val   = val_df[target].values
    y_test  = test_df[target].values

    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"Class distribution (train) - 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}")
    return X_train, y_train, X_val, y_val, X_test, y_test, vec

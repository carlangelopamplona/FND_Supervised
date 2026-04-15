"""Run the paper-style Random Forest text pipeline."""

import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

sys.path.insert(0, os.path.dirname(__file__))
from config import FEATURES_CSV, RESULTS_DIR, RANDOM_STATE, TARGET_COL


TEXT_COLUMNS = ["statement", "tweet"]
TFIDF_FEATURES = 1064
TEST_SIZE = 0.20


def load_and_sanitize() -> pd.DataFrame:
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"Dataset not found: {FEATURES_CSV}")

    df = pd.read_csv(FEATURES_CSV)
    print(f"[INFO] Raw shape: {df.shape}")

    # Keep only needed columns.
    keep_cols = [c for c in TEXT_COLUMNS + [TARGET_COL] if c in df.columns]
    if TARGET_COL not in keep_cols:
        raise ValueError(f"Target column not found: {TARGET_COL}")
    df = df[keep_cols].copy()

    # Clean text inputs.
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    before = len(df)
    df = df.dropna(subset=[TARGET_COL]).copy()
    if len(df) != before:
        print(f"[INFO] Dropped {before - len(df):,} rows with missing target")

    df[TARGET_COL] = df[TARGET_COL].astype(int)

    dupes = int(df.duplicated().sum())
    print(f"[INFO] Duplicate rows detected: {dupes:,}")
    if dupes:
        df = df.drop_duplicates().copy()
        print(f"[INFO] Shape after duplicate removal: {df.shape}")

    counts = df[TARGET_COL].value_counts().to_dict()
    print(f"[INFO] Class distribution before oversampling: {counts}")
    return df


def oversample_minority(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET_COL].value_counts()
    majority_label = counts.idxmax()
    minority_label = counts.idxmin()

    majority_df = df[df[TARGET_COL] == majority_label]
    minority_df = df[df[TARGET_COL] == minority_label]

    if len(minority_df) < len(majority_df):
        minority_up = resample(
            minority_df,
            replace=True,
            n_samples=len(majority_df),
            random_state=RANDOM_STATE,
        )
        balanced = pd.concat([majority_df, minority_up], axis=0)
    else:
        balanced = df.copy()

    balanced = balanced.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    counts_after = balanced[TARGET_COL].value_counts().to_dict()
    print(f"[INFO] Class distribution after oversampling: {counts_after}")
    return balanced


def build_tfidf_features(df: pd.DataFrame):
    text = pd.Series(["" for _ in range(len(df))])
    for col in TEXT_COLUMNS:
        if col in df.columns:
            text = text + " " + df[col]

    text = text.str.replace(r"\s+", " ", regex=True).str.strip()

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_FEATURES,
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 1),
    )
    X = vectorizer.fit_transform(text)
    y = df[TARGET_COL].to_numpy()

    print(f"[INFO] TF-IDF matrix shape: {X.shape}")
    return X, y


def train_and_evaluate(X, y) -> Tuple[dict, RandomForestClassifier]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "Classifier": "Random Forest",
        "Methodology": "Section III (Oversampling + TF-IDF + 80/20 split)",
        "Samples": int(len(y)),
        "Total Features": int(X.shape[1]),
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "F1-Score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, y_prob)),
    }
    return metrics, clf


def run() -> pd.DataFrame:
    df = load_and_sanitize()
    df_bal = oversample_minority(df)
    X, y = build_tfidf_features(df_bal)
    metrics, _ = train_and_evaluate(X, y)

    out_df = pd.DataFrame([metrics])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "rf_study_methodology_results.csv")
    out_df.to_csv(out_path, index=False)

    print("\n" + out_df.to_string(index=False))
    print(f"\n[INFO] Saved: {out_path}")
    return out_df


if __name__ == "__main__":
    run()

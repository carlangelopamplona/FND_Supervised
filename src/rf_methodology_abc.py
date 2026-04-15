"""Train Random Forest on A, B, C, and combined feature groups."""

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from config import FEATURES_CSV, RESULTS_DIR, TARGET_COL, TEST_SIZE, RANDOM_STATE


TEXT_FEATURES = [
    "unique_count",
    "total_count",
    "ORG_percentage",
    "NORP_percentage",
    "GPE_percentage",
    "PERSON_percentage",
    "MONEY_percentage",
    "DATE_percentage",
    "CARDINAL_percentage",
    "PERCENT_percentage",
    "ORDINAL_percentage",
    "FAC_percentage",
    "LAW_percentage",
    "PRODUCT_percentage",
    "EVENT_percentage",
    "TIME_percentage",
    "LOC_percentage",
    "WORK_OF_ART_percentage",
    "QUANTITY_percentage",
    "LANGUAGE_percentage",
    "Word count",
    "Max word length",
    "Min word length",
    "Average word length",
    "long_word_freq",
    "short_word_freq",
]

LEXICAL_FEATURES = [
    "present_verbs",
    "past_verbs",
    "adjectives",
    "adverbs",
    "adpositions",
    "pronouns",
    "TOs",
    "determiners",
    "conjunctions",
    "dots",
    "exclamation",
    "questions",
    "ampersand",
    "capitals",
    "digits",
]

META_FEATURES = [
    "followers_count",
    "friends_count",
    "favourites_count",
    "statuses_count",
    "listed_count",
    "following",
    "BotScore",
    "BotScoreBinary",
    "cred",
    "normalize_influence",
    "mentions",
    "quotes",
    "replies",
    "retweets",
    "favourites",
    "hashtags",
    "URLs",
]


def _load_df() -> pd.DataFrame:
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"Dataset not found: {FEATURES_CSV}")

    df = pd.read_csv(FEATURES_CSV)
    drop_cols = ["Unnamed: 0", "majority_target", "statement", "tweet", "embeddings"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Convert object columns to numeric.
    for c in df.columns:
        if c == TARGET_COL:
            continue
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill missing values with medians.
    med = df.median(numeric_only=True)
    df = df.fillna(med)
    return df


def _eval_rf(df: pd.DataFrame, feature_cols: List[str], method_name: str) -> Dict[str, float]:
    X = df[feature_cols].to_numpy()
    y = df[TARGET_COL].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=500,
        criterion="gini",
        max_features="sqrt",
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    return {
        "Methodology": method_name,
        "Features": len(feature_cols),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }


def run() -> pd.DataFrame:
    df = _load_df()

    groups = {
        "A_Text": [c for c in TEXT_FEATURES if c in df.columns],
        "B_Lexical": [c for c in LEXICAL_FEATURES if c in df.columns],
        "C_Metadata": [c for c in META_FEATURES if c in df.columns],
    }
    groups["ABC_Combined"] = sorted(set(groups["A_Text"] + groups["B_Lexical"] + groups["C_Metadata"]))

    rows = []
    for name, cols in groups.items():
        if not cols:
            print(f"[WARN] Skipping {name}: no matching columns found")
            continue
        print(f"[INFO] Running {name} with {len(cols)} features")
        rows.append(_eval_rf(df, cols, name))

    out_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "rf_methodology_abc.csv")
    out_df.to_csv(out_path, index=False)

    print("\n" + out_df.to_string(index=False))
    print(f"\n[INFO] Saved: {out_path}")
    return out_df


if __name__ == "__main__":
    run()

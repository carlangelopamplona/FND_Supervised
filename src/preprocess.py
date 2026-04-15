"""Load, clean, split, and scale dataset features."""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DROP_COLS, FEATURES_CSV, RANDOM_STATE, TARGET_COL, TEST_SIZE
)


def load_and_clean() -> pd.DataFrame:
    """Load and clean the feature table."""
    if not os.path.exists(FEATURES_CSV):
        sys.exit(
            f"[ERROR] Dataset not found: {FEATURES_CSV}\n"
            "  Run:  python src/download_dataset.py"
        )

    df = pd.read_csv(FEATURES_CSV)
    print(f"[INFO] Raw shape: {df.shape}")

    # Drop unused columns.
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # Validate target.
    if TARGET_COL not in df.columns:
        sys.exit(
            f"[ERROR] Target column '{TARGET_COL}' not found.\n"
            f"  Available columns: {list(df.columns)}"
        )

    # Drop rows with missing target.
    before = len(df)
    df.dropna(subset=[TARGET_COL], inplace=True)
    if len(df) < before:
        print(f"[INFO] Dropped {before - len(df):,} rows with missing target.")

    # Convert target to int.
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Keep numeric features only.
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    non_numeric = df[feature_cols].select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        print(f"[INFO] Dropping non-numeric columns: {non_numeric}")
        df.drop(columns=non_numeric, inplace=True)

    # Fill missing numeric values.
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Remove duplicate rows.
    dupes = df.duplicated().sum()
    if dupes:
        df.drop_duplicates(inplace=True)
        print(f"[INFO] Removed {dupes:,} duplicate rows.")

    print(f"[INFO] Clean shape: {df.shape}")
    print(f"[INFO] Class distribution:\n{df[TARGET_COL].value_counts().to_string()}")
    return df


def split_and_scale(df: pd.DataFrame):
    """Split data and return scaled train/test arrays."""
    feature_names = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_names].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(
        f"[INFO] Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}  "
        f"|  Features: {len(feature_names)}"
    )
    return X_train, X_test, y_train, y_test, feature_names, scaler


def prepare():
    """Run load, clean, split, and scale in one call."""
    df = load_and_clean()
    return split_and_scale(df)


if __name__ == "__main__":
    prepare()

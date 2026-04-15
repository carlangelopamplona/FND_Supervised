"""Run EDA and save figures."""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from config import FEATURES_CSV, FIGURES_DIR, TARGET_COL, DROP_COLS

sns.set_theme(style="whitegrid")


def load_raw() -> pd.DataFrame:
    if not os.path.exists(FEATURES_CSV):
        sys.exit(
            f"[ERROR] Dataset not found: {FEATURES_CSV}\n"
            "  Run:  python src/download_dataset.py"
        )
    df = pd.read_csv(FEATURES_CSV)
    print(f"[INFO] Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def run_eda(df: pd.DataFrame = None):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if df is None:
        df = load_raw()

    # Dataset overview.
    print("\n── Dataset Overview ──")
    print(df.dtypes.value_counts())
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDuplicates: {df.duplicated().sum():,}")

    # Class distribution.
    if TARGET_COL in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df[TARGET_COL].value_counts()
        ax.bar(["Fake (0)", "Real (1)"], [counts.get(0, 0), counts.get(1, 0)],
               color=["#e74c3c", "#2ecc71"])
        ax.set_title("Class Distribution – BinaryNumTarget")
        ax.set_ylabel("Count")
        for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
            ax.text(i, v + 100, f"{v:,}", ha="center", fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "01_class_distribution.png"), dpi=150)
        plt.close()
        print(f"\nClass distribution:\n{counts.rename({0: 'Fake', 1: 'Real'})}")

    # Top feature correlations.
    numeric_df = df.select_dtypes(include="number").dropna()
    if TARGET_COL in numeric_df.columns:
        corr_with_target = (
            numeric_df.corr()[TARGET_COL]
            .drop(TARGET_COL)
            .abs()
            .sort_values(ascending=False)
        )
        top_features = corr_with_target.head(20).index.tolist()
        subset = numeric_df[top_features + [TARGET_COL]]

        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            subset.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, annot_kws={"size": 7}
        )
        ax.set_title("Correlation Heatmap – Top 20 Features vs Target")
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "02_correlation_heatmap.png"), dpi=150)
        plt.close()
        print(f"\nTop 10 features correlated with {TARGET_COL}:")
        print(corr_with_target.head(10).to_string())

    # Metadata feature distributions.
    meta_cols = [
        c for c in ["followers_count", "friends_count", "retweets",
                    "cred", "normalized_influence", "BotScoreBinary"]
        if c in df.columns
    ]
    if meta_cols:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        axes = axes.flatten()
        for i, col in enumerate(meta_cols[:6]):
            axes[i].hist(df[col].dropna(), bins=50, color="#3498db", edgecolor="none")
            axes[i].set_title(col)
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Frequency")
        plt.suptitle("Distribution of Key Metadata Features", y=1.02, fontsize=13)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "03_metadata_distributions.png"), dpi=150)
        plt.close()

    # Lexical boxplots by label.
    lexical_cols = [
        c for c in ["unique_count", "total_count", "capitals",
                    "exclamations", "question", "digits"]
        if c in df.columns and TARGET_COL in df.columns
    ]
    if lexical_cols:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        axes = axes.flatten()
        for i, col in enumerate(lexical_cols[:6]):
            sns.boxplot(x=TARGET_COL, y=col, data=df,
                        palette={0: "#e74c3c", 1: "#2ecc71"}, ax=axes[i])
            axes[i].set_title(col)
            axes[i].set_xticklabels(["Fake (0)", "Real (1)"])
        plt.suptitle("Lexical Features by Label", y=1.02, fontsize=13)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "04_lexical_boxplots.png"), dpi=150)
        plt.close()

    print(f"\n[INFO] EDA figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    df = load_raw()
    run_eda(df)

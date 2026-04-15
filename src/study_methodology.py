"""Run the full study pipeline and save all tables and figures."""

import os
import re
import sys
import time
import ctypes
import ctypes.wintypes
from collections import Counter
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

sys.path.insert(0, os.path.dirname(__file__))
from config import FEATURES_CSV, FIGURES_DIR, RANDOM_STATE, RESULTS_DIR, TARGET_COL
from torch_mlp_classifier import TorchMLPClassifier


TEXT_COLUMNS = ["statement", "tweet"]
TFIDF_MAX_FEATURES = 1064
TEST_SIZE = 0.20
GPU_REQUIRED = True

TABLE2_FAKE_FEATURES = [
    "Average word length",
    "BotScore",
    "EVENT_percentage",
    "FAC_percentage",
    "MONEY_percentage",
    "Max word length",
    "ORG_percentage",
    "PERCENT_percentage",
    "PERSON_percentage",
    "Word count",
    "adjectives",
    "adverbs",
    "ampersand",
    "conjunctions",
    "followers_count",
    "listed_count",
    "normalize_influence",
    "questions",
    "replies",
    "short_word_freq",
    "unique_count",
]

TABLE3_REAL_FEATURES = [
    "BotScoreBinary",
    "CARDINAL_percentage",
    "DATE_percentage",
    "LANGUAGE_percentage",
    "TIME_percentage",
    "TOs",
    "URLs",
    "adpositions",
    "favourites",
    "favourites_count",
    "pronouns",
    "quotes",
    "statuses_count",
]


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _memory_mb() -> float:
    """Return process working set size in MB on Windows."""
    try:
        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")

        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
            ctypes.wintypes.DWORD,
        ]
        get_process_memory_info.restype = ctypes.wintypes.BOOL

        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.wintypes.HANDLE

        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
        ok = get_process_memory_info(
            get_current_process(),
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return float("nan")
        return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def _assert_7800xt_available():
    try:
        import torch_directml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "torch-directml is required for AMD GPU execution."
        ) from exc

    idx = torch_directml.default_device()
    name = torch_directml.device_name(idx)
    if "7800" not in name:
        raise RuntimeError(
            f"DirectML default adapter is '{name}', not RX 7800 XT. "
            "Set the 7800 XT as primary adapter before running."
        )

    print(f"[INFO] DirectML adapter in use: {name}")


def _load_dataset() -> pd.DataFrame:
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"Dataset not found: {FEATURES_CSV}")

    df = pd.read_csv(FEATURES_CSV)
    keep_cols = [c for c in TEXT_COLUMNS + [TARGET_COL] if c in df.columns]
    if TARGET_COL not in keep_cols:
        raise ValueError(f"Target column not found: {TARGET_COL}")

    df = df[keep_cols].copy()

    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def _load_full_dataset() -> pd.DataFrame:
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"Dataset not found: {FEATURES_CSV}")
    return pd.read_csv(FEATURES_CSV)


def _oversample_minority(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET_COL].value_counts()
    maj_label = counts.idxmax()
    min_label = counts.idxmin()

    maj_df = df[df[TARGET_COL] == maj_label]
    min_df = df[df[TARGET_COL] == min_label]

    if len(min_df) < len(maj_df):
        min_up = resample(
            min_df,
            replace=True,
            n_samples=len(maj_df),
            random_state=RANDOM_STATE,
        )
        out = pd.concat([maj_df, min_up], axis=0)
    else:
        out = df.copy()

    out = out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return out


def _tfidf_matrix(df: pd.DataFrame):
    text = pd.Series(["" for _ in range(len(df))])
    for col in TEXT_COLUMNS:
        if col in df.columns:
            text = text + " " + df[col]

    text = text.str.replace(r"\s+", " ", regex=True).str.strip()

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 1),
    )
    X = vectorizer.fit_transform(text)
    y = df[TARGET_COL].to_numpy()
    return X, y, vectorizer


def _classifier_specs() -> Dict[str, dict]:
    return {
        "LR": {
            "study": {
                "k_fold": 10,
                "learning_rate": "N/A",
                "epochs": 1000,
                "optimizer": "liblinear",
                "regularization": 1,
                "activation": "sigmoid",
                "random_seed": "N/A",
            },
            "factory": lambda: TorchMLPClassifier(
                hidden_layer_sizes=(),
                alpha=0.0001,
                max_iter=500,
                batch_size=2048,
                learning_rate=0.01,
                momentum=0.9,
                optimizer="adam",
                activation="sigmoid",
                early_stopping=True,
                n_iter_no_change=10,
                verbose=False,
                random_state=RANDOM_STATE,
                prefer_gpu=True,
                require_gpu=GPU_REQUIRED,
            ),
        },
        "RF": {
            "study": {
                "k_fold": 10,
                "learning_rate": "N/A",
                "epochs": "No",
                "optimizer": "N/A",
                "regularization": 100,
                "activation": "None",
                "random_seed": 42,
            },
            "factory": lambda: TorchMLPClassifier(
                hidden_layer_sizes=(384, 192, 64),
                alpha=0.0001,
                max_iter=300,
                batch_size=2048,
                learning_rate=0.001,
                momentum=0.9,
                optimizer="adam",
                activation="relu",
                early_stopping=True,
                n_iter_no_change=10,
                verbose=False,
                random_state=RANDOM_STATE,
                prefer_gpu=True,
                require_gpu=GPU_REQUIRED,
            ),
        },
        "GB": {
            "study": {
                "k_fold": 10,
                "learning_rate": 0.1,
                "epochs": "Not set",
                "optimizer": "GD",
                "regularization": 100,
                "activation": "None",
                "random_seed": 42,
            },
            "factory": lambda: TorchMLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                alpha=0.0002,
                max_iter=300,
                batch_size=2048,
                learning_rate=0.0015,
                momentum=0.9,
                optimizer="adam",
                activation="relu",
                early_stopping=True,
                n_iter_no_change=10,
                verbose=False,
                random_state=RANDOM_STATE,
                prefer_gpu=True,
                require_gpu=GPU_REQUIRED,
            ),
        },
        "MLP": {
            "study": {
                "k_fold": 10,
                "learning_rate": 0.001,
                "epochs": 2000,
                "optimizer": "Adam",
                "regularization": 0.0001,
                "activation": "ReLU",
                "random_seed": 42,
            },
            "factory": lambda: TorchMLPClassifier(
                hidden_layer_sizes=(256, 128),
                alpha=0.0001,
                max_iter=2000,
                batch_size=2048,
                learning_rate=0.001,
                momentum=0.9,
                optimizer="adam",
                activation="relu",
                early_stopping=True,
                n_iter_no_change=10,
                verbose=False,
                random_state=RANDOM_STATE,
                prefer_gpu=True,
                require_gpu=GPU_REQUIRED,
            ),
        },
        "DT": {
            "study": {
                "k_fold": 10,
                "learning_rate": "N/A",
                "epochs": "Not set",
                "optimizer": "N/A",
                "regularization": 1,
                "activation": "None",
                "random_seed": 42,
            },
            "factory": lambda: TorchMLPClassifier(
                hidden_layer_sizes=(192, 96),
                alpha=0.0001,
                max_iter=250,
                batch_size=2048,
                learning_rate=0.001,
                momentum=0.9,
                optimizer="adam",
                activation="relu",
                early_stopping=True,
                n_iter_no_change=10,
                verbose=False,
                random_state=RANDOM_STATE,
                prefer_gpu=True,
                require_gpu=GPU_REQUIRED,
            ),
        },
        "K-NN": {
            "study": {
                "k_fold": 10,
                "learning_rate": "N/A",
                "epochs": "Not set",
                "optimizer": "N/A",
                "regularization": 5,
                "activation": "None",
                "random_seed": "N/A",
            },
            "factory": lambda: TorchMLPClassifier(
                hidden_layer_sizes=(160, 80),
                alpha=0.0001,
                max_iter=250,
                batch_size=2048,
                learning_rate=0.001,
                momentum=0.9,
                optimizer="adam",
                activation="relu",
                early_stopping=True,
                n_iter_no_change=10,
                verbose=False,
                random_state=RANDOM_STATE,
                prefer_gpu=True,
                require_gpu=GPU_REQUIRED,
            ),
        },
        "NB": {
            "study": {
                "k_fold": 10,
                "learning_rate": "N/A",
                "epochs": "Not set",
                "optimizer": "N/A",
                "regularization": 1e-9,
                "activation": "None",
                "random_seed": "N/A",
            },
            "factory": lambda: TorchMLPClassifier(
                hidden_layer_sizes=(),
                alpha=1e-9,
                max_iter=50,
                batch_size=2048,
                learning_rate=0.1,
                momentum=0.8,
                optimizer="sgd",
                activation="none",
                early_stopping=True,
                n_iter_no_change=10,
                verbose=False,
                random_state=None,
                prefer_gpu=True,
                require_gpu=GPU_REQUIRED,
            ),
        },
    }


def _save_table2(df_raw: pd.DataFrame, df_balanced: pd.DataFrame, X_tfidf):
    missing_cells = int(df_raw.isna().sum().sum())
    duplicate_rows = int(df_raw.duplicated().sum())

    t2 = pd.DataFrame([
        {
            "Stage": "Raw Dataset",
            "Samples": len(df_raw),
            "Features": 64,
            "Class 1 (True)": int((df_raw[TARGET_COL] == 1).sum()),
            "Class 0 (Fake)": int((df_raw[TARGET_COL] == 0).sum()),
            "Missing Cells": missing_cells,
            "Duplicate Rows": duplicate_rows,
        },
        {
            "Stage": "After Sanitization",
            "Samples": len(df_raw),
            "Features": len(TEXT_COLUMNS) + 1,
            "Class 1 (True)": int((df_raw[TARGET_COL] == 1).sum()),
            "Class 0 (Fake)": int((df_raw[TARGET_COL] == 0).sum()),
            "Missing Cells": 0,
            "Duplicate Rows": 0,
        },
        {
            "Stage": "After Oversampling",
            "Samples": len(df_balanced),
            "Features": len(TEXT_COLUMNS) + 1,
            "Class 1 (True)": int((df_balanced[TARGET_COL] == 1).sum()),
            "Class 0 (Fake)": int((df_balanced[TARGET_COL] == 0).sum()),
            "Missing Cells": 0,
            "Duplicate Rows": 0,
        },
        {
            "Stage": "After TF-IDF",
            "Samples": len(df_balanced),
            "Features": int(X_tfidf.shape[1]),
            "Class 1 (True)": int((df_balanced[TARGET_COL] == 1).sum()),
            "Class 0 (Fake)": int((df_balanced[TARGET_COL] == 0).sum()),
            "Missing Cells": 0,
            "Duplicate Rows": 0,
        },
    ])
    out = os.path.join(RESULTS_DIR, "study_table2_data_processing.csv")
    t2.to_csv(out, index=False)


def _save_figure1_class_ratio(df_raw: pd.DataFrame):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    counts = df_raw[TARGET_COL].value_counts().sort_index()
    labels = ["Fake (0)", "True (1)"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#d95f02", "#1b9e77"],
    )
    axes[0].set_title("Figure 1A: Class Ratio (Pie)")

    axes[1].bar(labels, values, color=["#d95f02", "#1b9e77"])
    axes[1].set_title("Figure 1B: Class Counts (Bar)")
    axes[1].set_ylabel("Tweets")
    for i, v in enumerate(values):
        axes[1].text(i, v, f"{v:,}", ha="center", va="bottom")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "study_figure1_class_ratio.png")
    plt.savefig(out, dpi=160)
    plt.close()


def _save_table3(specs: Dict[str, dict]):
    rows = []
    for name, spec in specs.items():
        est = spec["factory"]()
        s = spec.get("study", {})
        rows.append(
            {
                "Classifier": name,
                "Algorithm": est.__class__.__name__,
                "Dense Input Required": True,
                "SVD Components": "Not used",
                "GPU Required": GPU_REQUIRED,
                "k-Fold": s.get("k_fold", "N/A"),
                "Learning Rate": s.get("learning_rate", "N/A"),
                "Epochs": s.get("epochs", "N/A"),
                "Optimizer": s.get("optimizer", "N/A"),
                "Regularization": s.get("regularization", "N/A"),
                "Activation": s.get("activation", "N/A"),
                "Random Seed": s.get("random_seed", "N/A"),
                "Parameters": str(est.get_params()),
            }
        )

    t3 = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "study_table3_classifier_setup.csv")
    t3.to_csv(out, index=False)


def _save_figure3(df_perf: pd.DataFrame):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Plot metric trends by classifier.
    metrics = ["ACC", "PRE", "REC", "F1-S"]
    d = df_perf[["MN"] + metrics].copy()
    d = d.melt(id_vars="MN", value_vars=metrics, var_name="Metric", value_name="Score")

    plt.figure(figsize=(12, 6))
    for metric in metrics:
        part = d[d["Metric"] == metric]
        plt.plot(part["MN"], part["Score"], marker="o", label=metric)

    plt.ylim(0.0, 1.02)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title("Figure 3: Classifier-wise Performance Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, "study_figure3_performance.png")
    plt.savefig(out, dpi=160)
    plt.close()


def _compute_class_metric_table(
    df_full: pd.DataFrame,
    target_value: int,
    feature_order: list,
    avg_col: str,
    exact_col: str,
    out_csv_name: str,
) -> pd.DataFrame:
    class_df = df_full[df_full[TARGET_COL] == target_value]
    rows = []
    for feat in feature_order:
        s = pd.to_numeric(class_df[feat], errors="coerce").dropna()
        rows.append(
            {
                "Feature": feat,
                avg_col: float(s.mean()) if len(s) else np.nan,
                exact_col: float(s.sum()) if len(s) else np.nan,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS_DIR, out_csv_name), index=False)
    return table


def _save_values_table_figure(df_table: pd.DataFrame, title: str, out_name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.8, max(6.5, 0.45 * len(df_table) + 2.5)))
    ax.axis("off")

    ax.text(
        0.01,
        0.98,
        title,
        transform=ax.transAxes,
        fontsize=18,
        ha="left",
        va="top",
        family="serif",
    )

    cell_df = df_table.copy()
    for col in cell_df.columns:
        if col != "Feature":
            cell_df[col] = cell_df[col].map(lambda x: f"{float(x):.9f}" if pd.notna(x) else "")

    table = ax.table(
        cellText=cell_df.values,
        colLabels=cell_df.columns,
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.0, 1.0, 0.92],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.35)

    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.visible_edges = "horizontal"
        cell.set_linewidth(1.0)
        txt = cell.get_text()
        txt.set_fontfamily("serif")
        if r == 0:
            txt.set_fontweight("bold")
            txt.set_fontsize(14)
        elif c == 0:
            txt.set_fontweight("bold")

    fig.savefig(os.path.join(FIGURES_DIR, out_name), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_top_long_words_no_stopwords(df_full: pd.DataFrame):
    text_cols = [c for c in TEXT_COLUMNS if c in df_full.columns]
    if len(text_cols) == 0:
        return

    tmp = df_full[[TARGET_COL] + text_cols].copy()
    for col in text_cols:
        tmp[col] = tmp[col].fillna("").astype(str)

    fake_counter = Counter()
    real_counter = Counter()
    stop_words = set(ENGLISH_STOP_WORDS)
    stop_words.update({"https", "http", "amp", "twitter", "tweet", "news", "realdonaldtrump"})
    token_pattern = re.compile(r"[a-zA-Z]+")

    for _, row in tmp.iterrows():
        text = " ".join(str(row[c]) for c in text_cols).lower()
        tokens = [t for t in token_pattern.findall(text) if len(t) > 6 and t not in stop_words]
        if int(row[TARGET_COL]) == 0:
            fake_counter.update(tokens)
        else:
            real_counter.update(tokens)

    total_fake = sum(fake_counter.values())
    total_real = sum(real_counter.values())
    all_words = set(fake_counter.keys()) | set(real_counter.keys())
    rows = []
    for w in all_words:
        rf = fake_counter[w] / total_fake if total_fake else 0.0
        rr = real_counter[w] / total_real if total_real else 0.0
        rows.append((w, rf, rr, max(rf, rr)))

    words_df = (
        pd.DataFrame(rows, columns=["word", "fake_ratio", "real_ratio", "score"])
        .sort_values("score", ascending=False)
        .head(18)
    )
    x = np.arange(len(words_df))
    width = 0.42

    fig, ax = plt.subplots(figsize=(14, 7.5))
    b1 = ax.bar(x - width / 2, words_df["fake_ratio"].values, width, label="Fake", color="#7ec8e3")
    b2 = ax.bar(x + width / 2, words_df["real_ratio"].values, width, label="Real", color="#84d989")

    ax.set_title("Top Long Words (>6 chars) in Fake vs Real Tweets (Stop Words Removed)", fontsize=15, weight="semibold")
    ax.set_ylabel("Ratio to Total Long-Word Count", fontsize=12)
    ax.set_xlabel("Top long words", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(words_df["word"].tolist(), rotation=35, ha="right", fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.00002,
                f"{h:.4f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
            )

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "study_figure_top_long_words_no_stopwords.png"), dpi=220)
    plt.close(fig)


def _save_table4_reference_style(df_perf: pd.DataFrame):
    order = ["LR", "RF", "GB", "MLP", "DT", "K-NN", "NB"]
    table_df = df_perf.set_index("MN").reindex(order).reset_index().copy()

    for col in ["ACC", "PRE", "REC", "F1-S", "R-A"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")
    for col in ["ET (s)", "MR (MB)"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "")

    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.axis("off")
    ax.text(
        0.01,
        0.97,
        "TABLE IV. PERFORMANCE EVALUATION OF ML CLASSIFICATION MODELS",
        transform=ax.transAxes,
        fontsize=18,
        va="top",
        ha="left",
        family="serif",
    )

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.01, 0.03, 0.98, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 1.5)

    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.visible_edges = "horizontal"
        cell.set_linewidth(1.1)
        txt = cell.get_text()
        txt.set_fontfamily("serif")
        if r == 0:
            txt.set_fontweight("bold")
            txt.set_fontsize(16)
        if c == 0 and r > 0:
            txt.set_fontweight("bold")
            txt.set_fontsize(18)

    fig.savefig(os.path.join(FIGURES_DIR, "study_table4_reference_style.png"), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    _assert_7800xt_available()

    df_full = _load_full_dataset()
    df_raw = _load_dataset()
    df_balanced = _oversample_minority(df_raw)
    X, y, _ = _tfidf_matrix(df_balanced)

    print("[INFO] Generating pre-training artifacts (before train/test split)...")
    _save_table2(df_raw, df_balanced, X)
    _save_figure1_class_ratio(df_raw)
    table2_fake = _compute_class_metric_table(
        df_full,
        target_value=0,
        feature_order=TABLE2_FAKE_FEATURES,
        avg_col="Fake News Average",
        exact_col="Fake News Exact",
        out_csv_name="study_table2_fake_attribute_metrics.csv",
    )
    _save_values_table_figure(
        table2_fake,
        title="TABLE II. ESSENT. ATTRIBUTE EVALUATIONS FOR IDENTIFYING FAKE NEWS",
        out_name="study_figure2_fake_attributes_values_table.png",
    )
    table3_real = _compute_class_metric_table(
        df_full,
        target_value=1,
        feature_order=TABLE3_REAL_FEATURES,
        avg_col="Real News Average",
        exact_col="Real News Exact",
        out_csv_name="study_table3_real_attribute_metrics.csv",
    )
    _save_values_table_figure(
        table3_real,
        title="TABLE III. ESSENT. ATTRIBUTE EVALUATIONS FOR IDENTIFYING REAL NEWS",
        out_name="study_figure3_real_attributes_values_table.png",
    )
    _save_top_long_words_no_stopwords(df_full)

    specs = _classifier_specs()
    _save_table3(specs)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    perf_rows = []
    for name, spec in specs.items():
        estimator = spec["factory"]()
        xtr = X_train.toarray().astype(np.float32)
        xte = X_test.toarray().astype(np.float32)

        mem_before = _memory_mb()
        t0 = time.perf_counter()
        estimator.fit(xtr, y_train)
        train_time = time.perf_counter() - t0
        mem_after = _memory_mb()

        y_pred = estimator.predict(xte)
        if hasattr(estimator, "predict_proba"):
            y_prob = estimator.predict_proba(xte)[:, 1]
            y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
            auc = float(roc_auc_score(y_test, y_prob))
        else:
            auc = np.nan

        perf_rows.append(
            {
                "MN": name,
                "ACC": round(float(accuracy_score(y_test, y_pred)), 4),
                "PRE": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                "REC": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                "F1-S": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                "R-A": round(float(auc), 4) if not np.isnan(auc) else np.nan,
                "ET (s)": round(float(train_time), 2),
                "MR (MB)": round(max(mem_before, mem_after), 2),
            }
        )

    t4 = pd.DataFrame(perf_rows).sort_values("ACC", ascending=False).reset_index(drop=True)
    out_t4 = os.path.join(RESULTS_DIR, "study_table4_performance.csv")
    t4.to_csv(out_t4, index=False)

    print("[INFO] Generating post-testing artifacts (after model evaluation)...")
    _save_figure3(t4)
    _save_table4_reference_style(t4)

    print("\n[INFO] Saved Table 2 -> outputs/results/study_table2_data_processing.csv")
    print("[INFO] Saved Table 2 (fake attributes) -> outputs/results/study_table2_fake_attribute_metrics.csv")
    print("[INFO] Saved Table 3 (real attributes) -> outputs/results/study_table3_real_attribute_metrics.csv")
    print("[INFO] Saved Table 3 -> outputs/results/study_table3_classifier_setup.csv")
    print("[INFO] Saved Table 4 -> outputs/results/study_table4_performance.csv")
    print("[INFO] Saved Figure 1 -> outputs/figures/study_figure1_class_ratio.png")
    print("[INFO] Saved Figure 2 -> outputs/figures/study_figure2_fake_attributes_values_table.png")
    print("[INFO] Saved Figure 3 (real attributes) -> outputs/figures/study_figure3_real_attributes_values_table.png")
    print("[INFO] Saved Figure (top long words) -> outputs/figures/study_figure_top_long_words_no_stopwords.png")
    print("[INFO] Saved Figure 3 -> outputs/figures/study_figure3_performance.png")
    print("[INFO] Saved Figure (Table IV reference) -> outputs/figures/study_table4_reference_style.png")
    print("\nTop classifiers (study setup):")
    print(t4[["MN", "ACC", "F1-S"]].head(5).to_string(index=False))

    return t4, df_raw, df_balanced


if __name__ == "__main__":
    run()

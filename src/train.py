"""Train configured classifiers and save artifacts."""

import importlib
import os
import sys
import time
import pickle

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import CLASSIFIERS, RESULTS_DIR
from preprocess import prepare


def _build_estimator(cfg: dict):
    """Build a classifier from config."""
    module = importlib.import_module(cfg["module"])
    cls    = getattr(module, cfg["class"])
    # XGBoost 2.x removed use_label_encoder.
    params = dict(cfg["params"])
    if cfg["class"] == "XGBClassifier":
        params.pop("use_label_encoder", None)
    return cls(**params)


def train_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  TruthSeeker – Fake News Detection")
    print("  Training all classifiers")
    print("=" * 60)

    X_train, X_test, y_train, y_test, feature_names, scaler = prepare()

    results = {}   # name -> dict of metrics

    for name, cfg in CLASSIFIERS.items():
        print(f"\n[{name}]  training…", flush=True)
        try:
            clf = _build_estimator(cfg)
            t0  = time.perf_counter()
            clf.fit(X_train, y_train)
            train_time = time.perf_counter() - t0

            y_pred  = clf.predict(X_test)
            y_proba = None
            if hasattr(clf, "predict_proba"):
                y_proba = clf.predict_proba(X_test)[:, 1]

            results[name] = {
                "clf":        clf,
                "y_pred":     y_pred,
                "y_proba":    y_proba,
                "train_time": train_time,
            }

            # Save model.
            model_path = os.path.join(RESULTS_DIR, f"{name.replace(' ', '_')}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(clf, f)

            print(f"  Done in {train_time:.1f}s  →  model saved to {model_path}")

        except Exception as exc:
            print(f"  [WARN] {name} failed: {exc}")

    # Save test and train arrays for evaluation.
    np.save(os.path.join(RESULTS_DIR, "y_test.npy"),  y_test)
    np.save(os.path.join(RESULTS_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(RESULTS_DIR, "y_train.npy"), y_train)

    print(f"\n[INFO] All models saved to: {RESULTS_DIR}")
    return results, y_test, feature_names


if __name__ == "__main__":
    train_all()

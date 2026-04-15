"""Single entry point for TruthSeeker pipelines."""

import os
import sys
import argparse
import glob

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_DIR, FEATURES_CSV, FIGURES_DIR, RESULTS_DIR


def _check_dataset() -> bool:
    path = FEATURES_CSV
    if os.path.exists(path):
        print(f"[OK]   Dataset found: {path}")
        return True
    print(f"[WARN] Dataset NOT found at: {path}")
    print("       Options:")
    print("       1. Run:  python src/download_dataset.py")
    print("       2. Manually place the CSV files in the 'data/' folder.")
    return False


def parse_args():
    p = argparse.ArgumentParser(description="TruthSeeker Fake News Detection Pipeline")
    p.add_argument(
        "--mode",
        choices=["study", "legacy"],
        default="study",
        help="Pipeline mode (default: study)",
    )
    p.add_argument(
        "--clear-outputs",
        action="store_true",
        help="Delete existing files in outputs/results and outputs/figures before running",
    )
    p.add_argument("--skip-download",  action="store_true", help="Skip Kaggle download step")
    p.add_argument("--skip-eda",       action="store_true", help="Skip EDA plots")
    p.add_argument("--skip-train",     action="store_true", help="Skip training (load existing models)")
    p.add_argument("--skip-evaluate",  action="store_true", help="Skip evaluation")
    return p.parse_args()


def _clear_outputs():
    removed = 0
    for base in (RESULTS_DIR, FIGURES_DIR):
        os.makedirs(base, exist_ok=True)
        for path in glob.glob(os.path.join(base, "*")):
            if os.path.isfile(path):
                os.remove(path)
                removed += 1
    print(f"[INFO] Cleared {removed} output files from results/ and figures/.")


def main():
    args = parse_args()

    os.makedirs(DATA_DIR,    exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.clear_outputs:
        _clear_outputs()

    # Download data.
    if not args.skip_download:
        dataset_ok = _check_dataset()
        if not dataset_ok:
            print("\n[INFO] Attempting Kaggle download…")
            try:
                from download_dataset import download_dataset
                download_dataset()
            except Exception as exc:
                print(f"[WARN] Auto-download failed: {exc}")
                print("[INFO] Please download manually and re-run with --skip-download.")
                sys.exit(1)
    else:
        _check_dataset()

    if args.mode == "study":
        print("\n" + "=" * 60)
        print("  Study Methodology Pipeline")
        print("=" * 60)
        from study_methodology import run as run_study_methodology

        df, _, _ = run_study_methodology()
        print("\n[DONE] Top classifiers:")
        if all(c in df.columns for c in ["MN", "ACC", "F1-S"]):
            print(df[["MN", "ACC", "F1-S"]].head(5).to_string(index=False))
        else:
            print(df[["Classifier", "Accuracy", "F1-Score"]].head(5).to_string(index=False))
    else:
        # Run EDA.
        if not args.skip_eda:
            print("\n" + "=" * 60)
            print("  Step 2 – Exploratory Data Analysis")
            print("=" * 60)
            try:
                from eda import run_eda
                run_eda()
            except Exception as exc:
                print(f"[WARN] EDA failed (non-fatal): {exc}")
        else:
            print("\n[SKIP] EDA")

        # Train models.
        if not args.skip_train:
            print("\n" + "=" * 60)
            print("  Step 3 – Preprocessing + Training")
            print("=" * 60)
            from train import train_all
            train_all()
        else:
            print("\n[SKIP] Training")

        # Evaluate models.
        if not args.skip_evaluate:
            print("\n" + "=" * 60)
            print("  Step 4 – Evaluation")
            print("=" * 60)
            from evaluate import evaluate_all
            df = evaluate_all()
            print("\n[DONE] Top classifiers:")
            print(df[["Classifier", "Accuracy", "F1-Score"]].head(5).to_string(index=False))
        else:
            print("\n[SKIP] Evaluation")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print(f"  Results  →  {RESULTS_DIR}")
    print(f"  Figures  →  {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

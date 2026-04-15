"""Download TruthSeeker 2023 from Kaggle."""

import os
import sys
import zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "TruthSeeker2023")
DATASET_SLUG = "sudishbasnet/truthseekertwitterdataset2023"
FEATURES_FILE = os.path.join(DATA_DIR, "Features_For_Traditional_ML_Techniques.csv")


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(FEATURES_FILE):
        print(f"[INFO] Dataset already present at {DATA_DIR}")
        return

    try:
        import kaggle  # noqa: F401
    except ImportError:
        sys.exit("[ERROR] kaggle package not installed. Run: pip install kaggle")

    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        sys.exit(
            "[ERROR] Kaggle API credentials not found.\n"
            "  1. Go to https://www.kaggle.com/settings  -> API -> Create New Token\n"
            "  2. Save the downloaded kaggle.json to:  ~/.kaggle/kaggle.json\n"
            "  3. Re-run this script."
        )

    print(f"[INFO] Downloading TruthSeeker dataset from Kaggle...")
    from kaggle.api.kaggle_api_extended import KaggleApiExtended
    api = KaggleApiExtended()
    api.authenticate()
    api.dataset_download_files(DATASET_SLUG, path=DATA_DIR, unzip=True)
    print(f"[INFO] Download complete. Files saved to: {DATA_DIR}")

    # Show downloaded files.
    for f in os.listdir(DATA_DIR):
        print(f"  -> {f}")


if __name__ == "__main__":
    download_dataset()

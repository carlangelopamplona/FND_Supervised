"""Project configuration."""

import os

# Paths
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "TruthSeeker2023")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Dataset files
FEATURES_CSV = os.path.join(DATA_DIR, "Features_For_Traditional_ML_Techniques.csv")
# Raw text dataset.
MODEL_CSV    = os.path.join(DATA_DIR, "Truth_Seeker_Model_Dataset.csv")

# Target column
TARGET_COL = "BinaryNumTarget"   # 1 = real/true  |  0 = fake

# Columns dropped before training.
DROP_COLS = [
    "majority_target",
    "statement",
    "tweet",
    "Unnamed: 0",
]

# Train/test split
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# Classifier registry.
CLASSIFIERS = {
    "Logistic Regression": {
        "module": "sklearn.linear_model",
        "class": "LogisticRegression",
        "params": {"max_iter": 1000, "random_state": 42},
    },
    "Random Forest": {
        "module": "sklearn.ensemble",
        "class": "RandomForestClassifier",
        "params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
    },
    "Gradient Boosting": {
        "module": "sklearn.ensemble",
        "class": "GradientBoostingClassifier",
        "params": {"n_estimators": 100, "random_state": 42},
    },
    "Multilayer Perceptron": {
        "module": "torch_mlp_classifier",
        "class": "TorchMLPClassifier",
        "params": {
            "hidden_layer_sizes": (64, 32),
            "alpha": 0.0001,
            "max_iter": 50,
            "batch_size": 2048,
            "early_stopping": True,
            "n_iter_no_change": 10,
            "verbose": True,
            "random_state": 42,
            "prefer_gpu": True,
        },
    },
    "Decision Tree": {
        "module": "sklearn.tree",
        "class": "DecisionTreeClassifier",
        "params": {"random_state": 42},
    },
    "K-Nearest Neighbors": {
        "module": "sklearn.neighbors",
        "class": "KNeighborsClassifier",
        "params": {"n_neighbors": 5},
    },
    "Naive Bayes": {
        "module": "sklearn.naive_bayes",
        "class": "GaussianNB",
        "params": {},
    },
}

"""
Quick inference test — train a model on the full dataset and show
recommendations for a custom RIASEC profile.

Usage:
    python3 scripts/test_inference.py

Edit the RIASEC scores and MODEL below to test different inputs.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_training_records
from src.models import MODEL_REGISTRY

# ============================================================
# Configure your test here
# ============================================================

# Your RIASEC scores (0.0 to 1.0)
RIASEC_PROFILE = {
    "Realistic":     0.2,
    "Investigative": 0.8,
    "Artistic":      0.3,
    "Social":        0.9,
    "Enterprising":  0.4,
    "Conventional":  0.2,
}

# Model to use: logistic_regression | random_forest | gradient_boosting | heuristic | knn
MODEL = "random_forest"

# How many job recommendations to return
TOP_N_JOBS = 5

# ============================================================

TRAIN_PATH = "docs/data/master_careers_riasec_categories.csv"
ONET_PATH  = "docs/data/master_careers_riasec_categories.csv"

x_features = list(RIASEC_PROFILE.keys())

print(f"Loading training data from {TRAIN_PATH}...")
train_records = load_training_records(TRAIN_PATH)

print(f"Training {MODEL}...")
model = MODEL_REGISTRY[MODEL](
    x_features=x_features,
    y_feature="Career Category",
    parameters={},
    top_n_jobs=TOP_N_JOBS,
    top_n_categories=3,
)

train_df = pd.DataFrame([
    {col: getattr(r, col.lower().replace(" ", "_")) for col in x_features} | {"Career Category": r.career_category}
    for r in train_records
])

model.train(train_df[x_features], train_df["Career Category"])

print(f"\nLoading O*NET database from {ONET_PATH}...")
onet_db = pd.read_csv(ONET_PATH)

print(f"\nRIASEC Profile: {RIASEC_PROFILE}")
print(f"\nTop {TOP_N_JOBS} recommendations:\n")

X_input = pd.DataFrame([RIASEC_PROFILE])
results = model.test(X_input, onet_db)
print(results.to_string(index=False))

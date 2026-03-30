"""
File: baseline.py
Path: src/models/baseline.py

Purpose:
  Legacy procedural Random Forest recommendation helpers retained for notebook
  compatibility. This module loads serialized artifacts and ranks O*NET jobs
  using category prediction + cosine similarity.

Original Author(s):
  - Nate G.
  - AI Assistant

AI Tools Used:
  - AI Assistant - documentation alignment

Editors:
  - AI Assistant (2026-03-30) — Updated file header/docstrings to template format

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-03-30

Assumptions & Constraints:
  - Expects a trained Random Forest model with `predict_proba`
  - Expects O*NET CSV schema with career category and RIASEC columns
  - Retained as legacy compatibility layer; class-based models are preferred

Related Docs:
  - docs/src/models/models.md
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Any


def load_pipeline_artifacts(model_path: str, db_path: str) -> Tuple[Any, pd.DataFrame]:
    """
    Name: load_pipeline_artifacts

    Purpose:
      Load the trained Random Forest model artifact and O*NET career database.

    Inputs:
      - model_path: str — filepath to saved model artifact (.joblib)
      - db_path: str — filepath to mapped O*NET CSV database

    Outputs:
      - Tuple[Any, pd.DataFrame] — (loaded_model, career_dataframe)

    Raises / Errors:
      - FileNotFoundError: if either artifact path is not found

    Notes:
      - Uses joblib for model deserialization
    """
    try:
        rf_model = joblib.load(model_path)
        onet_db = pd.read_csv(db_path)
        return rf_model, onet_db
    except FileNotFoundError as e:
        print(f"❌ Error loading artifacts: {e}")
        raise


def get_job_recommendations(
    student_vector: pd.DataFrame,
    rf_model: Any,
    onet_db: pd.DataFrame,
    top_n_categories: int = 3,
    top_n_jobs: int = 3,
) -> pd.DataFrame:
    """
    Name: get_job_recommendations

    Purpose:
      Generate job recommendations via two-stage retrieval:
      (1) predict top-N categories with Random Forest,
      (2) rank jobs inside those categories by cosine similarity.

    Inputs:
      - student_vector: pd.DataFrame — 1-row RIASEC feature vector
      - rf_model: Any — trained classifier implementing `predict_proba`
      - onet_db: pd.DataFrame — O*NET careers dataset
      - top_n_categories: int — number of predicted categories to keep
      - top_n_jobs: int — number of ranked jobs to return

    Outputs:
      - pd.DataFrame — columns [Title, Career Category, Match_Score]

    Raises / Errors:
      - KeyError: if required onet_db columns are missing

    Notes:
      - Column names are aligned to the legacy notebook pipeline format
      - Returns empty DataFrame if category filtering yields no candidates
    """
    # 1. Predict Category Probabilities
    # Probabilities allow us to pool the most likely "sections" of the job library
    probs = rf_model.predict_proba(student_vector)[0]

    # Get the names of the Top N highest probability categories
    top_indices = np.argsort(probs)[-top_n_categories:][::-1]
    top_categories = rf_model.classes_[top_indices]

    # 2. Filter the O*NET database for candidate generation
    filtered_jobs = onet_db[onet_db["Career Category"].isin(top_categories)].copy()

    if filtered_jobs.empty:
        return pd.DataFrame()

    # 3. Calculate Micro-Ranking via Cosine Similarity
    # Note: Column names must match the O*NET CSV structure
    riasec_cols = [
        "Realistic",
        "Investigative",
        "Artistic",
        "Social",
        "Enterprising",
        "Conventional",
    ]

    job_vectors = filtered_jobs[riasec_cols].values
    student_array = student_vector.values  # Converts DF to NumPy for the math

    similarities = cosine_similarity(student_array, job_vectors)[0]
    filtered_jobs["Match_Score"] = similarities

    # 4. Final Ranking
    best_jobs = filtered_jobs.sort_values(by="Match_Score", ascending=False).head(top_n_jobs)

    return best_jobs[["Title", "Career Category", "Match_Score"]]

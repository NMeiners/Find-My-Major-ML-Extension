"""
File: heuristic.py
Path: src/models/heuristic.py

Purpose:
  Implements a rule-based heuristic model that requires no training.
  Ranks all jobs in onet_db by cosine similarity to the student's feature vector
  and returns the top-N results. No category filtering is applied because the model
  has no classifier to map feature names to category labels.
  Serves as the simplest possible baseline for comparison.

Original Author(s):
  - Angela Fleenor
  - AI Assistant (Claude Sonnet 4.6)

AI Tools Used:
  - Claude Sonnet 4.6 - Code generation and documentation

Editors:
  - AI Assistant (2026-03-20) — Initial implementation

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-03-20

Assumptions & Constraints:
  - onet_db must contain a column matching each name in x_features
  - onet_db must contain a "Career Category" column and a "Title" column
  - x_features must be non-empty

Related Docs:
  - docs/src/models.md
"""

import pandas as pd
from src.models.base import BaseModel
from src.models.ranking import rank_jobs

# Column name for career categories in the O*NET database
_CATEGORY_COL = "Career Category"


class HeuristicModel(BaseModel):
    """
    Name: HeuristicModel

    Purpose:
      Rule-based model that ranks all jobs in onet_db by cosine similarity to the
      student's feature vector and returns the top-N results. No training required.
      Useful as a sanity-check baseline and interpretability reference.

    Inputs:
      - x_features: list[str] — feature column names from config
      - y_feature: str — target label column (unused; kept for interface consistency)
      - parameters: dict — unused for this model; accepted for interface consistency
      - top_n_jobs: int — number of job recommendations to return
      - top_n_categories: int — unused; no category filtering applied

    Outputs:
      - pd.DataFrame with columns [Title, Career Category, Match_Score] and top_n_jobs rows

    Raises / Errors:
      - KeyError: if onet_db is missing a required column

    Notes:
      - train() is a no-op since this model uses no learned parameters.
      - Ranks across all categories because there is no classifier to map feature names
        to category labels — passing all unique categories to rank_jobs achieves this.
    """

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Name: train

        Purpose:
          No-op. HeuristicModel is rule-based and requires no training.

        Inputs:
          - X_train: pd.DataFrame — ignored
          - y_train: pd.Series — ignored

        Outputs:
          - None

        Raises / Errors:
          - None

        Notes:
          - Exists solely to satisfy the BaseModel interface.
        """
        pass

    def test(self, X_test: pd.DataFrame, onet_db: pd.DataFrame) -> pd.DataFrame:
        """
        Name: test

        Purpose:
          For each row in X_test, find the feature with the highest score,
          use it as the predicted career category, and return top-N ranked jobs.

        Inputs:
          - X_test: pd.DataFrame — test features; must contain all columns in self.x_features
          - onet_db: pd.DataFrame — career database with Career Category and feature columns

        Outputs:
          - pd.DataFrame with columns [Title, Career Category, Match_Score] and self.top_n_jobs rows

        Raises / Errors:
          - KeyError: if onet_db or X_test is missing required feature columns

        Notes:
          - Designed for single-row inference (first row used if multiple rows provided).
          - Passes all unique category values so rank_jobs includes the full onet_db.
        """
        student_vector = X_test[self.x_features].iloc[[0]]

        # Include all categories — the heuristic has no classifier to predict a subset
        all_categories = onet_db[_CATEGORY_COL].unique().tolist()

        return rank_jobs(
            student_vector=student_vector,
            onet_db=onet_db,
            feature_cols=self.x_features,
            top_n_categories=len(all_categories),
            top_n_jobs=self.top_n_jobs,
            category_col=_CATEGORY_COL,
            predicted_categories=all_categories,
        )

    def get_name(self) -> str:
        """
        Name: get_name

        Purpose:
          Return the config key identifier for this model.

        Inputs:
          - None

        Outputs:
          - str — "heuristic"

        Raises / Errors:
          - None

        Notes:
          - Must match the MODEL_REGISTRY key in __init__.py.
        """
        return "heuristic"

"""
File: random_forest.py
Path: src/models/random_forest.py

Purpose:
  Implements a Random Forest classifier for RIASEC-based career category prediction.
  Refactors the logic from baseline.py into a class-based interface that inherits
  from BaseModel. Uses predict_proba to select top career categories, then ranks
  specific jobs via cosine similarity through ranking.py.

Original Author(s):
  - Angela Fleenor
  - AI Assistant (Claude Sonnet 4.6)

AI Tools Used:
  - Claude Sonnet 4.6 - Code generation and documentation

Editors:
  - AI Assistant (2026-03-20) — Initial implementation; logic refactored from baseline.py

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-03-20

Assumptions & Constraints:
  - train() must be called before test()
  - x_features must match column names in both X_train and X_test
  - parameters dict may include any valid sklearn RandomForestClassifier kwargs

Related Docs:
  - docs/src/models.md
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseModel
from src.models.ranking import rank_jobs

_CATEGORY_COL = "Career Category"


class RandomForestModel(BaseModel):
    """
    Name: RandomForestModel

    Purpose:
      Wraps sklearn's RandomForestClassifier to predict career category probabilities
      from RIASEC feature vectors. Selects top categories by probability mass, then
      ranks specific jobs within those categories using cosine similarity.
      Refactored from the procedural approach in baseline.py.

    Inputs:
      - x_features: list[str] — input feature column names from config
      - y_feature: str — target label column from config
      - parameters: dict — sklearn RandomForestClassifier kwargs (e.g. n_estimators, max_depth)
      - top_n_jobs: int — number of job recommendations to return
      - top_n_categories: int — number of top career categories to consider before ranking

    Outputs:
      - pd.DataFrame with columns [Title, Career Category, Match_Score] and top_n_jobs rows

    Raises / Errors:
      - RuntimeError: if test() is called before train()

    Notes:
      - Hyperparameters are unpacked from self.parameters and passed directly to sklearn.
      - This replaces the get_job_recommendations() function in baseline.py for new code.
        baseline.py is kept for notebook compatibility only.
    """

    def __init__(self, x_features, y_feature, parameters, top_n_jobs, top_n_categories):
        super().__init__(x_features, y_feature, parameters, top_n_jobs, top_n_categories)
        self._model = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Name: train

        Purpose:
          Fit a Random Forest classifier on the training data.

        Inputs:
          - X_train: pd.DataFrame — training features; must contain all columns in self.x_features
          - y_train: pd.Series — career category labels

        Outputs:
          - None (model state updated in place)

        Raises / Errors:
          - KeyError: if X_train is missing a required feature column

        Notes:
          - Hyperparameters unpacked from self.parameters; any valid RandomForestClassifier kwarg is accepted.
        """
        self._model = RandomForestClassifier(**self.parameters)
        self._model.fit(X_train[self.x_features], y_train)

    def test(self, X_test: pd.DataFrame, onet_db: pd.DataFrame) -> pd.DataFrame:
        """
        Name: test

        Purpose:
          Predict the top career categories for the student's feature vector,
          then return the top-N ranked jobs from those categories.

        Inputs:
          - X_test: pd.DataFrame — test features; must contain all columns in self.x_features
          - onet_db: pd.DataFrame — career database for job ranking

        Outputs:
          - pd.DataFrame with columns [Title, Career Category, Match_Score] and self.top_n_jobs rows

        Raises / Errors:
          - RuntimeError: if called before train()
          - KeyError: if X_test or onet_db is missing required columns

        Notes:
          - Uses predict_proba to select top self.top_n_categories categories by probability.
        """
        if self._model is None:
            raise RuntimeError("train() must be called before test().")

        student_vector = X_test[self.x_features].iloc[[0]]

        probs = self._model.predict_proba(student_vector)[0]
        top_indices = np.argsort(probs)[-self.top_n_categories:][::-1]
        top_categories = list(self._model.classes_[top_indices])

        return rank_jobs(
            student_vector=student_vector,
            onet_db=onet_db,
            feature_cols=self.x_features,
            top_n_categories=self.top_n_categories,
            top_n_jobs=self.top_n_jobs,
            category_col=_CATEGORY_COL,
            predicted_categories=top_categories,
        )

    def get_name(self) -> str:
        """
        Name: get_name

        Purpose:
          Return the config key identifier for this model.

        Inputs:
          - None

        Outputs:
          - str — "random_forest"

        Raises / Errors:
          - None

        Notes:
          - Must match the MODEL_REGISTRY key in __init__.py.
        """
        return "random_forest"

"""
File: knn.py
Path: src/models/knn.py

Purpose:
  Implements a K-Nearest Neighbors classifier for RIASEC-based career category
  prediction. Finds the K most similar training examples to the student's feature
  vector and uses their career labels to predict top categories. Provides an
  interpretable, instance-based alternative to tree and regression models.

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
  - train() must be called before test()
  - x_features must match column names in both X_train and X_test
  - parameters dict may include any valid sklearn KNeighborsClassifier kwargs (e.g. n_neighbors)

Related Docs:
  - docs/src/models.md
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from src.models.base import BaseModel
from src.models.ranking import rank_jobs

_CATEGORY_COL = "Career Category"


class KNNModel(BaseModel):
    """
    Name: KNNModel

    Purpose:
      Wraps sklearn's KNeighborsClassifier to predict career category probabilities
      from RIASEC feature vectors. Finds K nearest neighbors in the training set
      and votes on the most likely career categories. Intuition: find students
      with similar RIASEC profiles and recommend what worked for them.
      All hyperparameters come from the experiment config.

    Inputs:
      - x_features: list[str] — input feature column names from config
      - y_feature: str — target label column from config
      - parameters: dict — sklearn KNeighborsClassifier kwargs (e.g. n_neighbors, metric)
      - top_n_jobs: int — number of job recommendations to return
      - top_n_categories: int — number of top career categories to consider before ranking

    Outputs:
      - pd.DataFrame with columns [Title, Career Category, Match_Score] and top_n_jobs rows

    Raises / Errors:
      - RuntimeError: if test() is called before train()

    Notes:
      - Hyperparameters are unpacked from self.parameters and passed directly to sklearn.
      - Uses predict_proba to get soft category votes rather than a hard single prediction.
    """

    def __init__(self, x_features, y_feature, parameters, top_n_jobs, top_n_categories):
        super().__init__(x_features, y_feature, parameters, top_n_jobs, top_n_categories)
        self._model = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Name: train

        Purpose:
          Fit a KNN classifier on the training data.

        Inputs:
          - X_train: pd.DataFrame — training features; must contain all columns in self.x_features
          - y_train: pd.Series — career category labels

        Outputs:
          - None (model state updated in place)

        Raises / Errors:
          - KeyError: if X_train is missing a required feature column

        Notes:
          - KNN does not learn parameters in the traditional sense — it stores the training
            data and computes distances at test time.
          - Hyperparameters unpacked from self.parameters.
        """
        self._model = KNeighborsClassifier(**self.parameters)
        self._model.fit(X_train[self.x_features], y_train)

    def test(self, X_test: pd.DataFrame, onet_db: pd.DataFrame) -> pd.DataFrame:
        """
        Name: test

        Purpose:
          Predict the top career categories for the student's feature vector
          using nearest-neighbor voting, then return the top-N ranked jobs.

        Inputs:
          - X_test: pd.DataFrame — test features; must contain all columns in self.x_features
          - onet_db: pd.DataFrame — career database for job ranking

        Outputs:
          - pd.DataFrame with columns [Title, Career Category, Match_Score] and self.top_n_jobs rows

        Raises / Errors:
          - RuntimeError: if called before train()
          - KeyError: if X_test or onet_db is missing required columns

        Notes:
          - Uses predict_proba to select top self.top_n_categories categories by vote share.
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
          - str — "knn"

        Raises / Errors:
          - None

        Notes:
          - Must match the MODEL_REGISTRY key in __init__.py.
        """
        return "knn"

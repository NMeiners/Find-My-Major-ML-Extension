"""
File: inference.py
Path: src/models/inference.py

Purpose:
  Provides the run_inference() entry point for the Export/Inference module.
  Accepts any trained BaseModel instance and returns job recommendations
  without requiring the caller to know which specific model is being used.

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
  - model must already be trained (train() called) before passing to run_inference()
  - onet_db must contain all columns required by the model's x_features

Related Docs:
  - docs/src/models.md
"""

import pandas as pd
from src.models.base import BaseModel


def run_inference(
    model: BaseModel,
    X_test: pd.DataFrame,
    onet_db: pd.DataFrame,
) -> pd.DataFrame:
    """
    Name: run_inference

    Purpose:
      Runs a trained model's test() method and returns job recommendations.
      Acts as a clean entry point for the Export/Inference module so that
      downstream code works with any model without knowing its type.

    Inputs:
      - model: BaseModel — any trained model instance inheriting from BaseModel
      - X_test: pd.DataFrame — student feature vector; must contain model's x_features columns
      - onet_db: pd.DataFrame — career database for ranking; loaded from config onet_db_path

    Outputs:
      - pd.DataFrame with columns [Title, Career Category, Match_Score]
        and model.top_n_jobs rows

    Raises / Errors:
      - RuntimeError: if model has not been trained
      - KeyError: if X_test or onet_db is missing required columns

    Notes:
      - top_n_jobs is stored on the model from config — no extra argument needed here.
      - This function does not load data or config; the caller is responsible for that.
    """
    return model.test(X_test, onet_db)

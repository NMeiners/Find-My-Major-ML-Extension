"""
File: preprocess.py
Path: src/data/preprocess.py

Purpose:
  Converts TrainingRecord objects into feature matrices and label vectors
  ready for model training. Computes derived features (riasec_variance)
  and provides extraction utilities that align with the sklearn interface
  used in the model layer.

Original Author(s):
  - Claude Code

AI Tools Used:
  - Claude Code - initial implementation

Editors:
  - Claude Code (2026-03-16) — initial implementation

Last Editor:
  - Claude Code

Last Edit Date:
  2026-03-16

Assumptions & Constraints:
  - Input is a list of TrainingRecord objects (output of loader.py)
  - Feature columns are the 6 normalized RIASEC scores + riasec_variance
  - Label column is career_category
  - Does not mutate input records

Related Docs:
  - docs/src/data.md
  - docs/data/pipeline_architecture.md
"""

from __future__ import annotations

import pandas as pd

from src.data.schemas import TrainingRecord

# Canonical RIASEC field names as they appear on TrainingRecord
RIASEC_FIELDS: tuple[str, ...] = (
    "realistic",
    "investigative",
    "artistic",
    "social",
    "enterprising",
    "conventional",
)

# All feature columns produced by preprocess (RIASEC + derived)
FEATURE_COLS: tuple[str, ...] = RIASEC_FIELDS + ("riasec_variance",)


def records_to_dataframe(records: list[TrainingRecord]) -> pd.DataFrame:
    """
    Name: records_to_dataframe

    Purpose:
      Converts a list of TrainingRecord objects into a DataFrame containing
      the 6 RIASEC scores, the computed riasec_variance feature, and the
      career_category label column.

    Inputs:
      - records: list[TrainingRecord] — validated training records

    Outputs:
      - pd.DataFrame — columns: realistic, investigative, artistic, social,
        enterprising, conventional, riasec_variance, career_category

    Raises / Errors:
      - N/A

    Notes:
      - riasec_variance is the row-wise variance across the 6 RIASEC scores
        and captures how focused vs. spread a respondent's interests are
    """
    rows = [
        {
            "realistic": r.realistic,
            "investigative": r.investigative,
            "artistic": r.artistic,
            "social": r.social,
            "enterprising": r.enterprising,
            "conventional": r.conventional,
            "career_category": r.career_category,
        }
        for r in records
    ]
    if not rows:
        cols = list(RIASEC_FIELDS) + ["career_category", "riasec_variance"]
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    df["riasec_variance"] = df[list(RIASEC_FIELDS)].var(axis=1)
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Name: extract_features

    Purpose:
      Extracts the feature matrix from a preprocessed DataFrame. Returns
      only the columns used as model inputs.

    Inputs:
      - df: pd.DataFrame — output of records_to_dataframe()

    Outputs:
      - pd.DataFrame — columns: realistic, investigative, artistic, social,
        enterprising, conventional, riasec_variance

    Raises / Errors:
      - KeyError: if expected feature columns are missing from df

    Notes:
      - Returned DataFrame is compatible with sklearn estimator .fit() / .predict()
    """
    return df[list(FEATURE_COLS)]


def extract_labels(df: pd.DataFrame) -> pd.Series:
    """
    Name: extract_labels

    Purpose:
      Extracts the label vector from a preprocessed DataFrame.

    Inputs:
      - df: pd.DataFrame — output of records_to_dataframe()

    Outputs:
      - pd.Series — career_category values

    Raises / Errors:
      - KeyError: if career_category column is missing from df

    Notes:
      - Returned Series is compatible with sklearn estimator .fit()
    """
    return df["career_category"]


def build_training_matrix(
    records: list[TrainingRecord],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Name: build_training_matrix

    Purpose:
      Full pipeline convenience function: converts TrainingRecord objects
      directly into a feature matrix X and label vector y ready for sklearn.

    Inputs:
      - records: list[TrainingRecord] — validated training records

    Outputs:
      - tuple[pd.DataFrame, pd.Series] — (X, y) where X is the feature
        matrix and y is the career_category label vector

    Raises / Errors:
      - N/A

    Notes:
      - Equivalent to calling records_to_dataframe → extract_features /
        extract_labels in sequence
    """
    df = records_to_dataframe(records)
    return extract_features(df), extract_labels(df)

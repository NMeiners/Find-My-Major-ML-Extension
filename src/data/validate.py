"""
File: validate.py
Path: src/data/validate.py

Purpose:
  Provides validation functions for training and career profile DataFrames.
  Checks column presence, null values, RIASEC score ranges, category
  membership, and duplicate identifiers before data is used in training.

Original Author(s):
  - Claude Code

AI Tools Used:
  - Claude Code - initial implementation

Editors:
  - Claude Code (2026-03-14) — initial implementation

Last Editor:
  - Claude Code

Last Edit Date:
  2026-03-14

Assumptions & Constraints:
  - Validation runs on raw column names (before loader renames them)
  - Category alias resolution (CATEGORY_ALIASES) is the loader's responsibility
  - Training data validator accepts both canonical categories and known aliases
  - Career data validator requires canonical categories only
  - Does not modify the DataFrame; returns a list of error strings

Related Docs:
  - docs/data/data_governance.md
  - docs/data/pipeline_architecture.md
"""

from __future__ import annotations

import pandas as pd

from src.data.schemas import BROAD_CATEGORIES, CATEGORY_ALIASES

# Raw column names as they appear in the processed Excel files
_TRAINING_RIASEC_COLS: tuple[str, ...] = (
    "R normalized",
    "I normalized",
    "A normalized",
    "S normalized",
    "E normalized",
    "C normalized",
)
_TRAINING_REQUIRED_COLS: tuple[str, ...] = _TRAINING_RIASEC_COLS + ("Broad_Category",)

_CAREER_RIASEC_COLS: tuple[str, ...] = (
    "Realistic",
    "Investigative",
    "Artistic",
    "Social",
    "Enterprising",
    "Conventional",
)
_CAREER_REQUIRED_COLS: tuple[str, ...] = _CAREER_RIASEC_COLS + (
    "O*NET-SOC Code",
    "Title",
    "Broad_Category",
)

# Training data may contain alias values before loader normalization
_VALID_TRAINING_CATEGORIES: frozenset[str] = frozenset(BROAD_CATEGORIES) | frozenset(
    CATEGORY_ALIASES.keys()
)


def validate_training_dataframe(df: pd.DataFrame) -> list[str]:
    """
    Name: validate_training_dataframe

    Purpose:
      Validates a DataFrame of training records loaded from the processed
      Kaggle RIASEC Excel file. Returns a list of human-readable error
      messages; an empty list means the DataFrame passed all checks.

    Inputs:
      - df: pd.DataFrame — raw training DataFrame with original column names

    Outputs:
      - list[str] — error messages; empty if all checks pass

    Raises / Errors:
      - N/A (errors are collected and returned, not raised)

    Notes:
      - Accepts both canonical BROAD_CATEGORIES and CATEGORY_ALIASES values
        because alias resolution happens in the loader, not here
      - Returns early if required columns are missing (remaining checks
        would produce misleading errors)
    """
    errors: list[str] = []

    missing = [c for c in _TRAINING_REQUIRED_COLS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return errors

    for col in _TRAINING_REQUIRED_COLS:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} null value(s)")

    for col in _TRAINING_RIASEC_COLS:
        out_of_range = int(((df[col] < 0.0) | (df[col] > 1.0)).sum())
        if out_of_range > 0:
            errors.append(
                f"Column '{col}' has {out_of_range} value(s) outside [0.0, 1.0]"
            )

    invalid_mask = ~df["Broad_Category"].isin(_VALID_TRAINING_CATEGORIES)
    if invalid_mask.any():
        bad = sorted(df.loc[invalid_mask, "Broad_Category"].unique().tolist())
        errors.append(f"Unknown Broad_Category value(s) in training data: {bad}")

    return errors


def validate_career_dataframe(df: pd.DataFrame) -> list[str]:
    """
    Name: validate_career_dataframe

    Purpose:
      Validates a DataFrame of career profiles loaded from the master careers
      Excel file. Returns a list of human-readable error messages; an empty
      list means the DataFrame passed all checks.

    Inputs:
      - df: pd.DataFrame — career profiles DataFrame with original column names

    Outputs:
      - list[str] — error messages; empty if all checks pass

    Raises / Errors:
      - N/A (errors are collected and returned, not raised)

    Notes:
      - Requires canonical BROAD_CATEGORIES only (no aliases expected in the
        careers dataset)
      - Checks for duplicate O*NET-SOC Codes as an integrity invariant
      - Returns early if required columns are missing
    """
    errors: list[str] = []

    missing = [c for c in _CAREER_REQUIRED_COLS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return errors

    for col in _CAREER_REQUIRED_COLS:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} null value(s)")

    for col in _CAREER_RIASEC_COLS:
        out_of_range = int(((df[col] < 0.0) | (df[col] > 1.0)).sum())
        if out_of_range > 0:
            errors.append(
                f"Column '{col}' has {out_of_range} value(s) outside [0.0, 1.0]"
            )

    duplicate_count = int(df["O*NET-SOC Code"].duplicated().sum())
    if duplicate_count > 0:
        errors.append(f"Found {duplicate_count} duplicate O*NET-SOC Code(s)")

    invalid_mask = ~df["Broad_Category"].isin(BROAD_CATEGORIES)
    if invalid_mask.any():
        bad = sorted(df.loc[invalid_mask, "Broad_Category"].unique().tolist())
        errors.append(f"Unknown Broad_Category value(s) in career data: {bad}")

    return errors

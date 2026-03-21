"""
File: loader.py
Path: src/data/loader.py

Purpose:
  Loads the processed Kaggle RIASEC training data and the master O*NET
  careers dataset from CSV files. Validates raw DataFrames, applies
  category alias resolution, renames columns to canonical names, and
  constructs typed TrainingRecord and CareerProfile objects. Also provides
  deterministic stratified train/val/test splits.

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
  - Training CSV has columns: R normalized, I normalized, A normalized,
    S normalized, E normalized, C normalized, Career Category
  - Careers CSV has columns: O*NET-SOC Code, Title, Realistic, Investigative,
    Artistic, Social, Enterprising, Conventional, Career Category
  - Validation runs before alias resolution and column renaming
  - Default split is 70/15/15 (train/val/test) stratified by career_category
  - random_state=42 ensures reproducibility

Related Docs:
  - docs/src/data.md
  - docs/data/data_governance.md
  - docs/data/pipeline_architecture.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.schemas import CATEGORY_ALIASES, CareerProfile, TrainingRecord
from src.data.validate import validate_career_dataframe, validate_training_dataframe

# Default data file locations (relative to repo root)
TRAINING_CSV = Path("docs/data/Kaggle_Cleaned_Mapped_Categories.csv")
CAREERS_CSV = Path("docs/data/master_careers_riasec_categories.csv")

# Maps raw training CSV column names to TrainingRecord field names
_TRAINING_COL_MAP: dict[str, str] = {
    "R normalized": "realistic",
    "I normalized": "investigative",
    "A normalized": "artistic",
    "S normalized": "social",
    "E normalized": "enterprising",
    "C normalized": "conventional",
    "Career Category": "career_category",
}

# Maps raw careers CSV column names to CareerProfile field names
_CAREER_COL_MAP: dict[str, str] = {
    "O*NET-SOC Code": "code",
    "Title": "title",
    "Realistic": "realistic",
    "Investigative": "investigative",
    "Artistic": "artistic",
    "Social": "social",
    "Enterprising": "enterprising",
    "Conventional": "conventional",
    "Career Category": "career_category",
}


def load_training_records(path: Path | str = TRAINING_CSV) -> list[TrainingRecord]:
    """
    Name: load_training_records

    Purpose:
      Loads the processed Kaggle RIASEC CSV, validates it, resolves category
      aliases, and returns a list of TrainingRecord objects.

    Inputs:
      - path: Path | str — path to the training CSV file

    Outputs:
      - list[TrainingRecord] — one record per row in the CSV

    Raises / Errors:
      - FileNotFoundError: if the CSV file does not exist
      - ValueError: if the DataFrame fails validation

    Notes:
      - Rows with null values in required columns are dropped before
        TrainingRecord construction (validator will have already flagged them)
      - Category aliases (e.g. "Engineering & Architecture") are resolved
        to canonical BROAD_CATEGORIES values before construction
    """
    df = pd.read_csv(path)

    errors = validate_training_dataframe(df)
    if errors:
        raise ValueError(f"Training data validation failed:\n" + "\n".join(errors))

    df["Career Category"] = df["Career Category"].replace(CATEGORY_ALIASES)

    required_cols = list(_TRAINING_COL_MAP.keys())
    df = df[required_cols].dropna()
    df = df.rename(columns=_TRAINING_COL_MAP)

    return [TrainingRecord(**row) for row in df.to_dict(orient="records")]


def load_career_profiles(path: Path | str = CAREERS_CSV) -> list[CareerProfile]:
    """
    Name: load_career_profiles

    Purpose:
      Loads the master O*NET careers CSV, validates it, and returns a list
      of CareerProfile objects.

    Inputs:
      - path: Path | str — path to the careers CSV file

    Outputs:
      - list[CareerProfile] — one profile per row in the CSV

    Raises / Errors:
      - FileNotFoundError: if the CSV file does not exist
      - ValueError: if the DataFrame fails validation

    Notes:
      - Career data is expected to use canonical category names already;
        no alias resolution is applied
    """
    df = pd.read_csv(path)

    errors = validate_career_dataframe(df)
    if errors:
        raise ValueError(f"Career data validation failed:\n" + "\n".join(errors))

    required_cols = list(_CAREER_COL_MAP.keys())
    df = df[required_cols].dropna()
    df = df.rename(columns=_CAREER_COL_MAP)

    return [CareerProfile(**row) for row in df.to_dict(orient="records")]


def split_training_records(
    records: list[TrainingRecord],
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[list[TrainingRecord], list[TrainingRecord], list[TrainingRecord]]:
    """
    Name: split_training_records

    Purpose:
      Splits a list of TrainingRecord objects into train, validation, and
      test sets using stratified sampling to preserve category distribution.

    Inputs:
      - records: list[TrainingRecord] — full dataset to split
      - val_size: float — fraction for validation set (default 0.15)
      - test_size: float — fraction for test set (default 0.15)
      - random_state: int — seed for reproducibility (default 42)

    Outputs:
      - tuple[list[TrainingRecord], list[TrainingRecord], list[TrainingRecord]]
        — (train, val, test)

    Raises / Errors:
      - ValueError: if val_size + test_size >= 1.0

    Notes:
      - Stratified by career_category to preserve class distribution
      - Default 70/15/15 split
    """
    if val_size + test_size >= 1.0:
        raise ValueError(
            f"val_size + test_size must be < 1.0, got {val_size + test_size}"
        )

    labels = [r.career_category for r in records]

    train_val, test = train_test_split(
        records,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    val_fraction = val_size / (1.0 - test_size)
    train_val_labels = [r.career_category for r in train_val]

    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=train_val_labels,
        random_state=random_state,
    )

    return train, val, test

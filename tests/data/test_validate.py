"""
File: test_validate.py
Path: tests/data/test_validate.py

Purpose:
  Unit tests for src/data/validate.py. Covers validate_training_dataframe
  and validate_career_dataframe using synthetic DataFrames — no file I/O.

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
  - Tests run with pytest from the repository root
  - No external data files required (uses synthetic DataFrames)

Related Docs:
  - docs/data/data_governance.md
"""

import pandas as pd
import pytest

from src.data.validate import (
    validate_career_dataframe,
    validate_training_dataframe,
)

# ---------------------------------------------------------------------------
# Synthetic DataFrame builders
# ---------------------------------------------------------------------------


def _make_training_df(**overrides) -> pd.DataFrame:
    """Return a minimal valid training DataFrame with one row."""
    row = {
        "R normalized": 0.5,
        "I normalized": 0.6,
        "A normalized": 0.3,
        "S normalized": 0.7,
        "E normalized": 0.4,
        "C normalized": 0.2,
        "Broad_Category": "Education",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _make_career_df(**overrides) -> pd.DataFrame:
    """Return a minimal valid career DataFrame with one row."""
    row = {
        "O*NET-SOC Code": "11-1011.00",
        "Title": "Chief Executives",
        "Realistic": 0.18,
        "Investigative": 0.44,
        "Artistic": 0.31,
        "Social": 0.51,
        "Enterprising": 0.99,
        "Conventional": 0.71,
        "Broad_Category": "Business & Finance",
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# validate_training_dataframe
# ---------------------------------------------------------------------------


class TestValidateTrainingDataframe:
    """Tests for validate_training_dataframe."""

    def test_valid_dataframe_returns_no_errors(self):
        assert validate_training_dataframe(_make_training_df()) == []

    def test_accepts_canonical_category(self):
        df = _make_training_df(**{"Broad_Category": "Healthcare & Medicine"})
        assert validate_training_dataframe(df) == []

    def test_accepts_category_alias(self):
        # "Engineering & Architecture" is a known alias; loader maps it later
        df = _make_training_df(**{"Broad_Category": "Engineering & Architecture"})
        assert validate_training_dataframe(df) == []

    def test_missing_column_returns_error(self):
        df = _make_training_df()
        df = df.drop(columns=["R normalized"])
        errors = validate_training_dataframe(df)
        assert any("Missing required columns" in e for e in errors)

    def test_null_riasec_value_returns_error(self):
        df = _make_training_df()
        df.loc[0, "S normalized"] = None
        errors = validate_training_dataframe(df)
        assert any("S normalized" in e and "null" in e for e in errors)

    def test_null_category_returns_error(self):
        df = _make_training_df()
        df.loc[0, "Broad_Category"] = None
        errors = validate_training_dataframe(df)
        assert any("Broad_Category" in e and "null" in e for e in errors)

    def test_riasec_below_zero_returns_error(self):
        df = _make_training_df(**{"I normalized": -0.01})
        errors = validate_training_dataframe(df)
        assert any("I normalized" in e and "outside [0.0, 1.0]" in e for e in errors)

    def test_riasec_above_one_returns_error(self):
        df = _make_training_df(**{"C normalized": 1.5})
        errors = validate_training_dataframe(df)
        assert any("C normalized" in e and "outside [0.0, 1.0]" in e for e in errors)

    def test_unknown_category_returns_error(self):
        df = _make_training_df(**{"Broad_Category": "Underwater Basket Weaving"})
        errors = validate_training_dataframe(df)
        assert any("Unknown Broad_Category" in e for e in errors)

    def test_multiple_errors_all_reported(self):
        df = _make_training_df(**{"R normalized": -1.0, "Broad_Category": "BadCat"})
        errors = validate_training_dataframe(df)
        assert len(errors) >= 2

    def test_returns_early_when_columns_missing(self):
        # If required columns are absent, only the missing-columns error appears
        df = pd.DataFrame({"unrelated_col": [1]})
        errors = validate_training_dataframe(df)
        assert len(errors) == 1
        assert "Missing required columns" in errors[0]


# ---------------------------------------------------------------------------
# validate_career_dataframe
# ---------------------------------------------------------------------------


class TestValidateCareerDataframe:
    """Tests for validate_career_dataframe."""

    def test_valid_dataframe_returns_no_errors(self):
        assert validate_career_dataframe(_make_career_df()) == []

    def test_missing_column_returns_error(self):
        df = _make_career_df()
        df = df.drop(columns=["Title"])
        errors = validate_career_dataframe(df)
        assert any("Missing required columns" in e for e in errors)

    def test_null_riasec_value_returns_error(self):
        df = _make_career_df()
        df.loc[0, "Realistic"] = None
        errors = validate_career_dataframe(df)
        assert any("Realistic" in e and "null" in e for e in errors)

    def test_riasec_below_zero_returns_error(self):
        df = _make_career_df(**{"Artistic": -0.1})
        errors = validate_career_dataframe(df)
        assert any("Artistic" in e and "outside [0.0, 1.0]" in e for e in errors)

    def test_riasec_above_one_returns_error(self):
        df = _make_career_df(**{"Enterprising": 1.01})
        errors = validate_career_dataframe(df)
        assert any("Enterprising" in e and "outside [0.0, 1.0]" in e for e in errors)

    def test_duplicate_onet_code_returns_error(self):
        df = pd.concat([_make_career_df(), _make_career_df()], ignore_index=True)
        errors = validate_career_dataframe(df)
        assert any("duplicate" in e.lower() for e in errors)

    def test_unique_onet_codes_no_error(self):
        df = pd.concat(
            [_make_career_df(**{"O*NET-SOC Code": "11-1011.00"}),
             _make_career_df(**{"O*NET-SOC Code": "11-1021.00"})],
            ignore_index=True,
        )
        assert validate_career_dataframe(df) == []

    def test_unknown_category_returns_error(self):
        df = _make_career_df(**{"Broad_Category": "Engineering & Architecture"})
        errors = validate_career_dataframe(df)
        assert any("Unknown Broad_Category" in e for e in errors)

    def test_alias_not_accepted_in_career_data(self):
        # Career data must use canonical names; aliases are a training-data concern
        df = _make_career_df(**{"Broad_Category": "Engineering & Architecture"})
        errors = validate_career_dataframe(df)
        assert len(errors) > 0

    def test_returns_early_when_columns_missing(self):
        df = pd.DataFrame({"unrelated_col": [1]})
        errors = validate_career_dataframe(df)
        assert len(errors) == 1
        assert "Missing required columns" in errors[0]

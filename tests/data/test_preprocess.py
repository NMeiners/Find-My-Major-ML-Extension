"""
File: test_preprocess.py
Path: tests/data/test_preprocess.py

Purpose:
  Unit tests for src/data/preprocess.py. Covers records_to_dataframe,
  extract_features, extract_labels, and build_training_matrix.

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
  - Tests run with pytest from the repository root
  - No external data files required (uses synthetic TrainingRecords)

Related Docs:
  - docs/data/data_governance.md
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocess import (
    FEATURE_COLS,
    RIASEC_FIELDS,
    build_training_matrix,
    extract_features,
    extract_labels,
    records_to_dataframe,
)
from src.data.schemas import BROAD_CATEGORIES, TrainingRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CAT = BROAD_CATEGORIES[0]


def _make_record(
    r=0.5, i=0.6, a=0.3, s=0.7, e=0.4, c=0.2, category=_CAT
) -> TrainingRecord:
    return TrainingRecord(
        realistic=r, investigative=i, artistic=a,
        social=s, enterprising=e, conventional=c,
        career_category=category,
    )


def _make_records(n: int) -> list[TrainingRecord]:
    cats = list(BROAD_CATEGORIES)
    return [_make_record(category=cats[i % len(cats)]) for i in range(n)]


# ---------------------------------------------------------------------------
# records_to_dataframe
# ---------------------------------------------------------------------------


class TestRecordsToDataframe:

    def test_returns_dataframe(self):
        df = records_to_dataframe([_make_record()])
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_records(self):
        records = _make_records(10)
        df = records_to_dataframe(records)
        assert len(df) == 10

    def test_riasec_columns_present(self):
        df = records_to_dataframe([_make_record()])
        for col in RIASEC_FIELDS:
            assert col in df.columns

    def test_career_category_column_present(self):
        df = records_to_dataframe([_make_record()])
        assert "career_category" in df.columns

    def test_riasec_variance_column_present(self):
        df = records_to_dataframe([_make_record()])
        assert "riasec_variance" in df.columns

    def test_riasec_values_correct(self):
        df = records_to_dataframe([_make_record(r=0.1, i=0.2, a=0.3,
                                                 s=0.4, e=0.5, c=0.6)])
        assert df.loc[0, "realistic"] == pytest.approx(0.1)
        assert df.loc[0, "conventional"] == pytest.approx(0.6)

    def test_career_category_value_correct(self):
        df = records_to_dataframe([_make_record(category="Education")])
        assert df.loc[0, "career_category"] == "Education"

    def test_riasec_variance_is_non_negative(self):
        df = records_to_dataframe(_make_records(20))
        assert (df["riasec_variance"] >= 0).all()

    def test_uniform_riasec_has_zero_variance(self):
        record = _make_record(r=0.5, i=0.5, a=0.5, s=0.5, e=0.5, c=0.5)
        df = records_to_dataframe([record])
        assert df.loc[0, "riasec_variance"] == pytest.approx(0.0)

    def test_spread_riasec_has_nonzero_variance(self):
        record = _make_record(r=0.0, i=1.0, a=0.0, s=1.0, e=0.0, c=1.0)
        df = records_to_dataframe([record])
        assert df.loc[0, "riasec_variance"] > 0

    def test_empty_records_returns_empty_dataframe(self):
        df = records_to_dataframe([])
        assert len(df) == 0


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------


class TestExtractFeatures:

    def test_returns_dataframe(self):
        df = records_to_dataframe([_make_record()])
        X = extract_features(df)
        assert isinstance(X, pd.DataFrame)

    def test_contains_only_feature_columns(self):
        df = records_to_dataframe([_make_record()])
        X = extract_features(df)
        assert set(X.columns) == set(FEATURE_COLS)

    def test_career_category_not_in_features(self):
        df = records_to_dataframe([_make_record()])
        X = extract_features(df)
        assert "career_category" not in X.columns

    def test_row_count_preserved(self):
        records = _make_records(15)
        df = records_to_dataframe(records)
        X = extract_features(df)
        assert len(X) == 15

    def test_missing_column_raises(self):
        df = records_to_dataframe([_make_record()])
        df = df.drop(columns=["realistic"])
        with pytest.raises(KeyError):
            extract_features(df)


# ---------------------------------------------------------------------------
# extract_labels
# ---------------------------------------------------------------------------


class TestExtractLabels:

    def test_returns_series(self):
        df = records_to_dataframe([_make_record()])
        y = extract_labels(df)
        assert isinstance(y, pd.Series)

    def test_values_match_career_category(self):
        records = [_make_record(category=cat) for cat in BROAD_CATEGORIES]
        df = records_to_dataframe(records)
        y = extract_labels(df)
        assert list(y) == [r.career_category for r in records]

    def test_row_count_preserved(self):
        records = _make_records(15)
        df = records_to_dataframe(records)
        y = extract_labels(df)
        assert len(y) == 15

    def test_missing_column_raises(self):
        df = records_to_dataframe([_make_record()])
        df = df.drop(columns=["career_category"])
        with pytest.raises(KeyError):
            extract_labels(df)


# ---------------------------------------------------------------------------
# build_training_matrix
# ---------------------------------------------------------------------------


class TestBuildTrainingMatrix:

    def test_returns_tuple_of_dataframe_and_series(self):
        records = _make_records(10)
        X, y = build_training_matrix(records)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_x_has_correct_columns(self):
        X, _ = build_training_matrix(_make_records(10))
        assert set(X.columns) == set(FEATURE_COLS)

    def test_y_contains_career_categories(self):
        records = _make_records(10)
        _, y = build_training_matrix(records)
        assert set(y.unique()).issubset(set(BROAD_CATEGORIES))

    def test_x_and_y_same_length(self):
        records = _make_records(20)
        X, y = build_training_matrix(records)
        assert len(X) == len(y) == 20

    def test_consistent_with_individual_functions(self):
        records = _make_records(10)
        X_pipeline, y_pipeline = build_training_matrix(records)
        df = records_to_dataframe(records)
        X_manual = extract_features(df)
        y_manual = extract_labels(df)
        pd.testing.assert_frame_equal(X_pipeline.reset_index(drop=True),
                                      X_manual.reset_index(drop=True))
        pd.testing.assert_series_equal(y_pipeline.reset_index(drop=True),
                                       y_manual.reset_index(drop=True))

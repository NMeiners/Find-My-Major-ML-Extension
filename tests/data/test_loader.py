"""
File: test_loader.py
Path: tests/data/test_loader.py

Purpose:
  Unit tests for src/data/loader.py. Covers load_training_records,
  load_career_profiles, and split_training_records using temporary CSV
  files — no dependency on the real data files.

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
  - Uses tmp_path fixture to write synthetic CSVs; no real data files needed

Related Docs:
  - docs/data/data_governance.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import (
    load_career_profiles,
    load_training_records,
    split_training_records,
)
from src.data.schemas import BROAD_CATEGORIES, CareerProfile, TrainingRecord

# ---------------------------------------------------------------------------
# Synthetic CSV builders
# ---------------------------------------------------------------------------

_RIASEC_TRAINING_COLS = [
    "R normalized", "I normalized", "A normalized",
    "S normalized", "E normalized", "C normalized",
]

_RIASEC_CAREER_COLS = [
    "Realistic", "Investigative", "Artistic",
    "Social", "Enterprising", "Conventional",
]


def _write_training_csv(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_career_csv(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _training_row(category: str = "Education", **overrides) -> dict:
    row = {
        "R normalized": 0.5, "I normalized": 0.6, "A normalized": 0.3,
        "S normalized": 0.7, "E normalized": 0.4, "C normalized": 0.2,
        "Career Category": category,
        "extra_col": "ignored",
    }
    row.update(overrides)
    return row


def _career_row(
    code: str = "11-1011.00",
    title: str = "Chief Executives",
    category: str = "Business & Finance",
    **overrides,
) -> dict:
    row = {
        "O*NET-SOC Code": code,
        "Title": title,
        "Realistic": 0.18, "Investigative": 0.44, "Artistic": 0.31,
        "Social": 0.51, "Enterprising": 0.99, "Conventional": 0.71,
        "Career Category": category,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# load_training_records
# ---------------------------------------------------------------------------


class TestLoadTrainingRecords:
    """Tests for load_training_records."""

    def test_returns_training_records(self, tmp_path):
        csv = _write_training_csv(tmp_path / "train.csv", [_training_row()])
        records = load_training_records(csv)
        assert len(records) == 1
        assert isinstance(records[0], TrainingRecord)

    def test_riasec_values_loaded_correctly(self, tmp_path):
        csv = _write_training_csv(tmp_path / "train.csv", [_training_row()])
        record = load_training_records(csv)[0]
        assert record.realistic == 0.5
        assert record.investigative == 0.6
        assert record.conventional == 0.2

    def test_category_alias_resolved(self, tmp_path):
        csv = _write_training_csv(
            tmp_path / "train.csv",
            [_training_row(category="Engineering & Architecture")],
        )
        record = load_training_records(csv)[0]
        assert record.career_category == "Engineering, Manufacturing & Construction"

    def test_extra_columns_ignored(self, tmp_path):
        csv = _write_training_csv(tmp_path / "train.csv", [_training_row()])
        records = load_training_records(csv)
        assert len(records) == 1

    def test_multiple_rows_loaded(self, tmp_path):
        rows = [_training_row(category=cat) for cat in BROAD_CATEGORIES]
        csv = _write_training_csv(tmp_path / "train.csv", rows)
        records = load_training_records(csv)
        assert len(records) == len(BROAD_CATEGORIES)

    def test_invalid_csv_raises_value_error(self, tmp_path):
        csv = tmp_path / "bad.csv"
        pd.DataFrame([{"unrelated": 1}]).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="validation failed"):
            load_training_records(csv)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_training_records(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# load_career_profiles
# ---------------------------------------------------------------------------


class TestLoadCareerProfiles:
    """Tests for load_career_profiles."""

    def test_returns_career_profiles(self, tmp_path):
        csv = _write_career_csv(tmp_path / "careers.csv", [_career_row()])
        profiles = load_career_profiles(csv)
        assert len(profiles) == 1
        assert isinstance(profiles[0], CareerProfile)

    def test_fields_loaded_correctly(self, tmp_path):
        csv = _write_career_csv(tmp_path / "careers.csv", [_career_row()])
        profile = load_career_profiles(csv)[0]
        assert profile.code == "11-1011.00"
        assert profile.title == "Chief Executives"
        assert profile.realistic == 0.18
        assert profile.career_category == "Business & Finance"

    def test_multiple_rows_loaded(self, tmp_path):
        rows = [
            _career_row(code=f"11-{i:04d}.00", title=f"Career {i}", category=cat)
            for i, cat in enumerate(BROAD_CATEGORIES)
        ]
        csv = _write_career_csv(tmp_path / "careers.csv", rows)
        profiles = load_career_profiles(csv)
        assert len(profiles) == len(BROAD_CATEGORIES)

    def test_invalid_csv_raises_value_error(self, tmp_path):
        csv = tmp_path / "bad.csv"
        pd.DataFrame([{"unrelated": 1}]).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="validation failed"):
            load_career_profiles(csv)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_career_profiles(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# split_training_records
# ---------------------------------------------------------------------------


def _make_records(n: int) -> list[TrainingRecord]:
    """Make n synthetic TrainingRecords spread evenly across all categories."""
    categories = list(BROAD_CATEGORIES)
    records = []
    for i in range(n):
        records.append(TrainingRecord(
            realistic=0.5, investigative=0.5, artistic=0.5,
            social=0.5, enterprising=0.5, conventional=0.5,
            career_category=categories[i % len(categories)],
        ))
    return records


class TestSplitTrainingRecords:
    """Tests for split_training_records."""

    def test_default_split_ratios(self):
        records = _make_records(200)
        train, val, test = split_training_records(records)
        total = len(records)
        assert abs(len(train) / total - 0.70) < 0.02
        assert abs(len(val) / total - 0.15) < 0.02
        assert abs(len(test) / total - 0.15) < 0.02

    def test_no_records_lost(self):
        records = _make_records(200)
        train, val, test = split_training_records(records)
        assert len(train) + len(val) + len(test) == len(records)

    def test_splits_are_disjoint(self):
        records = _make_records(200)
        train, val, test = split_training_records(records)
        train_set = set(id(r) for r in train)
        val_set = set(id(r) for r in val)
        test_set = set(id(r) for r in test)
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_deterministic_with_same_seed(self):
        records = _make_records(200)
        train_a, val_a, test_a = split_training_records(records, random_state=42)
        train_b, val_b, test_b = split_training_records(records, random_state=42)
        assert [id(r) for r in train_a] == [id(r) for r in train_b]
        assert [id(r) for r in test_a] == [id(r) for r in test_b]

    def test_different_seeds_produce_different_splits(self):
        records = _make_records(200)
        train_a, _, _ = split_training_records(records, random_state=42)
        train_b, _, _ = split_training_records(records, random_state=99)
        assert [id(r) for r in train_a] != [id(r) for r in train_b]

    def test_invalid_sizes_raises(self):
        records = _make_records(100)
        with pytest.raises(ValueError, match="val_size \\+ test_size must be < 1.0"):
            split_training_records(records, val_size=0.5, test_size=0.5)

    def test_returns_training_record_instances(self):
        records = _make_records(100)
        train, val, test = split_training_records(records)
        assert all(isinstance(r, TrainingRecord) for r in train)
        assert all(isinstance(r, TrainingRecord) for r in val)
        assert all(isinstance(r, TrainingRecord) for r in test)

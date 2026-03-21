"""
File: test_training_schemas.py
Path: tests/data/test_training_schemas.py

Purpose:
  Unit tests for the TrainingRecord and CareerProfile dataclasses and the
  BROAD_CATEGORIES / CATEGORY_ALIASES constants added to src/data/schemas.py.

Original Author(s):
  - Claude Code

AI Tools Used:
  - Claude Code - initial implementation

Editors:
  - Claude Code (2026-03-14) — initial implementation
  - Claude Code (2026-03-16) — renamed broad_category → career_category to
    align with Nate's CSV file structure

Last Editor:
  - Claude Code

Last Edit Date:
  2026-03-16

Assumptions & Constraints:
  - Tests run with pytest from the repository root
  - No external data files required (uses synthetic values)

Related Docs:
  - docs/data/data_governance.md
"""

import pytest

from src.data.schemas import (
    BROAD_CATEGORIES,
    CATEGORY_ALIASES,
    CareerProfile,
    TrainingRecord,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_VALID_RIASEC = {
    "realistic": 0.5,
    "investigative": 0.6,
    "artistic": 0.3,
    "social": 0.7,
    "enterprising": 0.4,
    "conventional": 0.2,
}

_VALID_CATEGORY = BROAD_CATEGORIES[0]


# ---------------------------------------------------------------------------
# BROAD_CATEGORIES
# ---------------------------------------------------------------------------


class TestBroadCategories:
    """Tests for the BROAD_CATEGORIES constant."""

    def test_has_ten_categories(self):
        assert len(BROAD_CATEGORIES) == 10

    def test_no_duplicates(self):
        assert len(set(BROAD_CATEGORIES)) == len(BROAD_CATEGORIES)

    def test_contains_expected_categories(self):
        expected = {
            "Arts, Communications & Humanities",
            "Business & Finance",
            "Education",
            "Engineering, Manufacturing & Construction",
            "Healthcare & Medicine",
            "Law & Public Policy",
            "Psychology & Human Services",
            "Science & Mathematics",
            "Social Sciences",
            "Technology & IT",
        }
        assert set(BROAD_CATEGORIES) == expected


# ---------------------------------------------------------------------------
# CATEGORY_ALIASES
# ---------------------------------------------------------------------------


class TestCategoryAliases:
    """Tests for the CATEGORY_ALIASES mapping."""

    def test_engineering_alias_present(self):
        assert "Engineering & Architecture" in CATEGORY_ALIASES

    def test_aliases_map_to_canonical_categories(self):
        for canonical in CATEGORY_ALIASES.values():
            assert canonical in BROAD_CATEGORIES, (
                f"Alias target '{canonical}' is not in BROAD_CATEGORIES"
            )


# ---------------------------------------------------------------------------
# TrainingRecord
# ---------------------------------------------------------------------------


class TestTrainingRecord:
    """Tests for the TrainingRecord dataclass."""

    def test_valid_construction(self):
        record = TrainingRecord(**_VALID_RIASEC, career_category=_VALID_CATEGORY)
        assert record.realistic == 0.5
        assert record.career_category == _VALID_CATEGORY

    @pytest.mark.parametrize("category", BROAD_CATEGORIES)
    def test_all_valid_categories_accepted(self, category):
        record = TrainingRecord(**_VALID_RIASEC, career_category=category)
        assert record.career_category == category

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="career_category must be one of"):
            TrainingRecord(**_VALID_RIASEC, career_category="Underwater Basket Weaving")

    @pytest.mark.parametrize("field", [
        "realistic", "investigative", "artistic",
        "social", "enterprising", "conventional",
    ])
    def test_score_below_zero_raises(self, field):
        kwargs = {**_VALID_RIASEC, field: -0.01, "career_category": _VALID_CATEGORY}
        with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
            TrainingRecord(**kwargs)

    @pytest.mark.parametrize("field", [
        "realistic", "investigative", "artistic",
        "social", "enterprising", "conventional",
    ])
    def test_score_above_one_raises(self, field):
        kwargs = {**_VALID_RIASEC, field: 1.01, "career_category": _VALID_CATEGORY}
        with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
            TrainingRecord(**kwargs)

    def test_boundary_scores_accepted(self):
        boundary = {k: 0.0 for k in _VALID_RIASEC}
        record = TrainingRecord(**boundary, career_category=_VALID_CATEGORY)
        assert record.realistic == 0.0

        boundary_high = {k: 1.0 for k in _VALID_RIASEC}
        record_high = TrainingRecord(**boundary_high, career_category=_VALID_CATEGORY)
        assert record_high.conventional == 1.0

    def test_frozen(self):
        record = TrainingRecord(**_VALID_RIASEC, career_category=_VALID_CATEGORY)
        with pytest.raises(AttributeError):
            record.realistic = 0.9


# ---------------------------------------------------------------------------
# CareerProfile
# ---------------------------------------------------------------------------


class TestCareerProfile:
    """Tests for the CareerProfile dataclass."""

    def _make(self, **overrides) -> CareerProfile:
        kwargs = {
            "code": "11-1011.00",
            "title": "Chief Executives",
            **_VALID_RIASEC,
            "career_category": _VALID_CATEGORY,
            **overrides,
        }
        return CareerProfile(**kwargs)

    def test_valid_construction(self):
        profile = self._make()
        assert profile.code == "11-1011.00"
        assert profile.title == "Chief Executives"
        assert profile.realistic == 0.5

    @pytest.mark.parametrize("category", BROAD_CATEGORIES)
    def test_all_valid_categories_accepted(self, category):
        profile = self._make(career_category=category)
        assert profile.career_category == category

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="career_category must be one of"):
            self._make(career_category="Unknown Field")

    @pytest.mark.parametrize("field", [
        "realistic", "investigative", "artistic",
        "social", "enterprising", "conventional",
    ])
    def test_score_below_zero_raises(self, field):
        with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
            self._make(**{field: -0.01})

    @pytest.mark.parametrize("field", [
        "realistic", "investigative", "artistic",
        "social", "enterprising", "conventional",
    ])
    def test_score_above_one_raises(self, field):
        with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
            self._make(**{field: 1.01})

    def test_frozen(self):
        profile = self._make()
        with pytest.raises(AttributeError):
            profile.title = "New Title"

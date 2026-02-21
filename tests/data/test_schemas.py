"""
File: test_schemas.py
Path: tests/data/test_schemas.py

Purpose:
  Unit tests for src/data/schemas.py — validates dataclass construction,
  RIASEC area validation, and answer option value range enforcement.

Original Author(s):
  - Claude Code

AI Tools Used:
  - Claude Code - initial implementation

Editors:
  - Claude Code (2026-02-16) — initial implementation

Last Editor:
  - Claude Code

Last Edit Date:
  2026-02-16

Assumptions & Constraints:
  - Tests run with pytest

Related Docs:
  - docs/src/data.md
"""

import pytest

from src.data.schemas import (
    RIASEC_AREAS,
    AnswerOption,
    Question,
    QuestionSet,
)


class TestAnswerOption:
    """Tests for AnswerOption dataclass."""

    def test_valid_construction(self):
        opt = AnswerOption(value=1, name="Strongly Dislike")
        assert opt.value == 1
        assert opt.name == "Strongly Dislike"

    @pytest.mark.parametrize("value", [1, 2, 3, 4, 5])
    def test_valid_values(self, value):
        opt = AnswerOption(value=value, name="Test")
        assert opt.value == value

    @pytest.mark.parametrize("value", [0, 6, -1, 100])
    def test_invalid_values(self, value):
        with pytest.raises(ValueError, match="must be between 1 and 5"):
            AnswerOption(value=value, name="Test")

    def test_frozen(self):
        opt = AnswerOption(value=1, name="Test")
        with pytest.raises(AttributeError):
            opt.value = 2


class TestQuestion:
    """Tests for Question dataclass."""

    def test_valid_construction(self):
        q = Question(index=1, area="Realistic", text="Build kitchen cabinets")
        assert q.index == 1
        assert q.area == "Realistic"
        assert q.text == "Build kitchen cabinets"

    @pytest.mark.parametrize("area", RIASEC_AREAS)
    def test_valid_areas(self, area):
        q = Question(index=1, area=area, text="Test question")
        assert q.area == area

    def test_invalid_area(self):
        with pytest.raises(ValueError, match="must be one of"):
            Question(index=1, area="Invalid", text="Test")

    def test_frozen(self):
        q = Question(index=1, area="Realistic", text="Test")
        with pytest.raises(AttributeError):
            q.area = "Artistic"


class TestQuestionSet:
    """Tests for QuestionSet dataclass."""

    def test_valid_construction(self):
        questions = [
            Question(index=1, area="Realistic", text="Q1"),
            Question(index=2, area="Investigative", text="Q2"),
        ]
        options = [
            AnswerOption(value=1, name="Strongly Dislike"),
            AnswerOption(value=5, name="Strongly Like"),
        ]
        qs = QuestionSet(
            questions=questions,
            answer_options=options,
            total=2,
            dataset_id="DATA-onet-ip60-v1",
        )
        assert len(qs.questions) == 2
        assert len(qs.answer_options) == 2
        assert qs.total == 2
        assert qs.dataset_id == "DATA-onet-ip60-v1"

    def test_empty_question_set(self):
        qs = QuestionSet(
            questions=[],
            answer_options=[],
            total=0,
            dataset_id="DATA-test-v1",
        )
        assert len(qs.questions) == 0


class TestRIASECAreas:
    """Tests for the RIASEC_AREAS constant."""

    def test_has_six_areas(self):
        assert len(RIASEC_AREAS) == 6

    def test_contains_all_holland_codes(self):
        expected = {
            "Realistic",
            "Investigative",
            "Artistic",
            "Social",
            "Enterprising",
            "Conventional",
        }
        assert set(RIASEC_AREAS) == expected

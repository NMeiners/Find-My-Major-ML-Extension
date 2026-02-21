"""
File: test_store.py
Path: tests/data/test_store.py

Purpose:
  Unit tests for src/data/store.py — validates JSON round-trip serialization
  and file structure for QuestionSet persistence.

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
  - Tests use tmp_path fixture for isolated file I/O

Related Docs:
  - docs/src/data.md
"""

import json

import pytest

from src.data.schemas import AnswerOption, Question, QuestionSet
from src.data.store import load_questions, save_questions


@pytest.fixture
def sample_question_set():
    """A small QuestionSet fixture for testing."""
    return QuestionSet(
        questions=[
            Question(index=1, area="Realistic", text="Build kitchen cabinets"),
            Question(index=2, area="Investigative", text="Study the structure of the human body"),
            Question(index=3, area="Artistic", text="Compose or arrange music"),
        ],
        answer_options=[
            AnswerOption(value=1, name="Strongly Dislike"),
            AnswerOption(value=2, name="Dislike"),
            AnswerOption(value=3, name="Unsure"),
            AnswerOption(value=4, name="Like"),
            AnswerOption(value=5, name="Strongly Like"),
        ],
        total=3,
        dataset_id="DATA-onet-ip60-v1",
    )


class TestSaveQuestions:
    """Tests for save_questions."""

    def test_creates_file(self, tmp_path, sample_question_set):
        path = tmp_path / "test.json"
        save_questions(sample_question_set, path)
        assert path.exists()

    def test_creates_parent_directories(self, tmp_path, sample_question_set):
        path = tmp_path / "nested" / "dir" / "test.json"
        save_questions(sample_question_set, path)
        assert path.exists()

    def test_json_structure(self, tmp_path, sample_question_set):
        path = tmp_path / "test.json"
        save_questions(sample_question_set, path)

        with open(path) as f:
            data = json.load(f)

        assert "questions" in data
        assert "answer_options" in data
        assert "total" in data
        assert "dataset_id" in data
        assert len(data["questions"]) == 3
        assert data["questions"][0]["area"] == "Realistic"
        assert data["answer_options"][0]["value"] == 1
        assert data["dataset_id"] == "DATA-onet-ip60-v1"


class TestLoadQuestions:
    """Tests for load_questions."""

    def test_round_trip(self, tmp_path, sample_question_set):
        path = tmp_path / "test.json"
        save_questions(sample_question_set, path)
        loaded = load_questions(path)

        assert loaded.total == sample_question_set.total
        assert loaded.dataset_id == sample_question_set.dataset_id
        assert len(loaded.questions) == len(sample_question_set.questions)
        assert len(loaded.answer_options) == len(sample_question_set.answer_options)

        for orig, loaded_q in zip(sample_question_set.questions, loaded.questions):
            assert orig.index == loaded_q.index
            assert orig.area == loaded_q.area
            assert orig.text == loaded_q.text

        for orig, loaded_opt in zip(sample_question_set.answer_options, loaded.answer_options):
            assert orig.value == loaded_opt.value
            assert orig.name == loaded_opt.name

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_questions(tmp_path / "nonexistent.json")

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json")
        with pytest.raises(json.JSONDecodeError):
            load_questions(path)

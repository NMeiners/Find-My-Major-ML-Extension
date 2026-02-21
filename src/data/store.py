"""
File: store.py
Path: src/data/store.py

Purpose:
  Handles local persistence of Interest Profiler question data as JSON.
  Provides save/load functions for QuestionSet serialization.

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
  - Stores data in data/raw/ per data governance policy
  - JSON format for human readability and portability
  - Raw data files are immutable once written

Related Docs:
  - docs/src/data.md
  - docs/data/data_governance.md
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.data.schemas import AnswerOption, Question, QuestionSet

DEFAULT_PATH = Path("data/raw/interest_profiler_questions.json")


def save_questions(question_set: QuestionSet, path: Path = DEFAULT_PATH) -> None:
    """
    Name: save_questions

    Purpose:
      Serializes a QuestionSet to a JSON file on disk.

    Inputs:
      - question_set: QuestionSet — the data to persist
      - path: Path — output file path (default: data/raw/interest_profiler_questions.json)

    Outputs:
      - None (writes file to disk)

    Raises / Errors:
      - OSError: if the parent directory cannot be created or file cannot be written

    Notes:
      - Creates parent directories if they don't exist
      - Overwrites existing file at the same path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(question_set)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_questions(path: Path = DEFAULT_PATH) -> QuestionSet:
    """
    Name: load_questions

    Purpose:
      Deserializes a QuestionSet from a local JSON file.

    Inputs:
      - path: Path — input file path (default: data/raw/interest_profiler_questions.json)

    Outputs:
      - QuestionSet — reconstructed from the JSON data

    Raises / Errors:
      - FileNotFoundError: if the JSON file does not exist
      - json.JSONDecodeError: if the file contains invalid JSON
      - ValueError: if the data fails schema validation

    Notes:
      - Validates all questions and answer options through their dataclass constructors
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = [
        Question(index=q["index"], area=q["area"], text=q["text"])
        for q in data["questions"]
    ]

    answer_options = [
        AnswerOption(value=opt["value"], name=opt["name"])
        for opt in data["answer_options"]
    ]

    return QuestionSet(
        questions=questions,
        answer_options=answer_options,
        total=data["total"],
        dataset_id=data["dataset_id"],
    )

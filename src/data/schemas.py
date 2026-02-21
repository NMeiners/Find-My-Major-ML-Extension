"""
File: schemas.py
Path: src/data/schemas.py

Purpose:
  Defines core data structures for O*NET Interest Profiler questions.
  Provides validated dataclasses for questions, answer options, and question sets
  used throughout the project.

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
  - RIASEC areas are fixed to the 6 standard Holland codes
  - Answer option values are always 1-5 (Likert scale)
  - No PII or demographic data

Related Docs:
  - docs/src/data.md
  - docs/data/data_governance.md
"""

from __future__ import annotations

from dataclasses import dataclass

RIASEC_AREAS: tuple[str, ...] = (
    "Realistic",
    "Investigative",
    "Artistic",
    "Social",
    "Enterprising",
    "Conventional",
)


@dataclass(frozen=True)
class AnswerOption:
    """
    Name: AnswerOption

    Purpose:
      Represents a single Likert-scale answer choice for an Interest Profiler question.

    Inputs:
      - value: int — numeric score (1-5)
      - name: str — display label (e.g., "Strongly Dislike")

    Outputs:
      - N/A (data container)

    Raises / Errors:
      - ValueError: if value is not between 1 and 5

    Notes:
      - Frozen dataclass to prevent accidental mutation
    """

    value: int
    name: str

    def __post_init__(self) -> None:
        if not 1 <= self.value <= 5:
            raise ValueError(
                f"AnswerOption value must be between 1 and 5, got {self.value}"
            )


@dataclass(frozen=True)
class Question:
    """
    Name: Question

    Purpose:
      Represents a single O*NET Interest Profiler question with its RIASEC area.

    Inputs:
      - index: int — 1-based question number
      - area: str — one of the 6 RIASEC categories
      - text: str — the question text

    Outputs:
      - N/A (data container)

    Raises / Errors:
      - ValueError: if area is not a valid RIASEC category

    Notes:
      - Frozen dataclass to prevent accidental mutation
    """

    index: int
    area: str
    text: str

    def __post_init__(self) -> None:
        if self.area not in RIASEC_AREAS:
            raise ValueError(
                f"Question area must be one of {RIASEC_AREAS}, got '{self.area}'"
            )


@dataclass(frozen=True)
class QuestionSet:
    """
    Name: QuestionSet

    Purpose:
      Container for a complete set of Interest Profiler questions with metadata.

    Inputs:
      - questions: list[Question] — the profiler questions
      - answer_options: list[AnswerOption] — available answer choices
      - total: int — expected total number of questions
      - dataset_id: str — version identifier per data governance policy

    Outputs:
      - N/A (data container)

    Raises / Errors:
      - N/A

    Notes:
      - dataset_id follows the DATA-<short_name>-vX convention
    """

    questions: list[Question]
    answer_options: list[AnswerOption]
    total: int
    dataset_id: str

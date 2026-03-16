"""
File: schemas.py
Path: src/data/schemas.py

Purpose:
  Defines core data structures for O*NET Interest Profiler questions and
  processed training data. Provides validated dataclasses for questions,
  answer options, question sets, training records, and career profiles
  used throughout the project.

Original Author(s):
  - Claude Code

AI Tools Used:
  - Claude Code - initial implementation

Editors:
  - Claude Code (2026-02-16) — initial implementation
  - Claude Code (2026-03-14) — added TrainingRecord, CareerProfile,
    BROAD_CATEGORIES, and CATEGORY_ALIASES for processed dataset schema

Last Editor:
  - Claude Code

Last Edit Date:
  2026-03-14

Assumptions & Constraints:
  - RIASEC areas are fixed to the 6 standard Holland codes
  - Answer option values are always 1-5 (Likert scale)
  - Normalized RIASEC scores are in [0.0, 1.0]
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

BROAD_CATEGORIES: tuple[str, ...] = (
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
)

# Maps legacy source dataset category names to canonical BROAD_CATEGORIES values.
# Applied by the data loader before validation runs.
CATEGORY_ALIASES: dict[str, str] = {
    "Engineering & Architecture": "Engineering, Manufacturing & Construction",
}


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


_RIASEC_FLOAT_FIELDS: tuple[str, ...] = (
    "realistic",
    "investigative",
    "artistic",
    "social",
    "enterprising",
    "conventional",
)


@dataclass(frozen=True)
class TrainingRecord:
    """
    Name: TrainingRecord

    Purpose:
      Represents a single respondent row from the processed Kaggle RIASEC
      dataset. Holds normalized RIASEC scores and the assigned career
      broad category used as the training label.

    Inputs:
      - realistic: float — normalized R score in [0.0, 1.0]
      - investigative: float — normalized I score in [0.0, 1.0]
      - artistic: float — normalized A score in [0.0, 1.0]
      - social: float — normalized S score in [0.0, 1.0]
      - enterprising: float — normalized E score in [0.0, 1.0]
      - conventional: float — normalized C score in [0.0, 1.0]
      - career_category: str — canonical career category label

    Outputs:
      - N/A (data container)

    Raises / Errors:
      - ValueError: if any RIASEC score is outside [0.0, 1.0]
      - ValueError: if career_category is not in BROAD_CATEGORIES

    Notes:
      - Frozen dataclass to prevent accidental mutation
      - Category aliases must be resolved before construction (see CATEGORY_ALIASES)
      - Demographic fields are intentionally excluded per data governance policy
    """

    realistic: float
    investigative: float
    artistic: float
    social: float
    enterprising: float
    conventional: float
    career_category: str

    def __post_init__(self) -> None:
        for field in _RIASEC_FLOAT_FIELDS:
            val = getattr(self, field)
            if not 0.0 <= val <= 1.0:
                raise ValueError(
                    f"TrainingRecord.{field} must be in [0.0, 1.0], got {val}"
                )
        if self.career_category not in BROAD_CATEGORIES:
            raise ValueError(
                f"career_category must be one of {BROAD_CATEGORIES}, "
                f"got '{self.career_category}'"
            )


@dataclass(frozen=True)
class CareerProfile:
    """
    Name: CareerProfile

    Purpose:
      Represents a single career entry from the master careers dataset.
      Holds the career's O*NET code, title, normalized RIASEC profile,
      and broad category used for similarity ranking.

    Inputs:
      - code: str — O*NET-SOC code (e.g., "11-1011.00")
      - title: str — career title
      - realistic: float — normalized R score in [0.0, 1.0]
      - investigative: float — normalized I score in [0.0, 1.0]
      - artistic: float — normalized A score in [0.0, 1.0]
      - social: float — normalized S score in [0.0, 1.0]
      - enterprising: float — normalized E score in [0.0, 1.0]
      - conventional: float — normalized C score in [0.0, 1.0]
      - career_category: str — canonical career category label

    Outputs:
      - N/A (data container)

    Raises / Errors:
      - ValueError: if any RIASEC score is outside [0.0, 1.0]
      - ValueError: if career_category is not in BROAD_CATEGORIES

    Notes:
      - Frozen dataclass to prevent accidental mutation
      - RIASEC scores are on the same [0.0, 1.0] scale as TrainingRecord
        to allow direct cosine similarity comparison
    """

    code: str
    title: str
    realistic: float
    investigative: float
    artistic: float
    social: float
    enterprising: float
    conventional: float
    career_category: str

    def __post_init__(self) -> None:
        for field in _RIASEC_FLOAT_FIELDS:
            val = getattr(self, field)
            if not 0.0 <= val <= 1.0:
                raise ValueError(
                    f"CareerProfile.{field} must be in [0.0, 1.0], got {val}"
                )
        if self.career_category not in BROAD_CATEGORIES:
            raise ValueError(
                f"career_category must be one of {BROAD_CATEGORIES}, "
                f"got '{self.career_category}'"
            )

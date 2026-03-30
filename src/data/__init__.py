"""
File: __init__.py
Path: src/data/__init__.py

Purpose:
  Public interface for the data module. Exports schemas, API fetcher, and
  local storage functions for O*NET Interest Profiler questions.

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
  - All public interfaces are documented in docs/src/data.md

Related Docs:
  - docs/src/data.md
"""

from src.data.fetch_questions import fetch_questions
from src.data.schemas import AnswerOption, Question, QuestionSet
from src.data.store import load_questions, save_questions

__all__ = [
    "AnswerOption",
    "Question",
    "QuestionSet",
    "fetch_questions",
    "load_questions",
    "save_questions",
]

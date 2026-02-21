"""
File: fetch_onet_questions.py
Path: scripts/fetch_onet_questions.py

Purpose:
  CLI script to fetch O*NET Interest Profiler questions and save them locally.
  Loads credentials from a .env file, calls the API, and persists the result.

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
  - Requires a .env file with ONET_API_KEY
  - Must be run from the project root directory

Related Docs:
  - docs/src/data.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src.data.fetch_questions import fetch_questions
from src.data.store import save_questions

if __name__ == "__main__":
    load_dotenv()

    print("Fetching O*NET Interest Profiler questions...")
    question_set = fetch_questions()
    save_questions(question_set)

    print(
        f"Saved {len(question_set.questions)} questions "
        f"with {len(question_set.answer_options)} answer options "
        f"to data/raw/interest_profiler_questions.json"
    )

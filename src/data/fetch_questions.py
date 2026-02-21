"""
File: fetch_questions.py
Path: src/data/fetch_questions.py

Purpose:
  Fetches O*NET Interest Profiler questions from the public API and returns
  them as validated QuestionSet instances.

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
  - Requires a valid O*NET API key
  - Network access to api-v2.onetcenter.org
  - Questions are numbered 1-60 by default

Related Docs:
  - docs/src/data.md
"""

from __future__ import annotations

import os

import requests

from src.data.schemas import AnswerOption, Question, QuestionSet

ONET_API_BASE = "https://api-v2.onetcenter.org"
DATASET_ID = "DATA-onet-ip60-v1"


def _get_api_key() -> str:
    """
    Name: _get_api_key

    Purpose:
      Reads ONET_API_KEY from environment variables.

    Raises / Errors:
      - EnvironmentError: if the env var is missing
    """
    api_key = os.environ.get("ONET_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ONET_API_KEY environment variable is required. "
            "Copy .env.example to .env and fill in your API key."
        )
    return api_key


def fetch_questions(
    api_key: str | None = None, start: int = 1, end: int = 60
) -> QuestionSet:
    """
    Name: fetch_questions

    Purpose:
      Fetches Interest Profiler questions from the O*NET Web Services API.

    Inputs:
      - api_key: str | None — O*NET API key for authentication; if None,
        read from ONET_API_KEY env var
      - start: int — first question index to fetch (default 1)
      - end: int — last question index to fetch (default 60)

    Outputs:
      - QuestionSet — validated container of fetched questions

    Raises / Errors:
      - EnvironmentError: if env vars are missing and no api_key provided
      - requests.HTTPError: on 401 (invalid key) or 422 (invalid params)
      - requests.ConnectionError: on network failure

    Notes:
      - The API returns all 60 questions in a single call when start=1, end=60
    """
    if api_key is None:
        api_key = _get_api_key()
    url = f"{ONET_API_BASE}/mnm/interestprofiler/questions"
    headers = {
        "User-Agent": "python-OnetWebService/2.00 (bot)",
        "X-API-Key": api_key,
        "Accept": "application/json",
    }
    params = {"start": start, "end": end}

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    questions = [
        Question(
            index=q["index"],
            area=q["area"].capitalize(),
            text=q["text"],
        )
        for q in data["question"]
    ]

    answer_options = [
        AnswerOption(value=opt["value"], name=opt["name"])
        for opt in data["answer_option"]
    ]

    return QuestionSet(
        questions=questions,
        answer_options=answer_options,
        total=data["total"],
        dataset_id=DATASET_ID,
    )

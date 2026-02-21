
# Module: src/data

## Responsibility
Owns all data structures, API fetching, and local storage for O*NET Interest Profiler questions.
This module does NOT handle data preprocessing, feature engineering, or model-specific transformations.

## Public Interfaces
- `QuestionSet`: Dataclass container for a complete set of profiler questions with metadata
- `Question`: Dataclass for a single profiler question with RIASEC area
- `AnswerOption`: Dataclass for a Likert-scale answer choice (value 1-5)
- `fetch_questions(api_key, start, end)`: Fetches questions from the O*NET API
    - `requests`: external HTTP library
- `save_questions(question_set, path)`: Serializes a QuestionSet to JSON on disk
- `load_questions(path)`: Deserializes a QuestionSet from a local JSON file

## Internal Structure
- `schemas.py` — Core dataclasses (`Question`, `AnswerOption`, `QuestionSet`) and `RIASEC_AREAS` constant
- `fetch_questions.py` — O*NET API client for retrieving Interest Profiler questions
- `store.py` — JSON serialization/deserialization for local persistence

## Data Contracts
- Inputs:
    - O*NET API key (string) for fetching
    - File path (Path) for save/load operations
- Outputs:
    - `QuestionSet` containing validated `Question` and `AnswerOption` instances
    - JSON files in `data/raw/` following the data governance directory structure

## Constraints
- Performance: API calls have a 30-second timeout
- Memory: All 60 questions fit comfortably in memory
- Privacy: No PII or demographic data is collected or stored
- Bias considerations: Questions are sourced directly from O*NET without modification

## Related Modules
- src/models (will consume QuestionSet for scoring)
- src/evaluation (will use RIASEC areas for metric computation)

## Related Documentation
- docs/data/data_governance.md

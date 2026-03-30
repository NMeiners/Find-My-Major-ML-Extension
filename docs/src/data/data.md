# Module: src/data

## Responsibility
Owns data structures, loading, validation, preprocessing, and storage for the project datasets.
This includes:
- O*NET Interest Profiler question schemas and API fetch/storage
- Processed training/career CSV loading and schema validation
- Conversion of typed records into model-ready feature/label matrices

This module does NOT handle model training, model inference orchestration, or evaluation scoring.

## Public Interfaces
- `AnswerOption`, `Question`, `QuestionSet` (question schema dataclasses)
- `TrainingRecord`, `CareerProfile` (typed training/career dataclasses)
- `fetch_questions(api_key, start, end)`
- `save_questions(question_set, path)`
- `load_questions(path)`
- `load_training_records(path)`
- `load_career_profiles(path)`
- `split_training_records(records, val_size, test_size, random_state)`
- `records_to_dataframe(records)`
- `extract_features(df)`
- `extract_labels(df)`
- `build_training_matrix(records)`
- `validate_training_dataframe(df)`
- `validate_career_dataframe(df)`

## Internal Structure
- `schemas.py` — Dataclasses, category constants, and alias mapping
- `fetch_questions.py` — O*NET API client for question retrieval
- `store.py` — JSON persistence for question datasets
- `loader.py` — CSV loading + typed record construction + stratified splitting
- `validate.py` — DataFrame validation helpers for training/career CSVs
- `preprocess.py` — Feature/label extraction and derived feature construction
- `__init__.py` — Public export surface for question-related interfaces

## Data Contracts
- Inputs:
  - Question API responses from O*NET
  - Processed CSV files under `docs/data/`
- Outputs:
  - Typed record collections (`TrainingRecord`, `CareerProfile`)
  - DataFrames/Series compatible with sklearn model interfaces

## Constraints
- Validation is performed on raw CSV schema before typed conversion.
- RIASEC score fields are validated to `[0.0, 1.0]` in typed dataclasses.
- Train/val/test splitting is deterministic when `random_state` is fixed.
- Raw research datasets may contain extra non-model columns; model feature selection is explicit and config-driven.

## Related Modules
- `src/models` (consumes feature matrices and career profiles)
- `src/evaluation` (consumes typed records/dataset splits)

## Related Documentation
- `docs/src/data/data.md`
- `docs/templates/exp_config.yaml`
- `docs/data/data_governance.md`
- `docs/data/pipeline_architecture.md`

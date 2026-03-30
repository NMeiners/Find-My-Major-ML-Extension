# Data Pipeline Architecture

## Flow
1. Load raw processed CSVs from `docs/data/`.
2. Validate raw DataFrames (`validate.py`).
3. Normalize category aliases and map columns (`loader.py`).
4. Build typed records (`TrainingRecord`, `CareerProfile`).
5. Convert to model-ready matrices (`preprocess.py`) where needed.

## Key Modules
- `src/data/loader.py`
- `src/data/validate.py`
- `src/data/schemas.py`
- `src/data/preprocess.py`

## Notes
- Dataset splitting is deterministic with fixed seed.
- Feature selection remains config-driven at training time.

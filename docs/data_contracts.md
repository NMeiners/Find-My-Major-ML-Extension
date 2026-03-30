# Data Contracts (Cross-Module)

## Training Records
- Six normalized RIASEC features in `[0.0, 1.0]`.
- Canonical `career_category` label.

## Career Profiles
- O*NET code/title + six normalized RIASEC scores + canonical category.

## Recommendation Output
- DataFrame columns: `Title`, `Career Category`, `Match_Score`.

## Evaluation Output
- Result entries include metrics, latency, memory, model size, and constraint placeholders.

## Related
- `docs/src/data/data.md`
- `docs/src/models/models.md`
- `docs/src/evaluation/evaluation.md`

# Data Governance (Repository)

## Scope
Defines governance expectations for datasets stored under `docs/data/` and consumed by `src/data`.

## Principles
- Preserve source datasets for reproducibility.
- Validate schema and numeric ranges before model use.
- Restrict model feature usage to explicitly configured columns.
- Keep category labels canonicalized via documented alias mapping.

## Operational Rules
- Raw datasets may include extra research columns not used as model inputs.
- Data-loading and validation responsibilities live in `src/data/loader.py` and `src/data/validate.py`.
- Typed invariants for runtime use are enforced by `src/data/schemas.py`.

## Related
- `docs/src/data/data.md`
- `docs/data/pipeline_architecture.md`
- `docs/coding_rules.md`

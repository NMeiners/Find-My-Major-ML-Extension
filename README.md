# Find My Major ML Extension

Lightweight AI/ML-assisted career recommendation research project based on the Holland Code (RIASEC), designed for short advisor-student interactions and mobile/offline use cases.

## Problem

Many undecided students receive career guidance that is too broad during short advising sessions. Traditional Holland Code tools are often static and not adaptive enough for fast, personalized recommendations.

## Project Goal

Build and evaluate a prototype that can:

- Infer RIASEC profiles from minimal input (target: ~10-15 questions)
- Generate ranked career recommendations
- Refine recommendations using user feedback
- Run with low latency on mobile-friendly hardware (local/offline emphasis)

## Proposed System

The project proposes a local, lightweight recommendation flow:

1. Collect short student interaction signals
2. Infer a RIASEC vector using an ML model
3. Rank career suggestions aligned to inferred interests
4. Allow rejection/refinement feedback to improve follow-up suggestions

## Success Criteria (From Proposal)

- RIASEC inference agreement with fuller assessments (for example, cosine similarity or rank correlation)
- Ranking quality metrics (for example, NDCG@K, Precision@K)
- User relevance/satisfaction feedback
- Practical constraints: low latency and mobile-compatible model footprint

## Scope

In scope:

- AI/ML-based RIASEC and career recommendation prototyping
- Model experimentation and feature engineering
- Evaluation workflow and reproducibility practices

Out of scope (current project term):

- Full production UI
- Full production-grade mapping of all careers to MSU Denver programs

## Repository Status

Current repository state is an early scaffold with governance-first documentation:

- `src/` module files currently exist as placeholders
- `tests/` exists but does not yet include implemented test modules
- Documentation, contribution workflow, and CI policy are already defined

This means the project has structure and standards in place, while core model/evaluation implementation is still in-progress.

## Repository Structure

```text
.
|-- .github/
|   |-- workflows/ci.yml
|   `-- contributing/
|-- docs/
|   |-- README.md
|   |-- repo_structure.md
|   |-- coding_rules.md
|   |-- ai_usage_policy.md
|   |-- evaluation_workflow.md
|   |-- data/data_governance.md
|   |-- experiments/tracking.md
|   |-- ci/ci_design.md
|   `-- templates/
|-- experiments/
|-- notebooks/
|-- src/
|   |-- data/
|   |-- models/
|   |-- evaluation/
|   `-- export/
`-- tests/
```

## Working Agreement (Important)

Before contributing code, read:

1. `docs/repo_structure.md`
2. `docs/coding_rules.md`
3. `docs/ai_usage_policy.md`
4. `docs/evaluation_workflow.md`
5. `docs/data/data_governance.md`
6. `docs/experiments/tracking.md`

These documents are operational constraints, not optional references.

## Development and CI Notes

- CI workflow: `.github/workflows/ci.yml`
- Primary CI test entry point: `pytest`
- CI currently validates importability with `python -c "import src"`

As implementation is added, maintain doc-code parity and keep tests aligned with `src/`.

## Planned Technology Stack (Proposal)

- Python
- Jupyter Notebook / Google Colab (research workflow)
- scikit-learn
- PyTorch or TensorFlow
- Mobile export path (for example, TorchScript or TensorFlow Lite)
- O*NET occupational data/resources

## Data, Privacy, and Fairness Assumptions

- No PII collection in baseline workflow
- No demographic/protected attributes in modeling features
- Synthetic data may be used only for early prototyping/debugging
- Final evaluation should use anonymized participant data
- Bias checks should monitor skew in recommendations across RIASEC categories

## Milestone Targets (Proposal)

- Research complete: Week 6
- Initial ML prototype: Week 9
- Refinement/feedback loop: Week 12
- Final local demo: Week 16

## Proposal Source

Project planning details are documented in `F2_Project Proposal.docx` at the repository root.

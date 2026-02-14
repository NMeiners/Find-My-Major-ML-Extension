# Experiment Tracking and Logging

This document defines how experiments are tracked in this repository.

The goal is to ensure:

* Reproducibility
* Clear comparison between models
* Explicit recording of hyperparameters and seeds
* Separation between research execution and production code

This system is lightweight and tool-agnostic.

Experiment artifacts are NOT committed to the repository. They must be reproducible locally.

---

# 1. Core Principles

1. Every experiment must have a unique identifier.
2. Every experiment must generate structured metadata.
3. Results must be reproducible from code + commit history.
4. No experiment claim is valid without structured output.
5. Large artifacts (models, logs, outputs) must not be committed.

---

# 2. Directory Structure

The repository includes an experiments directory:

```
experiments/
    README.md
    logs/
    results/
```

However:

* `experiments/logs/` is ignored by `.gitignore`
* `experiments/results/` is ignored by `.gitignore`

These directories exist locally for structured output only.

---

# 3. Experiment ID Convention

Each experiment must use the following format:

```
EXP-YYYYMMDD-XX
```

Example:

```
EXP-20260213-01
```

This ID must appear in:

* The result file name
* The metadata log
* The notebook header (if applicable)
* The commit message (recommended)

---

# 4. Required Experiment Metadata

Each experiment must generate a structured JSON file containing at minimum:

```json
{
  "experiment_id": "EXP-20260213-01",
  "model_name": "baseline_model",
  "dataset_id": "v1",
  "random_seed": 42,
  "hyperparameters": {},
  "evaluation_metrics": {},
  "timestamp": "ISO-8601",
  "git_commit": "commit_hash"
}
```

Required fields:

* experiment_id
* model_name
* dataset_id
* random_seed
* hyperparameters
* evaluation_metrics (core metrics only)
* timestamp
* git_commit

Free-form notes may be included but may not replace structured fields.

---

# 5. Results Storage Rules

Experiment outputs must:

* Be saved as structured JSON or CSV
* Be written to `experiments/results/`
* Not overwrite previous runs unless explicitly intended
* Be reproducible from the same commit

Model binaries and large artifacts must not be committed.

---

# 6. Notebook Integration

If experiments are conducted in notebooks:

* The notebook must still generate structured metadata files.
* The experiment_id must appear in the notebook header.
* Re-running the notebook with the same seed must reproduce identical metrics.

Notebook cell output alone is not valid experiment evidence.

---

# 7. Baseline Comparison Rule

A baseline experiment must be defined early.

All new experiments must:

* Compare against baseline metrics
* Report improvements or regressions
* Justify trade-offs

Baseline results must be reproducible from code and commit history.

---

# 8. AI Usage Restrictions

AI may:

* Generate logging boilerplate
* Create metadata structures
* Refactor experiment scripts

AI may not:

* Fabricate metrics
* Modify past experiment outputs
* Remove required metadata fields
* Alter baseline definitions without documentation updates

All logged metrics must originate from actual evaluation functions.

---

# 9. Reproducibility Requirement

To reproduce an experiment, a reviewer must be able to:

1. Checkout the recorded commit.
2. Install dependencies.
3. Run the experiment script.
4. Regenerate identical metrics using the same seed.

If metrics cannot be reproduced, the experiment is invalid.

---

This experiment tracking system prioritizes reproducibility and discipline while avoiding unnecessary tooling complexity.

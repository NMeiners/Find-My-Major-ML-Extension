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
    config/
    logs/
    results/
```

However:

* `experiments/logs/` is ignored by `.gitignore`
* `experiments/results/` is ignored by `.gitignore`

These directories exist locally for structured output only.

---

# 3. Experiment and Run ID Convention

Experiments are identified by `experiment.id` in config files.

Current convention:

```
exp_<NNN>
```

Example:

```
exp_001
```

At runtime, `src/config/config_loader.py` injects a `run_id` using:

```
<experiment_id>_<YYYYMMDD_HHMMSS>
```

Example:

```
exp_001_20260323_171347
```

This produces output directories in the form:

`experiments/results/<experiment_id>/<run_id>/`

---

# 4. Required Experiment Metadata

Each run must persist structured evaluation output (currently `evaluation.json`)
that contains one result object per `(dataset, model)` pair.

Current output shape:

```json
[
  {
    "dataset": "dataset_0",
    "model": "logistic_regression",
    "metrics": {
      "ndcg@5": 0.0,
      "precision@5": 0.0
    },
    "latency_ms": 0.0,
    "memory_bytes": 0.0,
    "model_size_mb": 0.0,
    "constraint_violations": {},
    "samples_evaluated": 0
  }
]
```

Required per-result fields:

* dataset
* model
* metrics
* latency_ms
* memory_bytes
* model_size_mb
* constraint_violations
* samples_evaluated

Optional per-result fields:

* interrupted (boolean; present only when run is user-interrupted and partial model metrics are saved)

Run metadata (`experiment.id`, `run_id`) is encoded in the output directory path.

---

# 5. Results Storage Rules

Experiment outputs must:

* Be saved as structured JSON or CSV
* Be written under `experiments/results/<experiment_id>/<run_id>/`
* Not overwrite previous runs unless explicitly intended
* Be reproducible from the same commit

Model binaries and large artifacts must not be committed.

---

# 6. Notebook Integration

If experiments are conducted in notebooks:

* The notebook must still generate structured metadata files.
* The notebook should reference the same `experiment.id` used in config when reporting results.
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
3. Run `python main.py experiments/config/exp_config.yaml` (or the recorded config path).
4. Regenerate equivalent metrics using the same seed/config.

If metrics cannot be reproduced, the experiment is invalid.

---

This experiment tracking system prioritizes reproducibility and discipline while avoiding unnecessary tooling complexity.

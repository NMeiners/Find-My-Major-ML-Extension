# Evaluation Workflow

This document defines how models are evaluated in this repository.

The goal is to ensure:

* Reproducibility
* Metric consistency
* Fair comparison across experiments
* Proper handling of fairness considerations
* Clear separation between modeling and evaluation logic

This workflow is tool-agnostic. It defines governance, not tooling.

---

# 1. Core Principles

1. Evaluation logic must be centralized.
2. Metrics must be deterministic.
3. Dataset splits must be fixed and documented.
4. Evaluation code must not live inside model modules.
5. Any metric change requires documentation updates.
6. Fairness metrics must be explicit if demographic features are present.

---

# 2. Structural Separation

Evaluation logic must reside in a dedicated evaluation module (e.g., `src/evaluation/`).

Evaluation logic must not:

* Be embedded inside training loops
* Be duplicated across notebooks
* Be copy-pasted across modules

Notebooks may call evaluation functions but may not define final evaluation logic.

---

# 3. Dataset Splits

All evaluation must use explicitly defined splits.

Required documentation for splits:

* Split strategy (random, stratified, etc.)
* Random seed
* Train/validation/test proportions
* Any filtering steps

Evaluation must never be performed on training data for final reporting.

If split logic changes, documentation must be updated in the same commit.

---

# 4. Metric Governance

## 4.1 Core Performance Metrics

Core metrics must:

* Be implemented once
* Be unit tested
* Have documented definitions
* Remain stable across experiments

These metrics define primary model comparison.

---

## 4.2 Fairness Metrics

If protected or demographic attributes exist in the dataset:

* Subgroup performance must be computed
* Disparities must be reported
* Fairness definitions must be documented

Protected attributes must not influence training unless explicitly documented.

---

## 4.3 Experimental Metrics

Exploratory metrics are allowed but must:

* Be labeled experimental
* Not replace core metrics
* Not redefine success criteria

Experimental metrics may inform research but may not silently change evaluation standards.

---

# 5. Standard Evaluation Flow

1. Train model
2. Freeze model parameters
3. Call centralized evaluation function
4. Save structured results
5. Compare against baseline

Evaluation must not rely on notebook cell outputs for official claims.

---

# 6. Results Storage

Evaluation results must be saved in structured format (e.g., JSON or CSV).

Each result artifact must include:

* Model identifier
* Dataset identifier
* Random seed
* Evaluation timestamp
* Git commit hash (if available)

Notebook screenshots are not valid evaluation artifacts.

---

# 7. Baseline Rule

A baseline model must be defined before advanced experimentation.

All new models must:

* Compare against baseline
* Report improvements and regressions
* Justify trade-offs

A new model may not replace a baseline without documented improvement or rationale.

---

# 8. Change Control

The following require documentation updates:

* Adding or removing metrics
* Changing split strategy
* Modifying aggregation methods
* Redefining fairness metrics

Major evaluation methodology changes require an ADR.

---

# 9. AI Restrictions

AI may:

* Implement metric functions
* Refactor evaluation code
* Generate unit tests

AI may not:

* Redefine success criteria
* Remove fairness checks
* Modify evaluation standards without explicit instruction

All evaluation changes must be reviewed carefully by a human.

---

Evaluation governance ensures that research iteration remains valid, reproducible, and ethically defensible.

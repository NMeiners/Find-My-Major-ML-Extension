# Data Governance

This document defines how data is handled in this repository.

The goal is to ensure:

* Reproducibility
* Ethical handling of sensitive attributes
* Clear dataset versioning
* Prevention of data leakage
* Separation between raw data, processed data, and derived artifacts

This policy applies to all datasets used for training, validation, testing, and evaluation.

---

# 1. Core Principles

1. Data must be version-identifiable.
2. Raw data must never be modified in place.
3. Data preprocessing must be reproducible from code.
4. Sensitive attributes must be handled explicitly.
5. No dataset assumptions may remain undocumented.

---

# 2. Directory Structure

Recommended structure:

```
data/
    raw/
    processed/
    interim/
```

Rules:

* `raw/` contains immutable original datasets.
* `interim/` contains temporary transformation outputs.
* `processed/` contains finalized model-ready datasets.

Large datasets must be ignored via `.gitignore` unless they are small and legally redistributable.

---

# 3. Dataset Versioning

Each dataset must have a dataset identifier:

```
DATA-<short_name>-vX
```

Example:

```
DATA-survey-v1
```

The dataset_id must be recorded in:

* Experiment metadata
* Evaluation logs
* Baseline documentation

If preprocessing changes in a way that affects model input, the dataset version must increment.

---

# 4. Preprocessing Requirements

All preprocessing steps must:

* Be implemented in `src/`
* Be deterministic
* Be seed-controlled where randomness exists
* Avoid implicit global state

Preprocessing must not:

* Occur only inside notebooks
* Modify raw data files
* Depend on manual spreadsheet edits

If preprocessing logic changes, dataset version must change.

---

# 5. Train / Validation / Test Integrity

Data splits must:

* Be defined programmatically
* Use fixed random seeds
* Prevent overlap between training and evaluation sets

Data leakage is strictly prohibited.

Examples of leakage:

* Using test statistics during training
* Target leakage via derived features
* Scaling using full-dataset statistics

Any discovered leakage must be documented and corrected immediately.

---

# 6. Sensitive and Demographic Attributes

If the dataset includes protected or demographic attributes:

1. Their presence must be documented.
2. Their use in modeling must be explicit.
3. Their exclusion must also be explicit.
4. Fairness evaluation must be enabled (see evaluation_workflow.md).

Sensitive attributes must not be used implicitly.

If attributes are removed during preprocessing, this must be documented.

---

# 7. Data Access and Security

If datasets contain sensitive information:

* Data must not be committed to the repository.
* Access instructions must be documented separately.
* Local storage paths must not be hardcoded.

No credentials may be stored in the repository.

---

# 8. Documentation Requirements

For each dataset version, document:

* Source
* Collection method (if known)
* Known biases or limitations
* Feature definitions
* Preprocessing steps

Documentation may live in `docs/data/`.

---

# 9. AI Usage Restrictions

AI may:

* Generate preprocessing pipelines
* Assist with schema validation
* Help detect leakage risks

AI may not:

* Invent synthetic data without explicit instruction
* Modify dataset schema silently
* Remove protected attributes without documentation

All data-related changes require careful human review.

---

# 10. Reproducibility Requirement

Given:

* Dataset version
* Commit hash
* Random seed

A reviewer must be able to regenerate:

* Processed dataset
* Train/validation/test splits
* Model-ready features

If this cannot be done, the data workflow is invalid.

---

This data governance framework ensures that experimentation remains scientifically valid, ethically defensible, and reproducible.

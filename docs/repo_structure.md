# Repository Structure (Initial)

This document defines the **initial repository structure** for the project. It reflects the current development phase and is intentionally minimal. Additional directories may be added later through ADRs.

The purpose of this structure is to:

* Separate experimentation from production code
* Enforce modular design
* Provide predictable navigation for humans and AI agents
* Ensure documentation mirrors implementation

---

# Top-Level Structure

```
repo/
├── .devcontainer/
├── .github/
│   └── workflows/
├── docs/
├── models/
├── notebooks/
├── scripts/
├── src/
├── experiments/
├── tests/
├── environment/
└── README.md
```

Local/generated directories such as `__pycache__/` and `.pytest_cache/` may
exist in developer environments but are not part of the canonical repository
structure.

---

# Directory Definitions

## .github/workflows/

**Purpose:**
Contains GitHub Actions CI workflows.

**Rules:**

* All CI/CD automation must live here.
* Workflows must be documented in `docs/ci/`.
* No experimental scripts belong here.

---

## docs/

**Purpose:**
Authoritative documentation for repository structure, coding standards, AI usage policy, CI documentation, and module-level documentation.

**Structure (initial):**

```
docs/
├── templates/
│   └── README.md
├── repo_structure.md
```

**Rules:**

* Documentation mirrors repository structure where applicable.
* Structural code changes require corresponding documentation updates.
* Templates in `docs/templates/` are authoritative.

---

## notebooks/

**Purpose:**
Exploration, experimentation, visualization, and demo notebooks.

**Rules:**

* Notebooks may import from `src/`.
* `src/` must not depend on notebooks.
* Core logic must be implemented in `src/` before being considered stable.
* Notebooks must follow the Notebook Header Template.

---

## src/

**Purpose:**
Primary source code for reusable, production-quality logic.

**Characteristics:**

* Modular
* Deterministic
* Testable
* Independent of notebooks

**Rules:**

* All reusable logic must live here.
* Files must include the Source Code File Header Template.
* Public functions/classes must include standardized docstrings.

Submodules will be defined as development progresses.

---

## experiments/

**Purpose:**
Stores structured experiment artifacts and experiment-level documentation.

**Characteristics:**

* May reference notebooks
* May reference specific model versions
* Not production code

**Rules:**

* Each experiment must have a documented objective.
* Results summaries must be written (not just raw output files).

---

## tests/

**Purpose:**
Unit and integration tests for `src/` modules.

**Rules:**

* Tests must not depend on notebooks.
* Tests should validate deterministic behavior where applicable.
* Critical evaluation logic must be tested.

---

## environment/

**Purpose:**
Defines reproducible development environments.

**Examples:**

* `environment/requirements.txt`
* `environment.yml`
* Colab setup instructions

**Rules:**

* Dependency changes require documentation updates.
* Environment files must remain minimal and explicit.

---

## models/

**Purpose:**
Stores local model artifacts and model-related notes used during experimentation.

**Rules:**

* Treat this directory as non-source artifact space.
* Keep reproducibility by documenting how artifacts are generated.
* Large binaries should remain ignored unless explicitly required.

---

## scripts/

**Purpose:**
Houses utility scripts for data/bootstrap tasks that are not part of `src/` runtime modules.

**Rules:**

* Scripts may call into `src/` modules but should not duplicate core logic.
* Script behavior that becomes reusable should be promoted into `src/`.

---

## .devcontainer/

**Purpose:**
Development-container configuration for consistent local environments.

**Rules:**

* Keep environment/tooling setup here only.
* Do not place project runtime/business logic in this directory.

---

# Structural Principles

1. Separation of Concerns

   * Exploration (notebooks)
   * Stable logic (src)
   * Automation (.github)
   * Documentation (docs)

2. One-Way Promotion

   * Ideas move from notebooks → src.
   * Stable implementations may be demonstrated in notebooks.

3. Documentation Parity

   * Structural changes must be reflected in documentation.

4. Minimalism First

   * Only create new top-level directories via ADR.

---

This structure is intentionally conservative. It establishes discipline without over-engineering the project at this stage.

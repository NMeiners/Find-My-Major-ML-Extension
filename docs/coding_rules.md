# Coding Rules

This document defines mandatory coding standards for this repository.

It is designed to:

* Prevent low-quality or unverified AI-generated code from entering the codebase
* Enforce modular, maintainable architecture
* Preserve scientific integrity and reproducibility
* Allow structured experimentation appropriate for a research prototype

These rules apply to all contributors (human and AI).

---

# 1. Core Philosophy

1. Clarity over cleverness.
2. Determinism over convenience.
3. Modular design over monolithic scripts.
4. Explicit structure over implicit behavior.
5. Controlled experimentation over uncontrolled improvisation.

This is a research prototype. Exploration is encouraged — but architectural drift is not.

---

# 2. Spec-First Development Rule

Before modifying or creating code in a module:

1. Read the corresponding documentation in `docs/src/<module>.md` (if it exists).
2. Do not invent public interfaces not documented.
3. If a required interface does not exist:

   * Update the relevant documentation first, OR
   * Create/update an ADR if the change is architectural.

AI agents must not invent APIs, behaviors, or cross-module interactions that are not documented.

---

# 3. Anti–AI Slop Safeguards

These rules specifically protect against low-quality or hallucinated AI code.

## 3.1 No Blind Generation

* AI-generated code must be reviewed before merge.
* Large, unexplained code dumps are not allowed.
* Generated code should be broken into logical, reviewable commits.

## 3.2 Mandatory Templates

* All new source files must include the Source Code File Header Template.
* All public functions/classes must use the standardized docstring template.
* All private functions/classes must include a brief comment explaining their purpose.
* Missing headers or incomplete template fields are grounds for rejection.

## 3.3 Single Responsibility Enforcement

* Each file must have one primary responsibility.
* If a file grows beyond ~400 lines or mixes concerns, it should be refactored.

## 3.4 No Hidden Production Logic in Notebooks

* Notebooks may explore or prototype logic.
* Reusable logic must be migrated to `src/`.
* `src/` must never depend on notebooks.

## 3.5 No Silent Dependency Introduction

* New dependencies require:

  * Update to `environment/`
  * Clear commit explanation
  * Justification of necessity

## 3.6 No Silent Metric Redefinition

* Evaluation metrics must be implemented centrally.
* Models must not redefine ranking or similarity metrics internally.
* Changes to metric definitions require documentation update.

---

# 4. Modularity and Architecture Rules

## 4.1 Single Responsibility per Module

Each module must:

* Own a clearly defined concern
* Explicitly state what it does NOT handle

## 4.2 Explicit Interfaces

* Public functions must define clear inputs and outputs.
* No reliance on global state.
* No hidden side effects.

## 4.3 Dependency Direction (Initial Constraint)

Within `src/`, dependency direction should follow this general flow:

* `data/` → may be used by all other modules
* `models/` → may depend on `data/`
* `evaluation/` → may depend on `data/` but not on specific model implementations
* `inference/` → may depend on `models/` and `data/`
* `export/` → may depend on `models/` only

Circular dependencies are not allowed.

These constraints may evolve via ADR if needed.

## 4.4 One-Way Promotion Model

* Ideas move from notebooks → `src/`.
* Stable implementations in `src/` may be demonstrated in notebooks.
* Experimental logic should not bypass this promotion flow.

---

# 5. Data Contracts and Ethical Constraints

This project explicitly avoids demographic or protected attributes.

## 5.1 Data Ownership

* All core data structures must be defined in `src/data/`.
* New features or fields require documentation updates.

## 5.2 Prohibited Features

The following must NOT be introduced:

* Demographic attributes
* Protected characteristics
* Personally identifiable information (PII)

## 5.3 Schema Stability

* Changes to data schemas require documentation updates.
* Model code must not silently assume undocumented fields.

## 5.4 Invariants

The following invariants must be preserved unless formally changed:

* RIASEC vectors must remain length 6.
* Output rankings must be deterministic for identical inputs.
* Scores must fall within documented ranges.

---

# 6. Reproducibility and Determinism

* Random seeds must be explicitly set in experiments and training.
* Non-deterministic behavior must be documented.
* Evaluation results must be reproducible given the same seed and data.

Research flexibility is allowed, but reproducibility must be possible.

---

# 7. Notebook Rules (Flexible but Controlled)

Notebooks must:

* Follow the Notebook Header Template
* Declare seed and environment
* Import reusable logic from `src/`
* Avoid hardcoded local paths

Before commit:

* Restart kernel
* Run all cells
* Remove unnecessary outputs

Notebooks may contain exploratory logic, but that logic must not become production code without migration.

---

# 8. Testing Expectations

This is a research prototype, so testing requirements are proportional but meaningful.

Required for `src/` modules:

* Unit tests for evaluation and ranking functions
* Deterministic behavior tests where applicable
* Serialization/export tests (if exporting models)

Encouraged but not mandatory for early experimentation:

* Synthetic data sanity checks
* Integration tests for full inference flow

Tests must not depend on notebooks.

---

# 9. AI Agent Operating Constraints

AI agents must:

* Read relevant module documentation before modifying code
* Follow all templates verbatim
* Respect dependency direction rules
* Avoid architectural refactors unless explicitly instructed
* Avoid introducing new top-level directories

AI agents may:

* Refactor within a single module for clarity
* Improve naming, documentation, and structure
* Suggest improvements, but not enforce them without instruction

AI agents must not:

* Merge notebook experimentation directly into `src/`
* Rewrite large cross-module sections without instruction
* Invent undocumented interfaces or behaviors

---

# 10. Commit Discipline

Commits should:

* Be small and logically grouped
* Include clear explanation of intent
* Reference affected modules

Avoid vague commit messages such as:

* "update"
* "misc changes"
* "AI fix"

---

# 11. Escalation Rule

If a change affects:

* Architecture
* Evaluation methodology
* Data contracts
* CI workflows
* Dependency direction

An ADR must be created or updated.

---

# 12. Research Flexibility Clause

Because this is a research prototype:

* Exploration is encouraged.
* Architectural evolution is allowed.
* Refactoring is expected.

However:

* Structural changes must be intentional.
* Documentation must stay synchronized.
* Core invariants must not drift silently.

Discipline enables flexibility. Not the reverse.

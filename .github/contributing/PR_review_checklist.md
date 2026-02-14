# Pull Request (PR) Review Checklist

This document defines the required review standards before merging into `main`.

The purpose of this checklist is to:

* Prevent architectural drift
* Block low-quality or AI-generated slop
* Enforce reproducibility standards
* Maintain documentation–code parity
* Protect evaluation integrity

All PRs must satisfy the relevant sections below.

---

# 1. Structural Integrity

* [ ] Code is placed in the correct directory (see `repo_structure.md`).
* [ ] No business logic exists inside notebooks.
* [ ] Evaluation logic is not embedded inside model modules.
* [ ] `tests/` mirrors `src/` for new or modified modules.
* [ ] No large artifacts (models, datasets, logs) are committed.

---

# 2. Coding Standards

* [ ] File headers follow the standardized template.
* [ ] AI usage is disclosed in headers (if applicable).
* [ ] Code adheres to `coding_rules.md`.
* [ ] No monolithic functions without justification.
* [ ] No undocumented public interfaces.
* [ ] No silent dependency additions.

---

# 3. Tests and CI

* [ ] All tests pass locally (`pytest`).
* [ ] New logic includes appropriate tests.
* [ ] Tests are deterministic (fixed seeds where required).
* [ ] CI passes.

---

# 4. Evaluation Governance (If Applicable)

* [ ] Core metrics are unchanged unless documented.
* [ ] Evaluation logic remains centralized.
* [ ] Dataset splits are unchanged or properly documented.
* [ ] Baseline comparison is included for new models.
* [ ] Fairness metrics are included if demographic attributes are present.

If evaluation methodology changed:

* [ ] Documentation updated.
* [ ] ADR created (if architectural).

---

# 5. Data Governance (If Applicable)

* [ ] Dataset version is specified.
* [ ] Preprocessing changes are documented.
* [ ] No raw data modified in place.
* [ ] No evidence of data leakage.
* [ ] Sensitive attribute handling is explicit.

---

# 6. Experiment Tracking (If Applicable)

* [ ] Experiment ID is defined (EXP-YYYYMMDD-XX).
* [ ] Structured metadata is generated.
* [ ] Results are reproducible from commit + seed.
* [ ] No experiment claims rely solely on notebook output.

---

# 7. Documentation–Code Parity

* [ ] Relevant documentation updated.
* [ ] Module documentation reflects interface changes.
* [ ] Evaluation or data workflow documentation updated if needed.

No code-only PRs that change behavior without documentation updates.

---

# 8. AI Output Review

If AI was used:

* [ ] Output reviewed line-by-line.
* [ ] No hallucinated APIs or interfaces.
* [ ] No fabricated metrics or results.
* [ ] No architectural expansion beyond scope.

AI assistance does not reduce review standards.

---

# 9. Reviewer Final Questions

Before approving, the reviewer should ask:

1. Does this change increase structural clarity?
2. Is the behavior reproducible?
3. Would a new contributor understand this module?
4. Does this introduce hidden coupling?
5. Is documentation synchronized with code?

If any answer is "no" or "uncertain," request revision.

---

This checklist operationalizes the repository's governance framework. Passing CI is necessary but not sufficient for approval.

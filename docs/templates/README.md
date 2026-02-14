# Documentation Templates

This document defines **copyable, standardized templates** used across the repository. These templates are designed to:

* Be easy for humans to follow
* Be deterministic and predictable for AI agents
* Enforce modular, maintainable structure
* Reduce ambiguity in authorship, intent, and constraints

All templates below are **authoritative**. When creating new files, notebooks, or modules, contributors (human or AI) should copy the relevant template and fill it in completely.

---

## 1. Source Code File Header Template

**Use for:** All `.py` (and other source) files in `src/`, `tests/`, `ci/`, etc.

**Located at:** docs/templates/file_header.md

---

## 2. Function / Class Docstring Template

**Use for:** All public functions, classes, and methods.

**Located at:** docs/templates/function-class_header.md

---

## 3. Module Documentation Template

**Use for:** Files under `docs/src/` that describe each top-level module.

**Located at:** docs/templates/module_doc.md

---

## 4. Notebook Header Template

**Use for:** All Jupyter notebooks in the `notebooks/` directory.

**Located at:** docs/templates/notebook_header.md

---

## 5. Experiment Documentation Template

**Use for:** Experiment descriptions under `experiments/` or `docs/experiments/`.

**Located at:** docs/templates/experiment_doc.md

---

## 6. ADR (Architecture Decision Record) Template

**Use for:** Significant technical or architectural decisions.

**Located at:** docs/templates/adr.md

---

## Usage Rules (Critical)

* Templates must be copied verbatim before being filled in.
* Fields should not be removed; use `None` if not applicable.
* AI-generated code must include completed headers.
* Documentation updates must accompany structural or behavioral code changes.

These templates are intentionally rigid. Consistency is more valuable than brevity in this project.

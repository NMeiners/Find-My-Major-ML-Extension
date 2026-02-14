# Documentation Guide

This document explains how to use the `docs/` directory.

The documentation system is designed to:

* Provide structural clarity
* Enforce modular development
* Constrain AI-generated output
* Keep architecture intentional during research iteration

The documentation is not decorative. It is operational.

---

# 1. How to Use the Documentation

## Before Writing Code

1. Read `repo_structure.md` to understand where your change belongs.
2. Read `coding_rules.md` to understand constraints.
3. If modifying an existing module, read its corresponding file under `docs/src/` (when defined).
4. If using AI, begin with a prompt header from `ai_usage_policy.md`.

---

## When Creating New Files

1. Copy the appropriate template from `docs/templates/`.
2. Fill in all required fields.
3. Ensure the file is placed in the correct directory.
4. Update relevant documentation if structure or behavior changes.

---

## When Modifying Existing Code

1. Confirm the change aligns with module responsibility.
2. Avoid expanding module scope without documentation updates.
3. Update documentation if public interfaces change.
4. Create or update an ADR if architectural boundaries are affected.

---

## When Using Notebooks

1. Use notebooks for exploration or demonstration only.
2. Promote reusable logic to `src/`.
3. Follow the Notebook Header Template.
4. Restart and run all cells before committing.

---

# 2. Documentation–Code Parity Rule

Code and documentation must evolve together.

If you:

* Change public interfaces
* Modify evaluation methodology
* Add dependencies
* Alter data schemas
* Adjust module responsibilities

Then documentation must be updated in the same change.

---

# 3. AI Workflow Summary

1. Select the appropriate prompt header from `ai_usage_policy.md`.
2. Provide explicit task boundaries.
3. Review AI output line-by-line.
4. Ensure templates are used.
5. Verify compliance with `coding_rules.md`.

AI is a tool. Documentation is the control surface.

---

# 4. When in Doubt

If a change feels architectural rather than incremental:

* Pause.
* Document the intent.
* Create or update an ADR.

Intentional structure is more important than speed.

---

This documentation system is lightweight but strict. Its purpose is to enable experimentation without sacrificing maintainability.

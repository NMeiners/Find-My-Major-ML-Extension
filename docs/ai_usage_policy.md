# AI Usage Policy

This document defines how AI tools (e.g., ChatGPT, Codex, Copilot) may be used in this repository.

The goal is to:

* Prevent low-quality or hallucinated output
* Ensure AI contributions follow repository structure and standards
* Keep humans accountable for architectural decisions
* Make AI usage predictable and reproducible

This document is intentionally brief. It provides **copyable prompt headers** to guide correct AI interaction.

---

# 1. General Principles

1. AI is an assistant, not an authority.
2. Humans remain responsible for reviewing all AI-generated code.
3. AI must be explicitly directed to follow repository documentation.
4. AI may not define architecture unless explicitly instructed.
5. AI output must comply with `coding_rules.md` and repository templates.

---

# 2. Required Disclosure

When AI generates or significantly modifies code:

* The file header must list the AI tool under "AI Tools Used".
* The human reviewer must be listed under "Editors".

Undisclosed AI-generated code is not permitted.

---

# 3. Approved AI Task Categories

AI may be used for:

* Generating modular code within defined boundaries
* Refactoring within a single module
* Writing tests
* Generating documentation using repository templates
* Debugging specific, well-defined issues
* Translating notebook logic into `src/` modules

AI may not be used to:

* Define new architecture without instruction
* Modify evaluation methodology without instruction
* Introduce new dependencies silently
* Add new data features without documentation updates

---

# 4. Copyable Prompt Headers

All AI interactions related to this repository should begin with one of the following structured prompts.

---

## 4.1 Code Generation Prompt

Use when generating new source code.

```
While adhering strictly to the rules in docs/coding_rules.md and the relevant module documentation, generate modular, maintainable code for the following task:

<Task description>

Constraints:
- Follow repository templates
- Do not invent undocumented interfaces
- Do not introduce new dependencies
- Keep logic within the appropriate module
```

---

## 4.2 Refactor Prompt

Use when improving existing code.

```
While adhering to docs/coding_rules.md, refactor the following code for clarity and modularity.

Constraints:
- Do not change external behavior
- Do not modify evaluation logic
- Do not introduce new dependencies
- Keep changes within the current module

<Code snippet>
```

---

## 4.3 Bug Fix Prompt

Use when fixing a specific issue.

```
While adhering to docs/coding_rules.md, fix the following bug.

Requirements:
- Preserve documented behavior
- Do not introduce architectural changes
- Add or update tests if necessary

<Problem description or error>
```

---

## 4.4 Test Generation Prompt

Use when creating tests for existing logic.

```
While adhering to docs/coding_rules.md, generate unit tests for the following module:

<Module name>

Constraints:
- Tests must be deterministic
- Do not depend on notebooks
- Cover edge cases and documented invariants
```

---

## 4.5 Documentation Update Prompt

Use when updating module documentation.

```
Update the relevant documentation in docs/ to reflect the following change.

Constraints:
- Follow documentation templates
- Do not invent undocumented behavior
- Ensure consistency with coding_rules.md

<Description of change>
```

---

## 4.6 Notebook-to-Source Promotion Prompt

Use when converting exploratory notebook logic into production code.

```
Migrate the following notebook logic into src/ while adhering strictly to docs/coding_rules.md.

Requirements:
- Modularize logic appropriately
- Add standardized headers and docstrings
- Ensure deterministic behavior
- Do not embed experimental shortcuts

<Notebook code or description>
```

---

## 4.7 Repository Audit Prompt

Use when asking AI to validate repository integrity.

```
You are conducting a repository integrity audit. Do NOT modify any files.

Your task is to analyze the entire repository (docs/, src/, notebooks/, environment/, .github/) and report any inconsistencies, violations, or structural drift relative to the documented standards.

Validation Requirements:

1. Coding Rules Compliance
   - Identify violations of docs/coding_rules.md
   - Flag undocumented public interfaces
   - Detect circular dependencies (if inferable)
   - Flag production logic inside notebooks
   - Identify missing required headers or docstrings

2. Documentation–Code Parity
   - Identify mismatches between docs/src/ and actual src/ modules
   - Flag undocumented functions or classes
   - Flag documented interfaces that do not exist
   - Detect outdated or contradictory documentation

3. Dependency and Structure Validation
   - Identify undeclared dependencies
   - Flag improper dependency direction between modules
   - Detect new top-level directories not defined in repo_structure.md

4. Data and Evaluation Integrity
   - Flag silent metric redefinitions
   - Identify schema changes not reflected in documentation
   - Flag prohibited feature usage (demographic, protected attributes)

5. AI Usage Compliance
   - Flag missing AI disclosure in file headers
   - Identify large monolithic files that violate modularity expectations

Output Format:

- Provide findings grouped by category.
- For each issue:
  - File path
  - Severity (Low / Moderate / High / Critical)
  - Description of violation
  - Reference to the specific rule being violated
- Do NOT propose code rewrites.
- Do NOT modify any content.
- Conclude with an overall repository integrity assessment summary.

This is an audit-only task. Do NOT generate or modify any code or documentation.
```

---

# 5. AI Usage Red Flags

Human reviewers should reject AI output that:

* Introduces undocumented interfaces
* Collapses multiple concerns into one file
* Computes metrics inside model logic
* Adds demographic or protected attributes
* Bypasses documentation updates
* Rewrites large sections without justification

---

# 6. Review Responsibility

AI output must always be:

* Read line-by-line
* Checked against documentation
* Tested locally

Speed is secondary to correctness and maintainability.

---

This policy ensures AI remains a disciplined assistant aligned with repository standards rather than an uncontrolled code generator.

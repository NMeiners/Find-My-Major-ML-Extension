# Data Pipeline Architecture: RIASEC → O\*NET → Enriched Training Dataset

## Overview

This document describes the end-to-end data build pipeline that transforms the raw
Kaggle Holland Code (RIASEC) Test Responses dataset into an enriched training dataset
with O\*NET-derived RIASEC scores and career recommendations.

The enriched dataset is the input to the ML model — it is not produced by the model.

---

## Pipeline Diagram

```
[Kaggle Raw CSV — 48 questions per respondent]
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1 — Ingest & Clean                                   │
│  src/data/kaggle_loader.py                                  │
│  • Load raw CSV                                             │
│  • Drop rows with nulls or out-of-range values (not 1–5)   │
│  • Remove bots (all-same answer rows)                       │
│  • Remove duplicates                                        │
│  Output → data/interim/cleaned_48q.parquet                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2 — Map 48Q → 60Q  +  Fill Gaps with Dataset Avg    │
│  src/data/response_mapper.py                                │
│                                                             │
│  Using data/raw/question_mapping.json:                      │
│    ├─ EXACT match  → copy Kaggle answer directly            │
│    ├─ CLOSE match  → copy Kaggle answer directly            │
│    └─ NO mapping   → fill with dataset-level average for    │
│                       that O*NET question's RIASEC area     │
│                       (computed across all cleaned rows)    │
│                                                             │
│  Result: every respondent now has a full 60-answer vector   │
│  Output → data/interim/mapped_60q.parquet                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3 — Score via O*NET Interest Profiler API            │
│  src/data/onet_scorer.py                                    │
│  • Submit each 60-answer row to O*NET scoring endpoint      │
│  • Returns R, I, A, S, E, C scores (0–40 each)             │
│  • Batch with rate-limit handling + retry/backoff           │
│  • Checkpoint every N rows → resume on failure              │
│  Output → data/interim/scored_60q.parquet                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4 — Career Recommendations via O*NET                 │
│  src/data/career_fetcher.py                                 │
│  • From each row's R,I,A,S,E,C scores → identify top areas │
│  • Query O*NET careers endpoint per top area + score        │
│  • Cache results by (area, score_bucket) to limit API calls │
│  • Collect top-N career titles + SOC codes per respondent   │
│  Output → data/interim/careers.parquet                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5 — Assemble Final Training Dataset                  │
│  src/data/dataset_builder.py                                │
│  • Join: original 48Q answers + RIASEC scores + careers     │
│  • Final schema (one row = one respondent):                 │
│      q1…q48        raw Kaggle responses                     │
│      r,i,a,s,e,c   O*NET scores (0–40 each)                │
│      top_area_1/2  primary & secondary Holland code         │
│      career_1…N    recommended career titles                │
│      soc_1…N       SOC codes                               │
│  • Train / val / test split (70 / 15 / 15)                 │
│  Output → data/processed/riasec_enriched.parquet            │
│           data/processed/train.parquet                      │
│           data/processed/val.parquet                        │
│           data/processed/test.parquet                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Stage Details

### Stage 1 — Ingest & Clean (`kaggle_loader.py`)

The Kaggle dataset contains responses to a 48-item RIASEC Interest Profiler on a
1–5 Likert scale. Cleaning rules:

| Check | Action |
|---|---|
| Any answer outside 1–5 | Drop row |
| Any null / missing answer | Drop row |
| All 48 answers identical | Drop row (bot detection) |
| Exact duplicate rows | Drop, keep first |

Output is a clean DataFrame with standardized column names `q1…q48`.

### Stage 2 — Map & Fill (`response_mapper.py`)

Uses `data/raw/question_mapping.json` (48 entries, already built) to translate each
Kaggle question to its O\*NET 60-item equivalent.

**Gap-fill logic:** O\*NET questions that have no Kaggle equivalent are filled with
the **dataset-level mean** for that question's RIASEC area, computed once across all
cleaned rows. This gives every respondent a valid 60-answer vector while preserving
all individual signal from the 48 mapped questions.

### Stage 3 — Score (`onet_scorer.py`)

O\*NET Interest Profiler scoring endpoint:
```
GET /ws/mnm/interestprofiler/results?answers=<60 comma-separated values>
```

Returns six integer scores (R, I, A, S, E, C), each in the range 0–40.
Checkpointing ensures the stage can resume mid-dataset if the API call fails or
rate limits are hit.

### Stage 4 — Career Enrichment (`career_fetcher.py`)

O\*NET careers endpoint:
```
GET /ws/mnm/interestprofiler/results/{area}/{score}/careers
```

Called for the top 1–2 RIASEC areas per respondent. Results are cached by
`(area, score_bucket)` so repeated score combinations do not trigger redundant
API calls.

### Stage 5 — Assemble (`dataset_builder.py`)

Joins all interim outputs on row index and writes the final schema. The train/val/test
split is applied here (before any model sees the data) to prevent leakage.

---

## Directory Layout

```
data/
  raw/
    kaggle_riasec.csv                   ← place Kaggle download here
    interest_profiler_questions.json    ← O*NET 60Q (already fetched)
    question_mapping.json               ← 48→60 mapping (already built)
  interim/                              ← scratch files, not committed
    cleaned_48q.parquet
    mapped_60q.parquet
    scored_60q.parquet
    careers.parquet
  processed/                            ← final outputs, committed
    riasec_enriched.parquet
    train.parquet
    val.parquet
    test.parquet

src/data/
  schemas.py          ← extend with RIASECScore, CareerResult, EnrichedRow
  kaggle_loader.py    ← Stage 1  (to be implemented)
  response_mapper.py  ← Stage 2  (to be implemented)
  onet_scorer.py      ← Stage 3  (to be implemented)
  career_fetcher.py   ← Stage 4  (to be implemented)
  dataset_builder.py  ← Stage 5  (to be implemented)

scripts/
  build_dataset.py    ← orchestrator: runs all 5 stages in sequence
```

---

## Orchestrator Usage

```bash
python scripts/build_dataset.py \
  --kaggle-input  data/raw/kaggle_riasec.csv \
  --output-dir    data/processed/ \
  --batch-size    100 \
  --top-n-careers 5
```

Each stage reads from the previous stage's parquet output, so any single stage can
be re-run independently without restarting the full pipeline.

---

## New Schemas (to add to `schemas.py`)

| Schema | Fields |
|---|---|
| `RIASECScore` | `r, i, a, s, e, c: int` (each 0–40) |
| `CareerResult` | `soc_code: str, title: str, area: str, score_bucket: int` |
| `EnrichedRow` | `raw_answers: list[int]`, `riasec_score: RIASECScore`, `careers: list[CareerResult]` |

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Parquet for interim files | Typed columns, fast I/O, preserves dtypes across stages |
| Dataset-level average for gap-fill | Uses full dataset signal; keeps all respondents equal on unmapped slots |
| Checkpoint every N rows in Stage 3 | O\*NET API has rate limits; large datasets need resume-on-failure |
| Cache career lookups by `(area, score_bucket)` | Many respondents share the same top area + score; avoids redundant calls |
| Keep raw 48Q answers in final dataset | Model may learn directly from raw responses, not only derived scores |
| Train/val/test split at build time | Prevents data leakage before any model sees the data |

---

## Related Documentation

- `docs/data/data_governance.md`
- `docs/src/data.md`
- `data/raw/question_mapping.json`

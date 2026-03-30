# Module: src/models

## Responsibility
Owns model abstractions and model implementations for RIASEC-based career recommendation.
Model classes predict candidate career categories and then delegate job ranking to a shared ranking utility.

This module does NOT load raw CSVs, run dataset validation, or compute evaluation metrics.

## Public Interfaces
- `BaseModel`
  - `train(X_train, y_train)`
  - `test(X_test, onet_db)`
  - `get_name()`
- Model classes:
  - `HeuristicModel`
  - `LogisticRegressionModel`
  - `RandomForestModel`
  - `GradientBoostingModel`
  - `KNNModel`
- Shared functions:
  - `rank_jobs(student_vector, onet_db, feature_cols, top_n_categories, top_n_jobs, category_col, predicted_categories)`
  - `run_inference(model, X_test, onet_db)`
- Registry:
  - `MODEL_REGISTRY`
- Legacy compatibility API (maintained for notebooks):
  - `load_pipeline_artifacts(model_path, db_path)`
  - `get_job_recommendations(student_vector, rf_model, onet_db, top_n_categories=3, top_n_jobs=3)`

## Internal Structure
- `base.py` — abstract base contract
- `heuristic.py`, `logistic_regression.py`, `random_forest.py`, `gradient_boosting.py`, `knn.py` — model implementations
- `ranking.py` — centralized cosine-similarity ranking
- `inference.py` — model-agnostic inference entry point
- `baseline.py` — legacy procedural notebook helper
- `__init__.py` — exports classes/functions/registry

## Data Contracts
- Inputs:
  - `X_train` / `X_test`: DataFrames containing configured `x_features`
  - `y_train`: Series containing configured label column
  - `onet_db`: DataFrame containing ranking features, `Career Category`, and `Title`
- Outputs:
  - Model `test()` methods return DataFrame columns `[Title, Career Category, Match_Score]`
  - Returned row count is bounded by `top_n_jobs` (may be fewer when candidate pool is smaller)

## Config Keys
- `models[*].model` selects class via `MODEL_REGISTRY`
- `models[*].x_features` and `models[*].y_features` define feature/label columns
- `models[*].parameters.top_n_categories` is extracted before sklearn estimator construction
- `models[*].parameters.use_smote` (optional, RandomForestModel only) enables SMOTE balancing before fit
- `models[*].parameters.smote` (optional, RandomForestModel only) passes kwargs to `imblearn.over_sampling.SMOTE`
- `evaluation.top_k` controls `top_n_jobs`

## Constraints
- Ranking behavior is centralized in `ranking.py` to avoid per-model metric drift.
- No hardcoded feature names in model logic; values come from config.
- Legacy baseline helpers remain for notebook compatibility and are not used by the class-based training pipeline.

## Related Modules
- `src/data`
- `src/evaluation`
- `src/config`

## Related Documentation
- `docs/src/models/models.md`
- `docs/src/config/config_loader.md`
- `docs/templates/exp_config.yaml`
- `experiments/config/exp_config.yaml`

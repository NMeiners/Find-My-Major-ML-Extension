
# Module: src/models

## Responsibility
Owns all model classes, training logic, and job ranking for the RIASEC recommendation pipeline.
This module does NOT handle data loading, feature engineering, evaluation metrics, or inference serving.
All configuration values (features, hyperparameters, result counts) come from the experiment config — nothing is hardcoded.

## Public Interfaces

- `BaseModel`: Abstract base class all model implementations must inherit from.
    - Constructor: `BaseModel(x_features, y_feature, parameters, top_n_jobs, top_n_categories)`
    - `train(X_train, y_train)`: Fit the model on training data.
    - `test(X_test, onet_db)`: Return top-N job recommendations as a DataFrame.
    - `get_name()`: Return the model's string identifier (must match config key).

- `HeuristicModel(BaseModel)`: Rule-based baseline. No training required. Ranks all jobs in onet_db by cosine similarity (no category filter) since it has no classifier to map feature names to category labels.
- `LogisticRegressionModel(BaseModel)`: sklearn LogisticRegression classifier.
- `RandomForestModel(BaseModel)`: sklearn RandomForestClassifier (refactored from baseline.py).
- `GradientBoostingModel(BaseModel)`: sklearn GradientBoostingClassifier.
- `KNNModel(BaseModel)`: sklearn KNeighborsClassifier.

- `rank_jobs(student_vector, onet_db, feature_cols, top_n_categories, top_n_jobs, category_col, predicted_categories)`: Shared cosine similarity ranking function used by all models.
    - `sklearn.metrics.pairwise.cosine_similarity`: external

- `run_inference(model, X_test, onet_db)`: Runs a trained model's test() method. Entry point for the Export/Inference module.

- `MODEL_REGISTRY`: Dict mapping config string keys to model classes. Used by main.py to instantiate models from config.

## Internal Structure
- `base.py` — Abstract base class `BaseModel`
- `ranking.py` — Shared `rank_jobs()` cosine similarity ranking function
- `heuristic.py` — `HeuristicModel`: dominant-RIASEC rule-based model
- `logistic_regression.py` — `LogisticRegressionModel`
- `random_forest.py` — `RandomForestModel` (refactored from baseline.py)
- `gradient_boosting.py` — `GradientBoostingModel`
- `knn.py` — `KNNModel`
- `inference.py` — `run_inference()` entry point
- `baseline.py` — Legacy functions; kept for notebook compatibility only. Do not add to this file.
- `__init__.py` — Exports all public classes, `MODEL_REGISTRY`, and `run_inference`

## Data Contracts
- Inputs:
    - `X_train` / `X_test`: `pd.DataFrame` with columns matching `x_features` from config
    - `y_train`: `pd.Series` with career category labels
    - `onet_db`: `pd.DataFrame` loaded from `onet_db_path` (top-level config key); must contain `feature_cols` and `category_col`
- Outputs:
    - `test()` returns a `pd.DataFrame` with exactly `top_n_jobs` rows and columns: `[Title, Career Category, Match_Score]`
    - `Match_Score` is a cosine similarity value in [0, 1]

## Config Keys
Model classes are instantiated by `main.py` from the experiment config. The `get_name()` return value must match the config key:
- `"heuristic"` → `HeuristicModel`
- `"logistic_regression"` → `LogisticRegressionModel`
- `"random_forest"` → `RandomForestModel`
- `"gradient_boosting"` → `GradientBoostingModel`
- `"knn"` → `KNNModel`

All values read from config at construction — no hardcoded feature names, counts, or hyperparameters.

**Important:** `top_n_categories` is stored under each model's `parameters` block in the config but is **removed before** the remaining parameters are passed to sklearn. This prevents sklearn from receiving an unknown keyword argument. `main.py` handles this extraction via `parameters.pop('top_n_categories', 3)`.

## Constraints
- Performance: Models must complete training on the full dataset in reasonable time (sklearn classifiers, no deep learning)
- Memory: All models must fit in memory alongside the O*NET career database
- Privacy: No PII or demographic attributes may be used as features
- Bias considerations: Features are limited to RIASEC scores and documented extensions; no protected characteristics

## Related Modules
- src/data (produces X_train, y_train via DataLoader)
- src/evaluation (consumes trained model array and onet_db)
- src/export (consumes run_inference)

## Related Documentation
- docs/src/config/config_loader.md
- docs/templates/exp_config.yaml
- experiments/config/exp_config.yaml

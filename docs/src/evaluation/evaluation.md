# Module: src/evaluation

## Responsibility
Owns evaluation orchestration and reporting for trained recommendation models.
This includes:
- Running model evaluation across datasets
- Computing rank-based recommendation metrics
- Benchmarking latency/memory/model-size proxies
- Saving/loading formatted evaluation artifacts
- Providing fairness utility functions for category-distribution analysis

This module does NOT implement model training algorithms or raw CSV loading.

## Public Interfaces
- `Dataset(train_records, test_records, feature_columns, label_column='career_category')`
  - `split()` returns train/test DataFrames
- `evaluate_experiment(datasets, models, onet_db, config, output_path=None)`
- `evaluate_model(model, dataset, onet_db, config)`
- `save_evaluation_results(results, output_path)`
- `compute_all_metrics(predictions, ground_truth_category, k_values)`
- `ndcg_at_k(predictions, ground_truth_category, k)`
- `precision_at_k(predictions, ground_truth_category, k)`
- `cosine_similarity(vector1, vector2)`
- `benchmark_model_inference(model, test_data, onet_db, num_runs=5)`
- `measure_latency_ms(func, *args, **kwargs)`
- `estimate_memory_usage(obj)`
- `get_model_size_mb(model)`
- `format_evaluation_results(results)`
- `save_results_to_file(results, output_dir)`
- `load_results_from_file(input_dir)`
- Fairness utilities:
  - `compute_category_distribution(recommendations, category_names)`
  - `compute_fairness_score(distribution, ideal_distribution)`
  - `detect_overrepresented_categories(distribution, ideal_distribution, threshold=10.0)`
  - `detect_underrepresented_categories(distribution, ideal_distribution, threshold=10.0)`

## Internal Structure
- `evaluator.py` — orchestration loop + `Dataset`
- `metrics.py` — recommendation metrics (`ndcg@k`, `precision@k`) + vector similarity helper
- `benchmark.py` — inference latency/memory/model-size approximation
- `reporting.py` — formatting and file persistence helpers
- `fairness.py` — category distribution/fairness helper functions
- `__init__.py` — public re-exports

## Data Contracts
- Inputs:
  - `datasets`: list of `Dataset`
  - `models`: list implementing `train`, `test`, `get_name`
  - `onet_db`: DataFrame used by model `test()` for ranking
  - `config.evaluation` keys currently consumed by evaluator:
    - `metrics` (optional; supported: `ndcg_at_k`, `precision_at_k`; defaults to both)
    - `k_values` (optional, defaults to `[top_k]`; each value must be `<= top_k`)
    - `top_k` (optional, defaults to `5`)
    - `benchmark_runs` (optional, defaults to `1`)
    - `max_test_samples` (optional)
    - `progress_log_interval` (optional, defaults to `1000`; logs per-model sample progress)
- Outputs:
  - Per model/dataset result dict containing:
    - `metrics`
    - `latency_ms`
    - `memory_bytes`
    - `model_size_mb`
    - `constraint_violations`
    - `samples_evaluated`
    - `interrupted` (optional; `true` when user interruption occurs after partial sample evaluation)

## Constraints
- Metric implementations are centralized in `metrics.py`.
- Unsupported values in `evaluation.metrics` raise a configuration error.
- Invalid `k_values` (non-positive or greater than `top_k`) raise a configuration error.
- Evaluation loop computes per-sample metrics and aggregates averages.
- Evaluator enforces model prediction contract: output must be a DataFrame with
  `Title`, `Career Category`, and `Match_Score`.
- `constraint_violations` reports detected prediction contract violations.
- Benchmark memory/size values are approximate and computed with deep DataFrame
  sizing plus serialized model-size estimation when possible.
- Fairness helpers are available in-module and exported, but currently invoked explicitly by callers rather than auto-run in `evaluate_experiment`.

## Related Modules
- `src/models` — provides train/test interfaces
- `src/data` — provides typed records and splits
- `src/config` — provides runtime configuration

## Related Documentation
- `docs/evaluation_workflow.md`
- `docs/src/evaluation/evaluation.md`
- `docs/experiments/tracking.md`

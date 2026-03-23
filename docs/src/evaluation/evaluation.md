
# Module: src/evaluation

## Responsibility
Owns execution of model evaluation across datasets and configurations. This includes orchestrating training/testing loops, computing performance metrics, benchmarking models against accuracy and mobile deployment constraints (latency, memory, model size), and ensuring fairness in recommendations.

This module does NOT:

- Implement model logic (handled by src/models)
- Perform feature engineering (handled by src/features)
- Load raw datasets (handled by src/data)

## Public Interfaces
evaluate_experiment(datasets, models, onet_db, config)
- Runs full evaluation loop across datasets and models
- Calls:
    - Dataset.split(): src/evaluation
    - model.train(): src/models
    - evaluate_model(): src/evaluation
evaluate_model(model, dataset, onet_db, config)
- Evaluates a single model on a single dataset
- Calls:
    - model.test(): src/models
    - compute_all_metrics(): src/evaluation/metrics
    - benchmark_model_inference(): src/evaluation/benchmark
compute_all_metrics(predictions, ground_truth, k_values)
- Computes all configured evaluation metrics
- Calls:
    - ndcg_at_k(), precision_at_k(): src/evaluation/metrics
benchmark_model_inference(model, test_data, onet_db, num_runs)
- Measures latency and memory usage for model inference
- Calls:
    - model.test(): src/models
    - estimate_memory_usage(): src/evaluation/benchmark

## Internal Structure
src/evaluation/
│
├── evaluator.py        # Orchestration logic (main loop), Dataset class
├── metrics.py          # Accuracy + ranking metrics (cosine, NDCG@K, Precision@K)
├── benchmark.py        # Latency + memory measurement
├── reporting.py        # Result formatting and persistence
├── fairness.py         # Fairness evaluation and bias detection
├── __init__.py         # Public interface exports

## Data Contracts
Inputs:
- datasets: List[Dataset]
    - Dataset objects containing train/test TrainingRecord lists
    - Must support split() → (train_df, test_df)
- models: List[BaseModel]
    - Model instances implementing train(X, y) and test(X, onet_db)
    - Must return DataFrame with [Title, Career Category, Match_Score]
- onet_db: pd.DataFrame
    - O*NET careers database with RIASEC vectors and categories
- config: Dict
    - evaluation.k_values: List[int] — K values for ranking metrics
    - Other evaluation parameters
Outputs:
- Evaluation results per (dataset, model):
    {
    "model": str,
    "dataset": str,
    "metrics": {"ndcg@5": float, "precision@5": float, ...},
    "latency_ms": float,
    "memory_bytes": float,
    "model_size_mb": float,
    "constraint_violations": {...}
    }
- Stored in: experiments/results/<experiment_id>/evaluation.json

## Constraints
Performance:
- Evaluation must scale across multiple models and datasets
- Metric computation should be vectorized where possible
Memory:
- Must measure and report runtime memory usage
- Must support constraint-based evaluation (e.g., max_memory_mb, max_latency_ms, max_processing_power_mw)
Privacy:
- No PII should be processed or logged
- Evaluation data should be anonymized
Bias considerations:
- Monitor distribution of recommendations across RIASEC categories
- Ensure no systematic over-representation of specific career groups
- Detect fairness violations using compute_fairness_score()

## Related Modules
- src/models – provides model interfaces (train, predict)
- src/data – dataset loading and splitting
- src/features – feature generation pipeline
- src/config – experiment configuration loader

## Related Documentation
- docs/evaluation.md
- docs/data_contracts.md
- docs/experiments.md

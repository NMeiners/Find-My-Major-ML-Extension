"""
File: evaluator.py
Path: src/evaluation/evaluator.py

Purpose:
  Orchestrates the evaluation loop across datasets and models. Manages the
  evaluation workflow including training, testing, metric computation, and
  benchmarking. No metric logic is implemented here.

Original Author(s):
  - AI Assistant (GitHub Copilot)

AI Tools Used:
  - GitHub Copilot - Code generation and documentation

Editors:
  - AI Assistant (2026-03-23) — Initial implementation
  - AI Assistant (2026-03-30) — Added configurable metric selection enforcement
  - AI Assistant (2026-03-30) — Added evaluation contract checks and k-value validation

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-03-30

Assumptions & Constraints:
  - Models implement BaseModel interface (train, test, get_name)
  - Datasets are lists of TrainingRecord objects
  - O*NET database is pre-loaded DataFrame
  - Evaluation results stored in experiments/results/<id>/evaluation.json

Related Docs:
  - docs/src/evaluation/evaluation.md
  - docs/data_contracts.md
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd

from src.data.schemas import TrainingRecord
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.benchmark import benchmark_model_inference, get_model_size_mb

_SUPPORTED_METRICS = {"ndcg_at_k", "precision_at_k"}
_REQUIRED_PREDICTION_COLUMNS = ("Title", "Career Category", "Match_Score")


class EvaluationInterrupted(Exception):
    """Raised when evaluation is interrupted but partial model results exist."""

    def __init__(self, partial_result: Dict[str, Any]):
        super().__init__("Evaluation interrupted with partial model results.")
        self.partial_result = partial_result


# Normalize configured metric names into the supported evaluator metric set.
def _resolve_metric_selection(config: Dict[str, Any]) -> set[str]:
    evaluation_cfg = config.get("evaluation", {})
    configured = evaluation_cfg.get("metrics")

    if configured is None:
        return set(_SUPPORTED_METRICS)

    selected = set(configured)
    unsupported = selected - _SUPPORTED_METRICS
    if unsupported:
        unsupported_str = ", ".join(sorted(unsupported))
        supported_str = ", ".join(sorted(_SUPPORTED_METRICS))
        raise ValueError(
            f"Unsupported evaluation metric(s): {unsupported_str}. "
            f"Supported metrics: {supported_str}"
        )

    return selected


# Filter computed metric outputs based on configured metric families.
def _filter_metric_results(
    metric_results: Dict[str, Any],
    selected_metrics: set[str],
) -> Dict[str, Any]:
    filtered: Dict[str, Any] = {}

    for key, value in metric_results.items():
        if key.startswith("ndcg@") and "ndcg_at_k" in selected_metrics:
            filtered[key] = value
        elif key.startswith("precision@") and "precision_at_k" in selected_metrics:
            filtered[key] = value

    return filtered


def _validate_k_values(k_values: Any, top_k: int) -> list[int]:
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"evaluation.top_k must be a positive integer, got {top_k!r}")

    if not isinstance(k_values, list) or not k_values:
        raise ValueError("evaluation.k_values must be a non-empty list of positive integers")

    normalized: list[int] = []
    for value in k_values:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"evaluation.k_values must contain only positive integers, got {value!r}"
            )
        if value > top_k:
            raise ValueError(
                f"evaluation.k_values contains {value}, which exceeds evaluation.top_k={top_k}. "
                "Use k_values <= top_k so metrics reflect returned recommendations."
            )
        normalized.append(value)

    return normalized


def _resolve_top_k(evaluation_cfg: Dict[str, Any], model: Any) -> int:
    configured_top_k = evaluation_cfg.get("top_k")
    if configured_top_k is not None:
        if not isinstance(configured_top_k, int) or configured_top_k <= 0:
            raise ValueError(
                f"evaluation.top_k must be a positive integer, got {configured_top_k!r}"
            )
        return configured_top_k

    model_top_n_jobs = getattr(model, "top_n_jobs", None)
    if isinstance(model_top_n_jobs, int) and model_top_n_jobs > 0:
        return model_top_n_jobs

    return 5


def _collect_prediction_contract_violations(
    predictions: pd.DataFrame,
    top_n_jobs: int,
) -> Dict[str, int]:
    violations: Dict[str, int] = {}

    if len(predictions) > top_n_jobs:
        violations["row_count_exceeds_top_n_jobs"] = 1

    null_in_required = int(predictions[list(_REQUIRED_PREDICTION_COLUMNS)].isnull().any().any())
    if null_in_required:
        violations["null_in_required_columns"] = 1

    scores = pd.to_numeric(predictions["Match_Score"], errors="coerce")
    if scores.isnull().any():
        violations["non_numeric_match_score"] = 1
    elif ((scores < 0.0) | (scores > 1.0)).any():
        violations["match_score_out_of_range"] = 1

    return violations


def _merge_violation_counts(
    aggregate: Dict[str, int],
    sample_violations: Dict[str, int],
) -> None:
    for key, value in sample_violations.items():
        aggregate[key] = aggregate.get(key, 0) + value


def _build_model_result(
    all_metrics: list[Dict[str, Any]],
    latencies: list[float],
    memory_usages: list[float],
    model: Any,
    constraint_violations: Dict[str, int],
    samples_evaluated: int,
) -> Dict[str, Any]:
    if not all_metrics:
        raise ValueError("No test samples were evaluated. Check max_test_samples and test dataset size.")

    avg_metrics: Dict[str, float] = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    avg_latency = sum(latencies) / len(latencies)
    avg_memory = sum(memory_usages) / len(memory_usages)
    model_size = get_model_size_mb(model)

    return {
        'metrics': avg_metrics,
        'latency_ms': avg_latency,
        'memory_bytes': avg_memory,
        'model_size_mb': model_size,
        'constraint_violations': constraint_violations,
        'samples_evaluated': samples_evaluated,
    }


class Dataset:
    """
    Name: Dataset

    Purpose:
      Wraps training and test data for evaluation. Provides split() method
      and conversion to DataFrames for model training/testing.

    Inputs:
      - train_records: List[TrainingRecord] — training data
      - test_records: List[TrainingRecord] — test data
      - feature_columns: List[str] — RIASEC feature column names
      - label_column: str — career category label column name

    Outputs:
      - split() returns (train_df, test_df) DataFrames

    Raises / Errors:
      - ValueError: if records are empty or feature columns invalid

    Notes:
      - Converts TrainingRecord objects to DataFrames for model compatibility
    """

    def __init__(self, train_records: List[TrainingRecord], test_records: List[TrainingRecord],
                 feature_columns: List[str], label_column: str = 'career_category'):
        if not train_records or not test_records:
            raise ValueError("Train and test records cannot be empty")
        if not feature_columns:
            raise ValueError("Feature columns cannot be empty")

        self.train_records = train_records
        self.test_records = test_records
        self.feature_columns = feature_columns
        self.label_column = label_column

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Name: split

        Purpose:
          Returns train and test DataFrames for model training/testing.

        Inputs:
          - None

        Outputs:
          - Tuple[pd.DataFrame, pd.DataFrame] — (X_train, X_test) DataFrames

        Raises / Errors:
          - None

        Notes:
          - X_train/X_test contain feature columns only
          - Target labels are separate for training
        """
        # Convert records to DataFrames with configured feature column names
        def record_to_row(r: TrainingRecord) -> Dict[str, Any]:
            row: Dict[str, Any] = {}

            for feature_col in self.feature_columns:
                normalized_feature = feature_col.lower().replace(' ', '_')
                if not hasattr(r, normalized_feature):
                    raise KeyError(f"Training record missing feature '{normalized_feature}'")
                row[feature_col] = getattr(r, normalized_feature)

            row[self.label_column] = r.career_category
            return row

        train_df = pd.DataFrame([record_to_row(r) for r in self.train_records])
        test_df = pd.DataFrame([record_to_row(r) for r in self.test_records])

        return train_df, test_df


def evaluate_experiment(datasets: List[Dataset], models: List[Any], onet_db: pd.DataFrame,
                       config: Dict[str, Any], output_path: Path | None = None) -> List[Dict[str, Any]]:
    """
    Name: evaluate_experiment

    Purpose:
      Runs the full evaluation loop across all datasets and models.

    Inputs:
      - datasets: List[Dataset] — evaluation datasets
      - models: List[Any] — model instances implementing BaseModel interface
      - onet_db: pd.DataFrame — O*NET career database
      - config: Dict[str, Any] — experiment configuration

    Outputs:
      - List[Dict[str, Any]] — evaluation results for each (dataset, model) pair

    Raises / Errors:
      - AttributeError: if models don't implement required interface

    Notes:
      - Orchestrates the evaluation loop as specified in evaluation.md
      - Calls evaluate_model for each combination
    """
    results = []

    for dataset_idx, dataset in enumerate(datasets):
        X_train, X_test = dataset.split()

        for model in models:
            y_col = getattr(model, 'y_feature', dataset.label_column)
            y_train = X_train[y_col]
            # Reset model if method exists
            if hasattr(model, 'reset'):
                model.reset()

            # Train model
            model.train(X_train[dataset.feature_columns], y_train)

            # Evaluate model on this dataset
            try:
                model_results = evaluate_model(model, dataset, onet_db, config)
            except EvaluationInterrupted as exc:
                model_results = dict(exc.partial_result)
                model_results['dataset'] = f'dataset_{dataset_idx}'
                model_results['model'] = model.get_name()
                model_results['interrupted'] = True
                results.append(model_results)

                if output_path is not None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                raise KeyboardInterrupt
            except KeyboardInterrupt:
                # Persist any completed model results before exiting.
                if output_path is not None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                raise
            model_results['dataset'] = f'dataset_{dataset_idx}'
            model_results['model'] = model.get_name()

            results.append(model_results)

            # Save incremental results in case of long evaluation
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)

    return results


def evaluate_model(model: Any, dataset: Dataset, onet_db: pd.DataFrame,
                  config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Name: evaluate_model

    Purpose:
      Evaluates a single model on a single dataset.

    Inputs:
      - model: Any — model instance with train/test methods
      - dataset: Dataset — evaluation dataset
      - onet_db: pd.DataFrame — O*NET career database
      - config: Dict[str, Any] — experiment configuration

    Outputs:
      - Dict[str, Any] — evaluation results with metrics, latency, memory, etc.

    Raises / Errors:
      - AttributeError: if model doesn't implement test method
      - ValueError: if unsupported metrics are configured
      - ValueError: if zero test samples are evaluated

    Notes:
      - Computes metrics for each test sample and averages
      - Measures performance benchmarks
    """
    _, X_test = dataset.split()
    evaluation_cfg = config.get('evaluation', {})
    selected_metrics = _resolve_metric_selection(config)
    configured_top_k = _resolve_top_k(evaluation_cfg, model)
    k_values = _validate_k_values(evaluation_cfg.get('k_values', [configured_top_k]), configured_top_k)

    all_metrics = []
    latencies = []
    memory_usages = []
    constraint_violations: Dict[str, int] = {}

    benchmark_runs = evaluation_cfg.get('benchmark_runs', 1)
    if not isinstance(benchmark_runs, int) or benchmark_runs <= 0:
        raise ValueError(
            f"evaluation.benchmark_runs must be a positive integer, got {benchmark_runs!r}"
        )
    max_test_samples = evaluation_cfg.get('max_test_samples')
    if max_test_samples is not None:
        if not isinstance(max_test_samples, int) or max_test_samples <= 0:
            raise ValueError(
                "evaluation.max_test_samples must be null or a positive integer"
            )
    progress_log_interval = evaluation_cfg.get('progress_log_interval', 1000)
    if progress_log_interval is not None:
        if not isinstance(progress_log_interval, int) or progress_log_interval <= 0:
            raise ValueError(
                "evaluation.progress_log_interval must be null or a positive integer"
            )

    # Evaluate each test sample
    label_col = getattr(model, 'y_feature', dataset.label_column)

    samples_evaluated = 0
    for sample_idx, (_, test_row) in enumerate(X_test.iterrows()):
        if max_test_samples is not None and sample_idx >= max_test_samples:
            break
        # Create single-row DataFrame for this test sample
        X_sample = test_row[dataset.feature_columns].to_frame().T
        ground_truth = test_row[label_col]

        # Get predictions and benchmark metrics from model inference
        try:
            benchmark_results = benchmark_model_inference(model, X_sample, onet_db, num_runs=benchmark_runs)
        except KeyboardInterrupt as exc:
            if samples_evaluated > 0:
                partial = _build_model_result(
                    all_metrics=all_metrics,
                    latencies=latencies,
                    memory_usages=memory_usages,
                    model=model,
                    constraint_violations=constraint_violations,
                    samples_evaluated=samples_evaluated,
                )
                raise EvaluationInterrupted(partial) from exc
            raise
        predictions = benchmark_results['predictions']

        if not isinstance(predictions, pd.DataFrame):
            raise TypeError(
                f"{model.get_name()}.test() must return a pandas DataFrame, got {type(predictions)!r}"
            )

        missing_columns = [c for c in _REQUIRED_PREDICTION_COLUMNS if c not in predictions.columns]
        if missing_columns:
            raise ValueError(
                f"{model.get_name()} predictions missing required columns: {missing_columns}"
            )

        sample_violations = _collect_prediction_contract_violations(
            predictions=predictions,
            top_n_jobs=configured_top_k,
        )
        _merge_violation_counts(constraint_violations, sample_violations)

        # Compute metrics
        sample_metrics = compute_all_metrics(predictions, ground_truth, k_values)
        sample_metrics = _filter_metric_results(sample_metrics, selected_metrics)
        all_metrics.append(sample_metrics)

        latencies.append(benchmark_results['latency_ms'])
        memory_usages.append(benchmark_results['memory_bytes'])
        samples_evaluated += 1
        if progress_log_interval and samples_evaluated % progress_log_interval == 0:
            print(
                f"[{model.get_name()}] evaluated {samples_evaluated} sample(s)...",
                flush=True,
            )

    return _build_model_result(
        all_metrics=all_metrics,
        latencies=latencies,
        memory_usages=memory_usages,
        model=model,
        constraint_violations=constraint_violations,
        samples_evaluated=samples_evaluated,
    )


def save_evaluation_results(results: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Name: save_evaluation_results

    Purpose:
      Saves evaluation results to JSON file.

    Inputs:
      - results: List[Dict[str, Any]] — evaluation results
      - output_path: Path — output file path

    Outputs:
      - None (writes to file)

    Raises / Errors:
      - IOError: if file cannot be written

    Notes:
      - Creates output directory if it doesn't exist
      - Stores results in experiments/results/<id>/evaluation.json format
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

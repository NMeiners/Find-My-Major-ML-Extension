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

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-03-23

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
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd

from src.data.schemas import TrainingRecord
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.benchmark import benchmark_model_inference, get_model_size_mb


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
            model_results = evaluate_model(model, dataset, onet_db, config)
            model_results['dataset'] = f'dataset_{dataset_idx}'
            model_results['model'] = model.get_name()

            results.append(model_results)

            # Save incremental results in case of long evaluation
            if output_path is not None:
                from src.evaluation.reporting import save_results_to_file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    import json
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

    Notes:
      - Computes metrics for each test sample and averages
      - Measures performance benchmarks
    """
    _, X_test = dataset.split()
    k_values = config.get('evaluation', {}).get('k_values', [5, 10])

    all_metrics = []
    latencies = []
    memory_usages = []

    benchmark_runs = config.get('evaluation', {}).get('benchmark_runs', 1)
    max_test_samples = config.get('evaluation', {}).get('max_test_samples')

    # Evaluate each test sample
    label_col = getattr(model, 'y_feature', dataset.label_column)

    for idx, test_row in X_test.iterrows():
        if max_test_samples is not None and idx >= max_test_samples:
            break
        # Create single-row DataFrame for this test sample
        X_sample = test_row[dataset.feature_columns].to_frame().T
        ground_truth = test_row[label_col]

        # Get predictions and benchmark metrics from model inference
        benchmark_results = benchmark_model_inference(model, X_sample, onet_db, num_runs=benchmark_runs)
        predictions = benchmark_results['predictions']

        # Compute metrics
        sample_metrics = compute_all_metrics(predictions, ground_truth, k_values)
        all_metrics.append(sample_metrics)

        latencies.append(benchmark_results['latency_ms'])
        memory_usages.append(benchmark_results['memory_bytes'])

    # Average metrics across all test samples
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    # Average performance metrics
    avg_latency = sum(latencies) / len(latencies)
    avg_memory = sum(memory_usages) / len(memory_usages)
    model_size = get_model_size_mb(model)

    return {
        'metrics': avg_metrics,
        'latency_ms': avg_latency,
        'memory_bytes': avg_memory,
        'model_size_mb': model_size,
        'constraint_violations': {}  # Placeholder for future constraint checking
    }


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
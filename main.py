"""
File: main.py
Path: main.py

Purpose:
  Entry point for the machine learning training pipeline. Handles command-line argument parsing, configuration loading, and initiates the training process.

Original Author(s):
  - Nathan Meiners
  - AI Assistant

AI Tools Used:
  - GitHub Copilot - Code generation and documentation
  - Claude Sonnet 4.6 - Training loop implementation

Editors:
  - AI Assistant (2026-03-14) — Initial implementation
  - AI Assistant (2026-03-20) — Added config-driven model training loop
  - AI Assistant (2026-03-30) — Added deterministic seed initialization and output flag warnings

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-03-30

Assumptions & Constraints:
  - Executed from repository root
  - Config file exists and is valid
  - Output directory is writable
  - datasets.onet_db_path in config points to a valid CSV file
  - Training data CSV exists at datasets[0].train_path

Related Docs:
  - docs/src/config/config_loader.md
  - docs/src/models.md
"""

import argparse
import os
import sys
import random
from pathlib import Path

import pandas as pd
import numpy as np
from src.config.config_loader import load_config
from src.data.loader import load_training_records, split_training_records
from src.evaluation import Dataset, evaluate_experiment
from src.evaluation.reporting import save_results_to_file, save_results_to_csv
from src.models import MODEL_REGISTRY


def main():
    """
    Name: main

    Purpose:
      Parses command-line arguments, loads the experiment configuration, trains all
      models defined in the config, and returns the trained model array for evaluation.
      All model types, features, hyperparameters, and result counts are read from config.

    Inputs:
      - Command-line arguments: config_path (positional)

    Outputs:
      - list[BaseModel] — trained model instances (returned for downstream use)

    Raises / Errors:
      - SystemExit: if config file does not exist or loading fails
      - KeyError: if config is missing required keys (model name, onet_db_path, etc.)

    Notes:
      - Models are instantiated via MODEL_REGISTRY using the name string from config.
      - top_n_categories is read from each model's parameters block in config.
      - onet_db is loaded once and passed to all models at test time.
    """
    parser = argparse.ArgumentParser(description="Run ML training pipeline with config file.")
    parser.add_argument('config_path', help='Path to the experiment configuration YAML file')
    parser.add_argument('--export-csv', action='store_true', default=False,
                        help='Export evaluation results to CSV after run')

    args = parser.parse_args()

    config_path = args.config_path

    if not os.path.isfile(config_path):
        print(f"Error: Configuration file '{config_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    seed_value = config.get('experiment', {}).get('random_seed')
    if seed_value is not None:
        try:
            seed = int(seed_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"experiment.random_seed must be an integer if provided, got {seed_value!r}"
            ) from exc
        random.seed(seed)
        np.random.seed(seed)

    # Create output directory (per docs: experiments/results/<experiment_id>/<run_id>/)
    experiment_id = config['experiment']['id']
    base_dir = config['output']['directory']
    run_id = config['run']['run_id']

    # Ensure user output path exists
    output_dir = Path(base_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Configuration loaded successfully. Run ID: {run_id}")
    print(f"Output directory created: {output_dir}")

    # Determine O*NET database path and load it for job ranking
    onet_db_path = config.get('onet_db_path')
    if not onet_db_path:
        raise KeyError("Configuration missing 'onet_db_path'.")

    if not os.path.isfile(onet_db_path):
        raise FileNotFoundError(f"O*NET DB file not found: {onet_db_path}")

    onet_db = pd.read_csv(onet_db_path)

    # Determine the union of all model feature columns for dataset construction
    model_feature_columns = sorted({
        feature
        for model_cfg in config.get('models', [])
        for feature in model_cfg.get('x_features', [])
    })

    # Label column for all datasets is expected to be aligned with models' y_features
    label_column = config['models'][0]['y_features'][0] if config.get('models') else 'Career Category'

    # Build Dataset objects from config
    datasets = []

    for dataset_cfg in config.get('datasets', []):
        train_path = dataset_cfg.get('train_path')
        test_path = dataset_cfg.get('test_path')

        if not train_path or not os.path.isfile(train_path):
            raise FileNotFoundError(f"Training data file not found: {train_path}")

        train_records = load_training_records(train_path)

        if test_path:
            if not os.path.isfile(test_path):
                raise FileNotFoundError(f"Test data file not found: {test_path}")
            test_records = load_training_records(test_path)
        else:
            split_cfg = dataset_cfg.get('split', {})
            train_fraction = float(split_cfg.get('train', 0.7))
            val_fraction = float(split_cfg.get('validation', 0.15))
            test_fraction = float(split_cfg.get('test', 0.15))

            train_records, _, test_records = split_training_records(
                train_records,
                val_size=val_fraction,
                test_size=test_fraction,
            )

        if dataset_cfg.get('shuffle', False):
            random.shuffle(train_records)
            random.shuffle(test_records)

        datasets.append(Dataset(
            train_records=train_records,
            test_records=test_records,
            feature_columns=model_feature_columns,
            label_column=label_column,
        ))

    # Build model instances
    models = []
    for model_cfg in config['models']:
        model_name = model_cfg['model']
        if model_name not in MODEL_REGISTRY:
            print(f"Warning: Unknown model '{model_name}' in config — skipping.", file=sys.stderr)
            continue

        parameters = dict(model_cfg.get('parameters', {}))
        top_n_categories = parameters.pop('top_n_categories', 3)

        x_features = model_cfg['x_features']
        y_feature = model_cfg['y_features'][0]
        top_n_jobs = config['evaluation'].get('top_k', 5)

        model = MODEL_REGISTRY[model_name](
            x_features=x_features,
            y_feature=y_feature,
            parameters=parameters,
            top_n_jobs=top_n_jobs,
            top_n_categories=top_n_categories,
        )
        models.append(model)

    print(f"\n{len(datasets)} dataset(s) loaded, {len(models)} model(s) instantiated.")

    # Run evaluation across all datasets and models
    evaluation_results_path = output_dir / 'evaluation.json'
    try:
        evaluation_results = evaluate_experiment(
            datasets,
            models,
            onet_db,
            config,
            output_path=evaluation_results_path,
        )
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.", file=sys.stderr)
        if evaluation_results_path.exists():
            print(f"Partial results preserved at {evaluation_results_path}", file=sys.stderr)
        else:
            print("No partial evaluation results were written yet.", file=sys.stderr)
        sys.exit(130)

    if config['output'].get('save_metrics', False):
        output_path = save_results_to_file(
            evaluation_results,
            output_dir=output_dir,
        )
        print(f"Evaluation results saved to {output_path}")

    if config['output'].get('save_metrics_csv', False) or args.export_csv:
        csv_path = save_results_to_csv(
            evaluation_results,
            output_dir=output_dir,
        )
        print(f"Evaluation CSV exported to {csv_path}")

    if config['output'].get('save_models', False):
        print(
            "Warning: output.save_models=true but model artifact persistence "
            "is not implemented in this pipeline."
        )

    if config['output'].get('save_predictions', False):
        print(
            "Warning: output.save_predictions=true but per-sample prediction "
            "persistence is not implemented in this pipeline."
        )

    print(f"\nExperiment complete. Total results: {len(evaluation_results)}")

    return evaluation_results


if __name__ == "__main__":
    main()

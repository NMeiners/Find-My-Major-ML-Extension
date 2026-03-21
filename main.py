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

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-03-20

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
import pandas as pd
from src.config.config_loader import load_config
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

    # Create output directory
    base_dir = config['output']['directory']
    run_id = config['run']['run_id']
    output_dir = os.path.join(base_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Configuration loaded successfully. Run ID: {run_id}")
    print(f"Output directory created: {output_dir}")

    # Load training data from the first dataset entry in config
    dataset_cfg = config['datasets'][0]
    train_df = pd.read_csv(dataset_cfg['train_path'])

    # Load test data and apply column mapping if specified
    test_df = None
    if dataset_cfg.get('test_path'):
        test_df = pd.read_csv(dataset_cfg['test_path'])
        col_mapping = dataset_cfg.get('test_column_mapping')
        if col_mapping:
            test_df = test_df.rename(columns=col_mapping)

    # Load O*NET career database for job ranking (separate from training data)
    onet_db = pd.read_csv(config['onet_db_path'])

    top_n_jobs = config['evaluation']['top_k']

    # Train all models defined in the config
    trained_models = []
    for model_cfg in config['models']:
        model_name = model_cfg['model']
        if model_name not in MODEL_REGISTRY:
            print(f"Warning: Unknown model '{model_name}' in config — skipping.", file=sys.stderr)
            continue

        # Extract top_n_categories from parameters; remove it before passing to sklearn
        parameters = dict(model_cfg.get('parameters', {}))
        top_n_categories = parameters.pop('top_n_categories', 3)

        ModelClass = MODEL_REGISTRY[model_name]
        model = ModelClass(
            x_features=model_cfg['x_features'],
            y_feature=model_cfg['y_features'][0],
            parameters=parameters,
            top_n_jobs=top_n_jobs,
            top_n_categories=top_n_categories,
        )

        X_train = train_df[model_cfg['x_features']]
        y_train = train_df[model_cfg['y_features'][0]]

        print(f"Training {model_name}...")
        model.train(X_train, y_train)
        trained_models.append(model)
        print(f"  {model_name} trained.")

    print(f"\nAll models trained. {len(trained_models)} model(s) ready for evaluation.")
    # TODO: Pass trained_models and onet_db to evaluation module
    return trained_models


if __name__ == "__main__":
    main()
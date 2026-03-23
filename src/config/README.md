# Config Module

This module handles loading and validation of experiment configuration files for the ML training pipeline.

## Overview

The config system allows defining experiments with multiple datasets and models. The training pipeline iterates over all combinations:

```
for dataset in config["datasets"]:
    for model in config["models"]:
        run_training(dataset, model)
        run_testing(dataset, model)
```

This creates a matrix of experiments, running each model on each dataset.

## Configuration File Structure

Config files are YAML files stored in `experiments/config/`. See `docs/templates/exp_config.yaml` for a complete example.

### Required Sections

- `experiment.id`: Unique experiment identifier (used for run ID generation)
- `datasets`: List of dataset configurations
- `models`: List of model configurations
- `output.directory`: Base directory for experiment outputs

### Datasets Section

Each dataset entry defines:
- `name`: Dataset identifier
- `train_path`: Path to training data
- `test_path`: Path to test data (optional)
- `split`: Train/validation/test ratios (if no test_path)
- `shuffle`: Whether to shuffle data

### Models Section

Each model entry defines:
- `model`: Model type (e.g., "logistic_regression")
- `parameters`: Model hyperparameters
- `x_features`: List of input feature columns
- `y_features`: List of target columns

## Usage

1. Create a YAML config file in `experiments/config/`
2. Run: `python main.py experiments/config/your_config.yaml`
3. The system generates a unique run ID and output directory
4. Training runs all dataset-model combinations

## Runtime Metadata

The loader automatically adds:
- `run_id`: Unique identifier (format: `<experiment_id>_<YYYYMMDD_HHMMSS>`)
- `start_time`: ISO timestamp of execution

This metadata is available in `config["run"]` but not saved to the config file.
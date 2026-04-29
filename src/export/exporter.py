"""
File: exporter.py
Path: src/export/exporter.py

Purpose:
  Implements export support for frontend inference artifacts. Trains the configured
  experiment model on the configured training dataset, converts the trained
  sklearn estimator to ONNX, and exports a lightweight JSON career database for
  browser consumption.

Original Author(s):
  - AI Assistant

AI Tools Used:
  - GitHub Copilot - Module implementation

Editors:
  - AI Assistant (2026-04-29) — Added export module and config-driven ONNX export

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-04-29

Assumptions & Constraints:
  - export_inference_model is enabled in config.export
  - Supported export format is ONNX only
  - Training dataset CSV and onet_db_path CSV exist and include expected columns
  - Model registry classes support x_features/y_features the same way as evaluation

Related Docs:
  - docs/src/export/export.md
  - docs/src/config/config_loader.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.models import MODEL_REGISTRY

EXPORT_MODEL_FILENAME = "riasec_model.onnx"
EXPORT_DB_FILENAME = "riasec_jobs_db.json"
DEFAULT_EXPORT_FORMAT = "onnx"
REQUIRED_FRONTEND_COLUMNS = ["O*NET-SOC Code", "Title"]


class ExporterError(Exception):
    """Raised when export processing fails."""


def export_frontend_artifacts(config: dict[str, Any], output_dir: Path | str) -> dict[str, Path]:
    """
    Name: export_frontend_artifacts

    Purpose:
      Exports frontend artifacts for an experiment run when export is enabled.

    Inputs:
      - config: dict[str, Any] — loaded experiment configuration
      - output_dir: Path | str — directory where frontend artifacts will be written

    Outputs:
      - dict[str, Path] — paths to the generated ONNX model and frontend JSON database

    Raises / Errors:
      - KeyError: if required configuration fields are missing
      - FileNotFoundError: if configured data files cannot be found
      - RuntimeError: if the export model cannot be trained or serialized
      - ImportError: if skl2onnx is not installed for ONNX export
    """
    export_cfg = config.get('export', {})
    enabled = export_cfg.get('export_inference_model', False)
    if not enabled:
        return {}

    export_format = str(export_cfg.get('format', DEFAULT_EXPORT_FORMAT)).strip().lower()
    if export_format != DEFAULT_EXPORT_FORMAT:
        raise ValueError(
            f"Unsupported export format '{export_format}'. Only '{DEFAULT_EXPORT_FORMAT}' is supported."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_config = _get_first_enabled_dataset(config)
    model_config = _get_first_enabled_model(config)
    train_path = dataset_config.get('train_path')
    if not train_path:
        raise KeyError("Enabled dataset configuration is missing 'train_path'.")
    if not Path(train_path).is_file():
        raise FileNotFoundError(f"Training dataset file not found: {train_path}")

    onet_db_path = config.get('onet_db_path')
    if not onet_db_path:
        raise KeyError("Configuration missing 'onet_db_path'.")
    if not Path(onet_db_path).is_file():
        raise FileNotFoundError(f"O*NET DB file not found: {onet_db_path}")

    training_df = pd.read_csv(train_path)
    frontend_df = pd.read_csv(onet_db_path)

    model = _build_model_from_config(model_config, config)
    y_feature = model_config['y_features'][0]
    if y_feature not in training_df.columns:
        raise KeyError(
            f"Training dataset does not contain configured label column '{y_feature}'."
        )

    missing_features = [f for f in model.x_features if f not in training_df.columns]
    if missing_features:
        raise KeyError(
            f"Training dataset missing required feature columns: {missing_features}"
        )

    model.train(training_df[model.x_features], training_df[y_feature])
    sklearn_estimator = _get_wrapped_sklearn_estimator(model)

    onnx_path = output_path / EXPORT_MODEL_FILENAME
    _save_model_to_onnx(sklearn_estimator, model.x_features, onnx_path)

    json_path = output_path / EXPORT_DB_FILENAME
    _save_frontend_json(frontend_df, model.x_features, y_feature, json_path)

    return {
        'onnx_model': onnx_path,
        'frontend_db': json_path,
    }


def _get_first_enabled_dataset(config: dict[str, Any]) -> dict[str, Any]:
    for dataset_cfg in config.get('datasets', []):
        if dataset_cfg.get('enabled', True):
            return dataset_cfg
    raise ValueError("Experiment configuration must include at least one enabled dataset.")


def _get_first_enabled_model(config: dict[str, Any]) -> dict[str, Any]:
    for model_cfg in config.get('models', []):
        if model_cfg.get('enabled', True):
            return model_cfg
    raise ValueError("Experiment configuration must include at least one enabled model.")


def _build_model_from_config(model_config: dict[str, Any], config: dict[str, Any]) -> Any:
    model_name = model_config.get('model')
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}' in export config.")

    x_features = model_config.get('x_features')
    if not isinstance(x_features, list) or not x_features:
        raise KeyError(f"Model '{model_name}' must define a non-empty x_features list.")

    y_features = model_config.get('y_features')
    if not isinstance(y_features, list) or not y_features:
        raise KeyError(f"Model '{model_name}' must define a non-empty y_features list.")

    parameters = dict(model_config.get('parameters', {}))
    parameters.pop('n_jobs', None)
    top_n_categories = parameters.pop('top_n_categories', 3)
    top_n_jobs = config.get('evaluation', {}).get('top_k', 5)

    return MODEL_REGISTRY[model_name](
        x_features=x_features,
        y_feature=y_features[0],
        parameters=parameters,
        top_n_jobs=top_n_jobs,
        top_n_categories=top_n_categories,
    )


def _get_wrapped_sklearn_estimator(model: Any) -> Any:
    if hasattr(model, '_model') and model._model is not None:
        return model._model
    if hasattr(model, 'model') and getattr(model, 'model') is not None:
        return getattr(model, 'model')
    raise RuntimeError(
        "Cannot export model to ONNX because no underlying sklearn estimator was found."
    )


def _save_model_to_onnx(sklearn_model: Any, feature_columns: list[str], output_path: Path) -> None:
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as exc:
        raise ImportError(
            "Exporting to ONNX requires the skl2onnx package. "
            "Install skl2onnx before running export."
        ) from exc

    initial_type = [('float_input', FloatTensorType([None, len(feature_columns)]))]
    onx = convert_sklearn(sklearn_model, initial_types=initial_type)
    with open(output_path, 'wb') as model_file:
        model_file.write(onx.SerializeToString())


def _save_frontend_json(
    frontend_df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    output_path: Path,
) -> None:
    required_columns = [*REQUIRED_FRONTEND_COLUMNS, label_column, *feature_columns]
    missing_frontend_cols = [c for c in required_columns if c not in frontend_df.columns]
    if missing_frontend_cols:
        raise KeyError(
            f"Frontend database is missing required columns: {missing_frontend_cols}"
        )

    output_df = frontend_df[[*REQUIRED_FRONTEND_COLUMNS, label_column, *feature_columns]]
    output_df.to_json(output_path, orient='records')

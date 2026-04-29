"""
File: test_export.py
Path: tests/export/test_export.py

Purpose:
  Unit tests for src/export module. Validates model and data export functionality.

Original Author(s):
  - AI Assistant

AI Tools Used:
  - GitHub Copilot - Test generation

Editors:
  - AI Assistant (2026-04-20) — Initial test coverage implementation

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-04-20

Assumptions & Constraints:
  - Tests do not depend on notebooks
  - Mock file I/O for deterministic test results
  - Placeholder tests for future export functionality

Related Docs:
  - docs/ci/ci_design.md
"""

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from src.export import export_frontend_artifacts


def test_export_module_imports():
    """
    Name: test_export_module_imports

    Purpose:
      Validates that the export module can be imported without errors.

    Inputs:
      - N/A

    Outputs:
      - Pass: module imports successfully
      - Fail: import raises exception

    Raises / Errors:
      - ImportError: if module structure is invalid

    Notes:
      - Serves as smoke test for module initialization
    """
    try:
        import src.export
        assert src.export is not None
    except ImportError as e:
        pytest.fail(f"Failed to import src.export: {e}")


def test_export_placeholder():
    """
    Name: test_export_placeholder

    Purpose:
      Placeholder for future export functionality tests.

    Inputs:
      - N/A

    Outputs:
      - Pass: placeholder assertion

    Raises / Errors:
      - N/A

    Notes:
      - To be expanded with export validation tests
    """
    assert True  # Placeholder


def test_export_frontend_artifacts_writes_onnx_and_json(tmp_path, monkeypatch):
    train_data = pd.DataFrame([
        {
            "Realistic": 1.0,
            "Investigative": 2.0,
            "Artistic": 3.0,
            "Social": 4.0,
            "Enterprising": 5.0,
            "Conventional": 6.0,
            "Career Category": "Science",
        },
        {
            "Realistic": 2.0,
            "Investigative": 1.0,
            "Artistic": 4.0,
            "Social": 3.0,
            "Enterprising": 6.0,
            "Conventional": 5.0,
            "Career Category": "Arts",
        }
    ])
    frontend_data = pd.DataFrame([
        {
            "O*NET-SOC Code": "15-1121.00",
            "Title": "Software Developer",
            "Career Category": "Science",
            "Realistic": 1.0,
            "Investigative": 2.0,
            "Artistic": 3.0,
            "Social": 4.0,
            "Enterprising": 5.0,
            "Conventional": 6.0,
        }
    ])

    train_csv = tmp_path / "train.csv"
    frontend_csv = tmp_path / "frontend.csv"
    train_data.to_csv(train_csv, index=False)
    frontend_data.to_csv(frontend_csv, index=False)

    config = {
        "onet_db_path": str(frontend_csv),
        "datasets": [
            {
                "name": "test_dataset",
                "enabled": True,
                "train_path": str(train_csv),
            }
        ],
        "models": [
            {
                "model": "gradient_boosting",
                "enabled": True,
                "parameters": {
                    "n_estimators": 1,
                    "learning_rate": 0.1,
                    "max_depth": 1,
                    "top_n_categories": 1,
                },
                "x_features": [
                    "Realistic",
                    "Investigative",
                    "Artistic",
                    "Social",
                    "Enterprising",
                    "Conventional",
                ],
                "y_features": ["Career Category"],
            }
        ],
        "evaluation": {
            "top_k": 1,
        },
        "output": {
            "directory": str(tmp_path / "output"),
        },
        "export": {
            "export_inference_model": True,
            "format": "onnx",
            "verify_package": False,
        },
    }

    class MockOnnx:
        def SerializeToString(self):
            return b"mock-onnx"

    class MockConvertModule:
        @staticmethod
        def convert_sklearn(model, initial_types):
            return MockOnnx()

    class MockFloatTensorType:
        def __init__(self, shape):
            self.shape = shape

    mock_skl2onnx = type(sys)("skl2onnx")
    mock_skl2onnx.convert_sklearn = MockConvertModule.convert_sklearn
    mock_common = type(sys)("skl2onnx.common")
    mock_datatypes = type(sys)("skl2onnx.common.data_types")
    mock_datatypes.FloatTensorType = MockFloatTensorType
    mock_common.data_types = mock_datatypes
    mock_skl2onnx.common = mock_common
    monkeypatch.setitem(sys.modules, "skl2onnx", mock_skl2onnx)
    monkeypatch.setitem(sys.modules, "skl2onnx.common", mock_common)
    monkeypatch.setitem(sys.modules, "skl2onnx.common.data_types", mock_datatypes)

    output_dir = tmp_path / "exported"
    export_paths = export_frontend_artifacts(config, output_dir)

    assert export_paths["onnx_model"].exists()
    assert export_paths["frontend_db"].exists()
    assert export_paths["export_package"].exists()

    assert export_paths["onnx_model"].read_bytes() == b"mock-onnx"

    loaded = json.loads(export_paths["frontend_db"].read_text())
    assert isinstance(loaded, list)
    assert loaded[0]["Title"] == "Software Developer"

"""
File: test_config_loader.py
Path: tests/test_config_loader.py

Purpose:
  Unit tests for the config_loader module, verifying YAML loading, run ID generation, metadata injection, and error handling.

Original Author(s):
  - Nathan Meiners
  - AI Assistant

AI Tools Used:
  - GitHub Copilot - Test generation

Editors:
  - AI Assistant (2026-03-14) — Initial implementation

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-03-14

Assumptions & Constraints:
  - Tests are deterministic using mocks
  - No external dependencies beyond stdlib and pytest

Related Docs:
  - docs/src/config/config_loader.md
"""

import pytest
import yaml
import tempfile
import os
from unittest.mock import patch
from src.config.config_loader import load_config


class TestLoadConfig:
    """Test suite for load_config function."""

    def test_load_valid_config(self):
        """Test loading a valid YAML config injects run metadata."""
        config_yaml = """
experiment:
  id: test_exp
output:
  directory: test/results
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_yaml)
            config_path = f.name

        try:
            with patch('src.config.config_loader.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20260314_154212"
                mock_datetime.now.return_value.isoformat.return_value = "2026-03-14T15:42:12"

                config = load_config(config_path)

                assert 'run' in config
                assert config['run']['run_id'] == "test_exp_20260314_154212"
                assert config['run']['start_time'] == "2026-03-14T15:42:12"
                assert config['experiment']['id'] == "test_exp"
        finally:
            os.unlink(config_path)

    def test_invalid_file_path(self):
        """Test that invalid file path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_missing_experiment_id(self):
        """Test that missing experiment.id raises KeyError."""
        config_yaml = """
output:
  directory: test/results
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_yaml)
            config_path = f.name

        try:
            with pytest.raises(KeyError):
                load_config(config_path)
        finally:
            os.unlink(config_path)

    def test_run_id_format(self):
        """Test that run_id follows the correct format."""
        config_yaml = """
experiment:
  id: exp_001
output:
  directory: test/results
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_yaml)
            config_path = f.name

        try:
            with patch('src.config.config_loader.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20260314_154212"

                config = load_config(config_path)

                run_id = config['run']['run_id']
                assert run_id == "exp_001_20260314_154212"
                # Check format: experiment_id + _ + YYYYMMDD_HHMMSS
                assert run_id.startswith("exp_001_")
                timestamp = "20260314_154212"
                assert run_id.endswith(f"_{timestamp}")
                assert len(timestamp) == 15  # YYYYMMDD_HHMMSS
        finally:
            os.unlink(config_path)
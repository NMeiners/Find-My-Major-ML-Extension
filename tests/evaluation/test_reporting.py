"""
File: test_reporting.py
Path: tests/evaluation/test_reporting.py

Purpose:
  Unit tests for reporting and output functions.

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
  - Tests file I/O operations safely
  - Uses temporary directories for file tests
  - Tests formatting without real console output

Related Docs:
  - docs/src/evaluation/evaluation.md
"""

import unittest
import tempfile
import os
from pathlib import Path
from src.evaluation.reporting import format_evaluation_results, save_results_to_file, load_results_from_file


class TestFormatEvaluationResults(unittest.TestCase):
    """Test result formatting functions."""

    def setUp(self):
        """Set up test evaluation results."""
        self.results = [
            {
                'model': 'random_forest',
                'dataset': 'dataset_0',
                'metrics': {'ndcg@5': 0.85, 'precision@5': 0.60},
                'latency_ms': 15.5,
                'memory_bytes': 2048,
                'model_size_mb': 0.5,
                'constraint_violations': {}
            },
            {
                'model': 'knn',
                'dataset': 'dataset_0',
                'metrics': {'ndcg@5': 0.78, 'precision@5': 0.55},
                'latency_ms': 8.2,
                'memory_bytes': 1024,
                'model_size_mb': 0.1,
                'constraint_violations': {'memory': 'exceeded'}
            }
        ]

    def test_format_evaluation_results_basic(self):
        """Test basic result formatting."""
        formatted = format_evaluation_results(self.results)

        # Check that output contains expected elements
        self.assertIn("Evaluation Results Summary", formatted)
        self.assertIn("Model: random_forest", formatted)
        self.assertIn("Model: knn", formatted)
        self.assertIn("ndcg@5: 0.8500", formatted)
        self.assertIn("Latency: 15.50 ms", formatted)
        self.assertIn("Memory: 2.00 KB", formatted)
        self.assertIn("Model Size: 0.50 MB", formatted)

    def test_format_evaluation_results_empty(self):
        """Test formatting with empty results."""
        formatted = format_evaluation_results([])
        self.assertIn("No evaluation results to display", formatted)

    def test_format_evaluation_results_with_violations(self):
        """Test formatting results with constraint violations."""
        formatted = format_evaluation_results(self.results)
        self.assertIn("Constraint Violations", formatted)
        self.assertIn("memory: exceeded", formatted)


class TestSaveLoadResults(unittest.TestCase):
    """Test saving and loading evaluation results."""

    def setUp(self):
        """Set up test data and temporary directory."""
        self.test_results = [
            {
                'model': 'test_model',
                'dataset': 'test_dataset',
                'metrics': {'ndcg@5': 0.75},
                'latency_ms': 10.0,
                'memory_bytes': 1024,
                'model_size_mb': 0.2,
                'constraint_violations': {}
            }
        ]
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_save_results_to_file(self):
        """Test saving results to file."""
        output_path = save_results_to_file(self.test_results, Path(self.temp_dir) / "test_exp")

        # Check that file was created
        self.assertTrue(output_path.exists())

        # Check file path
        expected_path = Path(self.temp_dir) / "test_exp" / "evaluation.json"
        self.assertEqual(output_path, expected_path)

    def test_load_results_from_file(self):
        """Test loading results from file."""
        # First save results
        save_path = save_results_to_file(self.test_results, Path(self.temp_dir) / "test_exp")

        # Then load them back
        loaded_results = load_results_from_file(Path(self.temp_dir) / "test_exp")

        # Check that loaded data matches original
        self.assertEqual(len(loaded_results), 1)
        self.assertEqual(loaded_results[0]['model'], 'test_model')
        self.assertEqual(loaded_results[0]['metrics']['ndcg@5'], 0.75)

    def test_save_load_roundtrip(self):
        """Test that save/load preserves data exactly."""
        # Save and load
        save_path = save_results_to_file(self.test_results, Path(self.temp_dir) / "roundtrip")
        loaded = load_results_from_file(Path(self.temp_dir) / "roundtrip")

        # Compare all fields
        self.assertEqual(loaded, self.test_results)

    def test_load_nonexistent_file(self):
        """Test error handling for nonexistent files."""
        with self.assertRaises(FileNotFoundError):
            load_results_from_file(Path(self.temp_dir) / "nonexistent")


if __name__ == '__main__':
    unittest.main()
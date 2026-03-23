"""
File: test_evaluator.py
Path: tests/evaluation/test_evaluator.py

Purpose:
  Unit tests for evaluation orchestration functions.

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
  - Uses mock models and datasets
  - Tests evaluation loop behavior
  - No real model training/inference

Related Docs:
  - docs/src/evaluation/evaluation.md
"""

import unittest
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
from src.evaluation.evaluator import Dataset, evaluate_model, evaluate_experiment
from src.data.schemas import TrainingRecord


class TestDataset(unittest.TestCase):
    """Test Dataset class functionality."""

    def setUp(self):
        """Set up test data."""
        self.train_records = [
            TrainingRecord(realistic=0.8, investigative=0.6, artistic=0.4,
                          social=0.7, enterprising=0.5, conventional=0.3,
                          career_category='Engineering, Manufacturing & Construction'),
            TrainingRecord(realistic=0.5, investigative=0.8, artistic=0.6,
                          social=0.4, enterprising=0.7, conventional=0.3,
                          career_category='Science & Mathematics')
        ]
        self.test_records = [
            TrainingRecord(realistic=0.7, investigative=0.5, artistic=0.6,
                          social=0.8, enterprising=0.4, conventional=0.3,
                          career_category='Arts, Communications & Humanities')
        ]
        self.feature_columns = ['realistic', 'investigative', 'artistic', 'social', 'enterprising', 'conventional']

    def test_dataset_creation(self):
        """Test Dataset object creation."""
        dataset = Dataset(self.train_records, self.test_records, self.feature_columns)

        self.assertEqual(len(dataset.train_records), 2)
        self.assertEqual(len(dataset.test_records), 1)
        self.assertEqual(dataset.feature_columns, self.feature_columns)

    def test_dataset_split(self):
        """Test Dataset.split() method."""
        dataset = Dataset(self.train_records, self.test_records, self.feature_columns)
        train_df, test_df = dataset.split()

        # Check DataFrame shapes
        self.assertEqual(train_df.shape, (2, 7))  # 2 rows, 6 features + category
        self.assertEqual(test_df.shape, (1, 7))   # 1 row, 6 features + category

        # Check column names
        expected_cols = self.feature_columns + ['career_category']
        self.assertEqual(list(train_df.columns), expected_cols)
        self.assertEqual(list(test_df.columns), expected_cols)

        # Check data values
        self.assertEqual(train_df.iloc[0]['realistic'], 0.8)
        self.assertEqual(train_df.iloc[0]['career_category'], 'Engineering, Manufacturing & Construction')
        self.assertEqual(test_df.iloc[0]['career_category'], 'Arts, Communications & Humanities')

    def test_dataset_empty_train(self):
        """Test error handling for empty train records."""
        with self.assertRaises(ValueError):
            Dataset([], self.test_records, self.feature_columns)

    def test_dataset_empty_test(self):
        """Test error handling for empty test records."""
        with self.assertRaises(ValueError):
            Dataset(self.train_records, [], self.feature_columns)

    def test_dataset_empty_features(self):
        """Test error handling for empty feature columns."""
        with self.assertRaises(ValueError):
            Dataset(self.train_records, self.test_records, [])


class TestEvaluateModel(unittest.TestCase):
    """Test evaluate_model function."""

    def setUp(self):
        """Set up mock objects."""
        # Create mock dataset
        train_records = [
            TrainingRecord(realistic=0.8, investigative=0.6, artistic=0.4,
                          social=0.7, enterprising=0.5, conventional=0.3,
                          career_category='Engineering, Manufacturing & Construction')
        ]
        test_records = [
            TrainingRecord(realistic=0.7, investigative=0.5, artistic=0.6,
                          social=0.8, enterprising=0.4, conventional=0.3,
                          career_category='Arts, Communications & Humanities')
        ]
        self.dataset = Dataset(train_records, test_records,
                              ['realistic', 'investigative', 'artistic', 'social', 'enterprising', 'conventional'])

        # Create mock model
        self.mock_model = Mock()
        self.mock_model.get_name.return_value = 'test_model'
        self.mock_model.y_feature = 'career_category'
        self.mock_model.test.return_value = pd.DataFrame({
            'Title': ['Job1', 'Job2'],
            'Career Category': ['Arts', 'Engineering'],
            'Match_Score': [0.9, 0.8]
        })

        # Mock O*NET database
        self.onet_db = pd.DataFrame({
            'Title': ['Job1', 'Job2'],
            'Career Category': ['Arts', 'Engineering']
        })

        self.config = {'evaluation': {'k_values': [5]}}

    @patch('src.evaluation.evaluator.benchmark_model_inference')
    @patch('src.evaluation.evaluator.compute_all_metrics')
    def test_evaluate_model_basic(self, mock_compute_metrics, mock_benchmark):
        """Test basic evaluate_model functionality."""
        mock_benchmark.return_value = {
            'latency_ms': 10.0,
            'memory_bytes': 1024,
            'predictions': self.mock_model.test.return_value
        }
        mock_compute_metrics.return_value = {'ndcg@5': 0.8, 'precision@5': 0.6}

        result = evaluate_model(self.mock_model, self.dataset, self.onet_db, self.config)

        # Check result structure
        self.assertIn('metrics', result)
        self.assertIn('latency_ms', result)
        self.assertIn('memory_bytes', result)
        self.assertIn('model_size_mb', result)
        self.assertIn('constraint_violations', result)

        # Check values
        self.assertEqual(result['metrics'], {'ndcg@5': 0.8, 'precision@5': 0.6})
        self.assertEqual(result['latency_ms'], 10.0)
        self.assertEqual(result['memory_bytes'], 1024)

    def test_evaluate_model_calls_benchmark(self):
        """Test that evaluate_model calls benchmark_model_inference for each test sample."""
        with patch('src.evaluation.evaluator.benchmark_model_inference') as mock_benchmark, \
             patch('src.evaluation.evaluator.compute_all_metrics'):

            mock_benchmark.return_value = {
                'latency_ms': 10.0,
                'memory_bytes': 1024,
                'predictions': self.mock_model.test.return_value
            }
            evaluate_model(self.mock_model, self.dataset, self.onet_db, self.config)

            # Should call benchmark once for one test sample
            self.assertEqual(mock_benchmark.call_count, 1)

    def test_evaluate_experiment_writes_incremental_output(self):
        """Test evaluate_experiment writes output path progressively."""
        dataset = self.dataset
        model = self.mock_model
        onet_db = self.onet_db
        config = {'evaluation': {'k_values': [5]}}

        output_dir = tempfile.mkdtemp()
        output_path = Path(output_dir) / 'evaluation.json'

        with patch('src.evaluation.evaluator.evaluate_model') as mock_eval:
            mock_eval.return_value = {
                'metrics': {'ndcg@5': 0.5},
                'latency_ms': 5.0,
                'memory_bytes': 512,
                'model_size_mb': 0.1,
                'constraint_violations': {}
            }
            results = evaluate_experiment([dataset], [model], onet_db, config, output_path=output_path)

        self.assertTrue(output_path.exists())
        with open(output_path, 'r') as f:
            saved_results = json.load(f)

        self.assertEqual(saved_results, results)

        shutil.rmtree(output_dir)

    def test_evaluate_model_no_test_method(self):
        """Test error handling when model has no test method."""
        bad_model = Mock(spec=['get_name'])
        bad_model.get_name.return_value = 'bad_model'

        with self.assertRaises(AttributeError):
            evaluate_model(bad_model, self.dataset, self.onet_db, self.config)


if __name__ == '__main__':
    unittest.main()
"""
File: test_benchmark.py
Path: tests/evaluation/test_benchmark.py

Purpose:
  Unit tests for benchmarking functions.

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
  - Tests are deterministic where possible
  - Mock model for testing
  - No external dependencies

Related Docs:
  - docs/src/evaluation/evaluation.md
"""

import unittest
import time
from unittest.mock import Mock
from src.evaluation.benchmark import measure_latency_ms, estimate_memory_usage, benchmark_model_inference, get_model_size_mb


class TestLatencyMeasurement(unittest.TestCase):
    """Test latency measurement functions."""

    def test_measure_latency_ms(self):
        """Test basic latency measurement."""
        def dummy_func():
            time.sleep(0.01)  # 10ms sleep
            return 42

        latency = measure_latency_ms(dummy_func)
        self.assertGreater(latency, 5.0)  # Should be at least 5ms
        self.assertLess(latency, 50.0)    # Should be less than 50ms

    def test_measure_latency_with_args(self):
        """Test latency measurement with function arguments."""
        def add_func(a, b):
            time.sleep(0.005)
            return a + b

        latency = measure_latency_ms(add_func, 1, 2)
        self.assertGreater(latency, 2.0)
        self.assertEqual(add_func(1, 2), 3)  # Function should still work


class TestMemoryEstimation(unittest.TestCase):
    """Test memory usage estimation."""

    def test_estimate_memory_usage(self):
        """Test memory estimation for different objects."""
        small_obj = 42
        small_size = estimate_memory_usage(small_obj)

        list_obj = [1, 2, 3, 4, 5]
        list_size = estimate_memory_usage(list_obj)

        dict_obj = {'a': 1, 'b': 2, 'c': 3}
        dict_size = estimate_memory_usage(dict_obj)

        # Lists and dicts should be larger than integers
        self.assertGreater(list_size, small_size)
        self.assertGreater(dict_size, small_size)

        # All sizes should be positive
        self.assertGreater(small_size, 0)
        self.assertGreater(list_size, 0)
        self.assertGreater(dict_size, 0)


class TestBenchmarkModelInference(unittest.TestCase):
    """Test model inference benchmarking."""

    def setUp(self):
        """Set up mock model for testing."""
        self.mock_model = Mock()
        self.mock_model.test.return_value = "mock_predictions"
        self.test_data = "mock_test_data"
        self.onet_db = "mock_onet_db"

    def test_benchmark_model_inference_basic(self):
        """Test basic model inference benchmarking."""
        results = benchmark_model_inference(self.mock_model, self.test_data, self.onet_db, num_runs=2)

        # Check that results contain expected keys
        self.assertIn('latency_ms', results)
        self.assertIn('memory_bytes', results)

        # Check that values are reasonable
        self.assertGreater(results['latency_ms'], 0)
        self.assertGreater(results['memory_bytes'], 0)

        # Check that model.test was called
        self.assertEqual(self.mock_model.test.call_count, 2)

    def test_benchmark_model_inference_calls_with_correct_args(self):
        """Test that benchmark calls model with correct arguments."""
        benchmark_model_inference(self.mock_model, self.test_data, self.onet_db, num_runs=1)

        self.mock_model.test.assert_called_once_with(self.test_data, self.onet_db)

    def test_benchmark_model_inference_no_test_method(self):
        """Test error handling when model has no test method."""
        bad_model = Mock(spec=[])  # Mock with no methods

        with self.assertRaises(AttributeError):
            benchmark_model_inference(bad_model, self.test_data, self.onet_db)


class TestModelSize(unittest.TestCase):
    """Test model size estimation."""

    def test_get_model_size_mb(self):
        """Test model size estimation."""
        model = [1, 2, 3, 4, 5] * 1000  # Create a reasonably sized object
        size_mb = get_model_size_mb(model)

        # Size should be positive and reasonable
        self.assertGreater(size_mb, 0)
        self.assertLess(size_mb, 1.0)  # Should be less than 1MB for this test

    def test_get_model_size_mb_empty_object(self):
        """Test model size for empty object."""
        model = None
        size_mb = get_model_size_mb(model)

        self.assertGreaterEqual(size_mb, 0)


if __name__ == '__main__':
    unittest.main()
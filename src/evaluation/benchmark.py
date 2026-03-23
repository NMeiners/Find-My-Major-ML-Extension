"""
File: benchmark.py
Path: src/evaluation/benchmark.py

Purpose:
  Measures performance benchmarks for model evaluation including per-sample latency
  and memory usage estimation. Uses standard library only to avoid new dependencies.

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
  - Memory measurement uses sys.getsizeof as approximation
  - Latency measured in milliseconds
  - No external profiling libraries allowed
  - Benchmarks are approximate and for relative comparison

Related Docs:
  - docs/src/evaluation/evaluation.md
"""

import sys
import time
from typing import Callable, Any, Dict, List
import gc
import pandas as pd


def measure_latency_ms(func: Callable, *args, **kwargs) -> float:
    """
    Name: measure_latency_ms

    Purpose:
      Measures execution time of a function in milliseconds.

    Inputs:
      - func: Callable — function to benchmark
      - *args, **kwargs — arguments to pass to func

    Outputs:
      - float — execution time in milliseconds

    Raises / Errors:
      - Any exception raised by func

    Notes:
      - Uses time.perf_counter for high precision timing
      - Returns wall-clock time including any I/O
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000
    return latency_ms


def estimate_memory_usage(obj: Any) -> int:
    """
    Name: estimate_memory_usage

    Purpose:
      Estimates memory usage of an object using sys.getsizeof.

    Inputs:
      - obj: Any — object to measure

    Outputs:
      - int — estimated memory usage in bytes

    Raises / Errors:
      - None

    Notes:
      - This is an approximation and may underestimate complex objects
      - Does not account for shared memory or garbage collection
    """
    return sys.getsizeof(obj)


def benchmark_model_inference(model: Any, test_data: Any, onet_db: pd.DataFrame, num_runs: int = 5) -> Dict[str, float]:
    """
    Name: benchmark_model_inference

    Purpose:
      Benchmarks model inference performance across multiple runs.

    Inputs:
      - model: Any 1 model instance with test() method
      - test_data: Any 1 test dataset (format expected by model.test())
      - onet_db: pd.DataFrame 1 O*NET database used for ranking jobs
      - num_runs: int — number of benchmark runs for averaging

    Outputs:
      - Dict[str, float] — benchmark results with keys 'latency_ms', 'memory_bytes'

    Raises / Errors:
      - AttributeError: if model does not have test() method

    Notes:
      - Runs inference multiple times and averages results
      - Forces garbage collection between runs for cleaner measurements
      - Memory measurement is of the prediction output
    """
    if not hasattr(model, 'test'):
        raise AttributeError("Model must have a 'test' method")

    latencies = []
    memory_usages = []

    for _ in range(num_runs):
        # Force garbage collection for cleaner memory measurement
        gc.collect()

        # Measure latency
        start_time = time.perf_counter()
        predictions = model.test(test_data, onet_db)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

        # Estimate memory usage of predictions
        memory_bytes = estimate_memory_usage(predictions)
        memory_usages.append(memory_bytes)

    return {
        'latency_ms': sum(latencies) / len(latencies),
        'memory_bytes': sum(memory_usages) / len(memory_usages),
        'predictions': predictions
    }


def get_model_size_mb(model: Any) -> float:
    """
    Name: get_model_size_mb

    Purpose:
      Estimates model size in megabytes.

    Inputs:
      - model: Any — model instance to measure

    Outputs:
      - float — estimated model size in MB

    Raises / Errors:
      - None

    Notes:
      - Uses sys.getsizeof which may not capture all model state
      - For ML models, this is typically an underestimate
    """
    size_bytes = estimate_memory_usage(model)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb
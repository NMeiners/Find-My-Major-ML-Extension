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
  - AI Assistant (2026-03-30) — Improved deep memory and serialized model-size estimation

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-03-30

Assumptions & Constraints:
  - Memory measurement is approximate and uses deep sizing for common container types
  - Latency measured in milliseconds
  - No external profiling libraries allowed
  - Benchmarks are approximate and for relative comparison

Related Docs:
  - docs/src/evaluation/evaluation.md
"""

import sys
import time
import pickle
from typing import Callable, Any, Dict
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
    _ = func(*args, **kwargs)
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000
    return latency_ms


def estimate_memory_usage(obj: Any, _seen: set[int] | None = None) -> int:
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
      - This is still an approximation for complex graphs and shared references
      - Uses deep sizing for DataFrame/Series and common Python containers
    """
    if obj is None:
        return 0

    if _seen is None:
        _seen = set()

    obj_id = id(obj)
    if obj_id in _seen:
        return 0
    _seen.add(obj_id)

    if isinstance(obj, pd.DataFrame):
        return int(obj.memory_usage(index=True, deep=True).sum())

    if isinstance(obj, pd.Series):
        return int(obj.memory_usage(index=True, deep=True))

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(
            estimate_memory_usage(key, _seen) + estimate_memory_usage(value, _seen)
            for key, value in obj.items()
        )
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(estimate_memory_usage(item, _seen) for item in obj)
    elif hasattr(obj, "__dict__"):
        size += estimate_memory_usage(vars(obj), _seen)

    return int(size)


def benchmark_model_inference(
    model: Any,
    test_data: Any,
    onet_db: pd.DataFrame,
    num_runs: int = 5,
) -> Dict[str, Any]:
    """
    Name: benchmark_model_inference

    Purpose:
      Benchmarks model inference performance across multiple runs.

    Inputs:
      - model: Any — model instance with test() method
      - test_data: Any — test dataset (format expected by model.test())
      - onet_db: pd.DataFrame — O*NET database used for ranking jobs
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
        # For multi-run micro-benchmarks, force GC between runs.
        # Skip this in the common num_runs=1 path to avoid major slowdown
        # during large evaluation loops.
        if num_runs > 1:
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
      - Prefers serialized estimator size via pickle when possible
      - Falls back to approximate recursive object sizing when serialization fails
    """
    model_obj = getattr(model, "_model", model)

    try:
        size_bytes = len(pickle.dumps(model_obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        size_bytes = estimate_memory_usage(model_obj)

    size_mb = size_bytes / (1024 * 1024)
    return size_mb

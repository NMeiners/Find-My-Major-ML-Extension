"""
File: __init__.py
Path: src/evaluation/__init__.py

Purpose:
  Exposes public interfaces for the evaluation module.

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
  - Only public interfaces are exposed
  - Internal modules are not imported

Related Docs:
  - docs/src/evaluation/evaluation.md
"""

from .evaluator import evaluate_experiment, evaluate_model, Dataset, save_evaluation_results
from .metrics import compute_all_metrics, cosine_similarity, ndcg_at_k, precision_at_k
from .benchmark import benchmark_model_inference, get_model_size_mb
from .reporting import format_evaluation_results, save_results_to_file, load_results_from_file
from .fairness import (compute_category_distribution, compute_fairness_score,
                      detect_overrepresented_categories, detect_underrepresented_categories)

__all__ = [
    'evaluate_experiment',
    'evaluate_model',
    'Dataset',
    'save_evaluation_results',
    'compute_all_metrics',
    'cosine_similarity',
    'ndcg_at_k',
    'precision_at_k',
    'benchmark_model_inference',
    'get_model_size_mb',
    'format_evaluation_results',
    'save_results_to_file',
    'load_results_from_file',
    'compute_category_distribution',
    'compute_fairness_score',
    'detect_overrepresented_categories',
    'detect_underrepresented_categories'
]
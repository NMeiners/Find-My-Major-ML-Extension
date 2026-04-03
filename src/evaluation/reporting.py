"""
File: reporting.py
Path: src/evaluation/reporting.py

Purpose:
  Handles result formatting and persistence for evaluation outputs.
  Formats evaluation results for human readability and saves to disk.

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
  - Results stored in experiments/results/<experiment_id>/evaluation.json
  - Output is human-readable JSON format
  - No external logging libraries

Related Docs:
  - docs/src/evaluation/evaluation.md
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any


def format_evaluation_results(results: List[Dict[str, Any]]) -> str:
    """
    Name: format_evaluation_results

    Purpose:
      Formats evaluation results into human-readable string.

    Inputs:
      - results: List[Dict[str, Any]] — raw evaluation results

    Outputs:
      - str — formatted results summary

    Raises / Errors:
      - None

    Notes:
      - Creates summary table format for console output
      - Shows key metrics and performance indicators
    """
    if not results:
        return "No evaluation results to display."

    lines = []
    lines.append("Evaluation Results Summary")
    lines.append("=" * 50)

    for result in results:
        lines.append(f"\nModel: {result['model']}")
        lines.append(f"Dataset: {result['dataset']}")
        lines.append("-" * 30)

        # Metrics
        metrics = result.get('metrics', {})
        for metric_name, value in metrics.items():
            lines.append(f"  {metric_name}: {value:.4f}")

        # Performance
        lines.append(f"  Latency: {result.get('latency_ms', 0):.2f} ms")
        lines.append(f"  Memory: {result.get('memory_bytes', 0)/1024:.2f} KB")
        lines.append(f"  Model Size: {result.get('model_size_mb', 0):.2f} MB")

        # Constraints
        violations = result.get('constraint_violations', {})
        if violations:
            lines.append("  Constraint Violations:")
            for constraint, value in violations.items():
                lines.append(f"    {constraint}: {value}")

    return "\n".join(lines)


def _get_csv_headers(results: List[Dict[str, Any]]) -> list[str]:
    base_fields = [
        'model',
        'dataset',
        'latency_ms',
        'memory_bytes',
        'model_size_mb',
        'samples_evaluated',
        'interrupted',
    ]

    metric_fields = sorted({
        metric_name
        for result in results
        for metric_name in result.get('metrics', {}).keys()
    })

    violation_fields = sorted({
        violation_name
        for result in results
        for violation_name in result.get('constraint_violations', {}).keys()
    })

    return base_fields + metric_fields + violation_fields


def save_results_to_csv(
    results: List[Dict[str, Any]],
    output_dir: Path,
    file_name: str = "evaluation.csv",
) -> Path:
    """
    Name: save_results_to_csv

    Purpose:
      Exports flattened evaluation results to a CSV file.

    Inputs:
      - results: List[Dict[str, Any]] — evaluation results to export
      - output_dir: Path — output directory path
      - file_name: str — CSV file name

    Outputs:
      - Path — path to saved CSV file

    Raises / Errors:
      - IOError: if file cannot be written

    Notes:
      - Writes header row using metric and constraint names discovered in results
      - Supports empty results by exporting a header-only CSV
    """
    output_path = output_dir / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = _get_csv_headers(results)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {
                'model': result.get('model'),
                'dataset': result.get('dataset'),
                'latency_ms': result.get('latency_ms'),
                'memory_bytes': result.get('memory_bytes'),
                'model_size_mb': result.get('model_size_mb'),
                'samples_evaluated': result.get('samples_evaluated'),
                'interrupted': result.get('interrupted', False),
            }
            row.update(result.get('metrics', {}))
            row.update(result.get('constraint_violations', {}))
            writer.writerow(row)

    return output_path


def save_results_to_file(results: List[Dict[str, Any]],
                        output_dir: Path) -> Path:
    """
    Name: save_results_to_file

    Purpose:
      Saves evaluation results to JSON file in standard location.

    Inputs:
      - results: List[Dict[str, Any]] — evaluation results to save
      - output_dir: Path — full output directory path

    Outputs:
      - Path — path to saved file

    Raises / Errors:
      - IOError: if directory cannot be created or file cannot be written

    Notes:
      - Creates directory structure if it doesn't exist
      - Saves as output_dir/evaluation.json
    """
    output_path = output_dir / "evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return output_path


def load_results_from_file(input_dir: Path) -> List[Dict[str, Any]]:
    """
    Name: load_results_from_file

    Purpose:
      Loads evaluation results from JSON file.

    Inputs:
      - input_dir: Path — full input directory path

    Outputs:
      - List[Dict[str, Any]] — loaded evaluation results

    Raises / Errors:
      - FileNotFoundError: if results file doesn't exist
      - json.JSONDecodeError: if file is corrupted

    Notes:
      - Loads from input_dir/evaluation.json
    """
    input_path = input_dir / "evaluation.json"

    with open(input_path, 'r') as f:
        results = json.load(f)

    return results
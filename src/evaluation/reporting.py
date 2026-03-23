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
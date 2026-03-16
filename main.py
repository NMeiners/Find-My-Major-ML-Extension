"""
File: main.py
Path: main.py

Purpose:
  Entry point for the machine learning training pipeline. Handles command-line argument parsing, configuration loading, and initiates the training process.

Original Author(s):
  - Nathan Meiners
  - AI Assistant

AI Tools Used:
  - GitHub Copilot - Code generation and documentation

Editors:
  - AI Assistant (2026-03-14) — Initial implementation

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-03-14

Assumptions & Constraints:
  - Executed from repository root
  - Config file exists and is valid
  - Output directory is writable

Related Docs:
  - docs/src/config/config_loader.md
"""

import argparse
import os
import sys
from src.config.config_loader import load_config


def main():
    """
    Name: main

    Purpose:
      Parses command-line arguments, loads the experiment configuration, creates the output directory, and prepares for training.

    Inputs:
      - Command-line arguments: config_path (positional)

    Outputs:
      - None (prints status, prepares config for training)

    Raises / Errors:
      - SystemExit: If config file does not exist or loading fails

    Notes:
      - Creates output directory under config['output']['directory']/run_id
      - Currently prints loaded config; training logic to be added later
    """
    parser = argparse.ArgumentParser(description="Run ML training pipeline with config file.")
    parser.add_argument('config_path', help='Path to the experiment configuration YAML file')

    args = parser.parse_args()

    config_path = args.config_path

    if not os.path.isfile(config_path):
        print(f"Error: Configuration file '{config_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    base_dir = config['output']['directory']
    run_id = config['run']['run_id']
    output_dir = os.path.join(base_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Configuration loaded successfully. Run ID: {run_id}")
    print(f"Output directory created: {output_dir}")
    # TODO: Pass config to training logic


if __name__ == "__main__":
    main()
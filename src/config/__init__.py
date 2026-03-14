"""
Module: src/config

Responsibility
This module handles configuration loading and management for experiments.
It owns the loading of YAML config files and generation of runtime metadata.
It does NOT handle training logic, data processing, or model evaluation.

Public Interfaces
- load_config: Loads and validates experiment configuration from YAML file
    - Calls: yaml.safe_load (external), datetime (stdlib)

Internal Structure
- config_loader.py: Main loader implementation

Data Contracts
- Inputs: YAML file path (str)
- Outputs: Dict with config and runtime metadata

Constraints
- Performance: Fast loading for small config files
- Memory: Minimal memory usage
- Privacy: No PII handling
- Bias considerations: N/A

Related Modules
- src/models/
- src/evaluation/

Related Documentation
- docs/src/config/config_loader.md
"""
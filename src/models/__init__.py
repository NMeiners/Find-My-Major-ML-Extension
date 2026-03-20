"""
File: __init__.py
Path: src/models/__init__.py

Purpose:
  Exports all public model classes, the MODEL_REGISTRY, and run_inference
  so that other modules can import from src.models cleanly. The MODEL_REGISTRY
  maps config string keys to model classes, enabling main.py to instantiate
  models by name from the experiment config without hardcoding class references.

Original Author(s):
  - Angela Fleenor
  - AI Assistant (Claude Sonnet 4.6)

AI Tools Used:
  - Claude Sonnet 4.6 - Code generation and documentation

Editors:
  - AI Assistant (2026-03-20) — Initial implementation

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-03-20

Assumptions & Constraints:
  - Registry keys must exactly match the model name strings used in exp_config.yaml
  - Adding a new model requires: new class file + entry in MODEL_REGISTRY here

Related Docs:
  - docs/src/models.md
"""

from src.models.base import BaseModel
from src.models.heuristic import HeuristicModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.gradient_boosting import GradientBoostingModel
from src.models.knn import KNNModel
from src.models.inference import run_inference

# Maps config model name strings to their class.
# To add a new model: create its class file and add an entry here.
MODEL_REGISTRY = {
    "heuristic": HeuristicModel,
    "logistic_regression": LogisticRegressionModel,
    "random_forest": RandomForestModel,
    "gradient_boosting": GradientBoostingModel,
    "knn": KNNModel,
}

__all__ = [
    "BaseModel",
    "HeuristicModel",
    "LogisticRegressionModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "KNNModel",
    "run_inference",
    "MODEL_REGISTRY",
]

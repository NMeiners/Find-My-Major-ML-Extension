"""
File: base.py
Path: src/models/base.py

Purpose:
  Defines the abstract base class that all model implementations must inherit.
  Enforces a consistent train/test interface so every model is interchangeable
  in the training pipeline. All configuration values are injected at construction
  time — no hardcoded feature names, counts, or hyperparameters.

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
  - All subclasses receive config values at construction time
  - x_features must be a non-empty list of strings matching column names in X_train/X_test
  - onet_db passed to test() must contain all columns in x_features

Related Docs:
  - docs/src/models.md
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseModel(ABC):
    """
    Name: BaseModel

    Purpose:
      Abstract base class for all RIASEC recommendation models.
      Enforces a consistent interface (train/test/get_name) so the training
      pipeline can treat all models interchangeably.

    Inputs:
      - x_features: list[str] — input feature column names, from config models[i].x_features
      - y_feature: str — target label column name, from config models[i].y_features[0]
      - parameters: dict — model-specific hyperparameters, from config models[i].parameters
      - top_n_jobs: int — number of job recommendations to return, from config evaluation.top_k
      - top_n_categories: int — number of career categories to filter before ranking,
                                from config models[i].parameters.top_n_categories

    Outputs:
      - Subclass instances that implement train() and test()

    Raises / Errors:
      - TypeError: if instantiated directly (abstract class)

    Notes:
      - Nothing is hardcoded. All values come from the experiment config.
      - Subclasses must implement train(), test(), and get_name().
      - test() must always return a DataFrame with columns [Title, Career Category, Match_Score]
        and exactly top_n_jobs rows.
    """

    def __init__(
        self,
        x_features: list,
        y_feature: str,
        parameters: dict,
        top_n_jobs: int,
        top_n_categories: int,
    ):
        self.x_features = x_features
        self.y_feature = y_feature
        self.parameters = parameters
        self.top_n_jobs = top_n_jobs
        self.top_n_categories = top_n_categories

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Name: train

        Purpose:
          Fit the model on labeled training data.

        Inputs:
          - X_train: pd.DataFrame — training features; must contain all columns in self.x_features
          - y_train: pd.Series — career category labels

        Outputs:
          - None (model state is updated in place)

        Raises / Errors:
          - KeyError: if X_train is missing a required feature column

        Notes:
          - Must use only self.x_features columns from X_train, never hardcode column names.
        """
        ...

    @abstractmethod
    def test(self, X_test: pd.DataFrame, onet_db: pd.DataFrame) -> pd.DataFrame:
        """
        Name: test

        Purpose:
          Generate job recommendations for each row in X_test.

        Inputs:
          - X_test: pd.DataFrame — test features; must contain all columns in self.x_features
          - onet_db: pd.DataFrame — career database for ranking; loaded from config onet_db_path

        Outputs:
          - pd.DataFrame with columns [Title, Career Category, Match_Score]
            and exactly self.top_n_jobs rows

        Raises / Errors:
          - RuntimeError: if called before train() (for models that require training)
          - KeyError: if onet_db is missing required feature columns

        Notes:
          - Uses self.top_n_jobs and self.top_n_categories — never hardcodes these values.
          - Delegates final ranking to rank_jobs() from ranking.py.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """
        Name: get_name

        Purpose:
          Return this model's string identifier, which must match the key used in
          the experiment config and MODEL_REGISTRY.

        Inputs:
          - None

        Outputs:
          - str — model identifier (e.g. "random_forest", "logistic_regression")

        Raises / Errors:
          - None

        Notes:
          - Must match the config key exactly for the MODEL_REGISTRY lookup to work.
        """
        ...

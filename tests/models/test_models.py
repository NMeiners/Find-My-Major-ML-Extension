"""
File: test_models.py
Path: tests/models/test_models.py

Purpose:
  Unit tests for all model classes in src/models. Verifies that each model
  correctly implements the BaseModel interface, trains without error on synthetic
  data, and returns results with the expected shape and columns. All test
  parameters (feature names, result counts) are defined as variables — nothing
  is hardcoded in assertions.

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
  - Tests use synthetic in-memory DataFrames; no real data files required
  - All models must be trainable on the small synthetic dataset provided
  - onet_db fixture must contain the same feature columns as x_features

Related Docs:
  - docs/src/models.md
"""

import pytest
import pandas as pd
import numpy as np

from src.models.base import BaseModel
from src.models.heuristic import HeuristicModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.gradient_boosting import GradientBoostingModel
from src.models.knn import KNNModel
from src.models import MODEL_REGISTRY

# ── Shared test configuration ─────────────────────────────────────────────────
# Changing these values exercises the "nothing hardcoded" design requirement.
FEATURES = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]
LABEL_COL = "career_label"
CATEGORY_COL = "Career Category"
TOP_N_JOBS = 3
TOP_N_CATEGORIES = 2
N_TRAIN_ROWS = 30
N_ONET_ROWS = 20
CATEGORIES = ["Science", "Arts", "Business"]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def train_data():
    """
    Name: train_data

    Purpose:
      Produces a small synthetic training DataFrame with RIASEC feature columns
      and a career label column.

    Inputs:
      - None

    Outputs:
      - tuple[pd.DataFrame, pd.Series] — (X_train, y_train)

    Notes:
      - Uses a fixed random seed for reproducibility.
    """
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((N_TRAIN_ROWS, len(FEATURES))), columns=FEATURES)
    y = pd.Series(
        [CATEGORIES[i % len(CATEGORIES)] for i in range(N_TRAIN_ROWS)],
        name=LABEL_COL,
    )
    return X, y


@pytest.fixture
def onet_db():
    """
    Name: onet_db

    Purpose:
      Produces a small synthetic O*NET career database DataFrame with feature
      columns, a Career Category column, and a Title column.

    Inputs:
      - None

    Outputs:
      - pd.DataFrame — synthetic career database

    Notes:
      - Categories must overlap with those in train_data for filtering to return results.
    """
    rng = np.random.default_rng(0)
    db = pd.DataFrame(rng.random((N_ONET_ROWS, len(FEATURES))), columns=FEATURES)
    db[CATEGORY_COL] = [CATEGORIES[i % len(CATEGORIES)] for i in range(N_ONET_ROWS)]
    db["Title"] = [f"Job {i}" for i in range(N_ONET_ROWS)]
    return db


@pytest.fixture
def student_vector():
    """
    Name: student_vector

    Purpose:
      Produces a single-row test DataFrame representing one student's feature scores.

    Inputs:
      - None

    Outputs:
      - pd.DataFrame — one-row student feature vector
    """
    rng = np.random.default_rng(7)
    return pd.DataFrame(rng.random((1, len(FEATURES))), columns=FEATURES)


def _make_model(ModelClass, extra_params=None):
    """Build a model instance with shared test config values."""
    params = dict(extra_params or {})
    return ModelClass(
        x_features=FEATURES,
        y_feature=LABEL_COL,
        parameters=params,
        top_n_jobs=TOP_N_JOBS,
        top_n_categories=TOP_N_CATEGORIES,
    )


# ── Registry tests ────────────────────────────────────────────────────────────

def test_registry_contains_all_models():
    """
    Name: test_registry_contains_all_models

    Purpose:
      Verify that MODEL_REGISTRY maps all expected config keys to model classes.

    Notes:
      - If you add a new model, add its key here.
    """
    expected_keys = {
        "heuristic", "logistic_regression", "random_forest",
        "gradient_boosting", "knn",
    }
    assert expected_keys == set(MODEL_REGISTRY.keys())


def test_registry_values_are_base_model_subclasses():
    """
    Name: test_registry_values_are_base_model_subclasses

    Purpose:
      Verify every class in MODEL_REGISTRY is a proper subclass of BaseModel.
    """
    for name, cls in MODEL_REGISTRY.items():
        assert issubclass(cls, BaseModel), f"{name} is not a subclass of BaseModel"


# ── Interface tests (parametrized over all models) ────────────────────────────

ALL_MODELS = [
    (HeuristicModel, {}),
    (LogisticRegressionModel, {"max_iter": 100}),
    (RandomForestModel, {"n_estimators": 10}),
    (GradientBoostingModel, {"n_estimators": 10}),
    (KNNModel, {"n_neighbors": 3}),
]


@pytest.mark.parametrize("ModelClass,params", ALL_MODELS)
def test_model_is_base_model_instance(ModelClass, params):
    """
    Name: test_model_is_base_model_instance

    Purpose:
      Verify each model is an instance of BaseModel.
    """
    model = _make_model(ModelClass, params)
    assert isinstance(model, BaseModel)


@pytest.mark.parametrize("ModelClass,params", ALL_MODELS)
def test_get_name_returns_string(ModelClass, params):
    """
    Name: test_get_name_returns_string

    Purpose:
      Verify get_name() returns a non-empty string matching a MODEL_REGISTRY key.
    """
    model = _make_model(ModelClass, params)
    name = model.get_name()
    assert isinstance(name, str) and len(name) > 0
    assert name in MODEL_REGISTRY


@pytest.mark.parametrize("ModelClass,params", ALL_MODELS)
def test_train_runs_without_error(ModelClass, params, train_data):
    """
    Name: test_train_runs_without_error

    Purpose:
      Verify train() completes without raising an exception.
    """
    X_train, y_train = train_data
    model = _make_model(ModelClass, params)
    model.train(X_train, y_train)  # should not raise


@pytest.mark.parametrize("ModelClass,params", ALL_MODELS)
def test_test_returns_dataframe(ModelClass, params, train_data, onet_db, student_vector):
    """
    Name: test_test_returns_dataframe

    Purpose:
      Verify test() returns a pd.DataFrame after training.
    """
    X_train, y_train = train_data
    model = _make_model(ModelClass, params)
    model.train(X_train, y_train)
    result = model.test(student_vector, onet_db)
    assert isinstance(result, pd.DataFrame)


@pytest.mark.parametrize("ModelClass,params", ALL_MODELS)
def test_test_returns_correct_columns(ModelClass, params, train_data, onet_db, student_vector):
    """
    Name: test_test_returns_correct_columns

    Purpose:
      Verify test() output has exactly [Title, Career Category, Match_Score] columns.
    """
    X_train, y_train = train_data
    model = _make_model(ModelClass, params)
    model.train(X_train, y_train)
    result = model.test(student_vector, onet_db)
    assert list(result.columns) == ["Title", CATEGORY_COL, "Match_Score"]


@pytest.mark.parametrize("ModelClass,params", ALL_MODELS)
def test_test_returns_top_n_jobs_rows(ModelClass, params, train_data, onet_db, student_vector):
    """
    Name: test_test_returns_top_n_jobs_rows

    Purpose:
      Verify test() returns exactly TOP_N_JOBS rows — not a hardcoded number.
      Changing TOP_N_JOBS at the top of this file must change the assertion.
    """
    X_train, y_train = train_data
    model = _make_model(ModelClass, params)
    model.train(X_train, y_train)
    result = model.test(student_vector, onet_db)
    assert len(result) == TOP_N_JOBS


@pytest.mark.parametrize("ModelClass,params", ALL_MODELS)
def test_match_scores_in_valid_range(ModelClass, params, train_data, onet_db, student_vector):
    """
    Name: test_match_scores_in_valid_range

    Purpose:
      Verify all Match_Score values are cosine similarity values in [0, 1].
    """
    X_train, y_train = train_data
    model = _make_model(ModelClass, params)
    model.train(X_train, y_train)
    result = model.test(student_vector, onet_db)
    assert result["Match_Score"].between(0.0, 1.0).all()


# ── Error handling tests ──────────────────────────────────────────────────────

@pytest.mark.parametrize("ModelClass,params", [
    (LogisticRegressionModel, {"max_iter": 100}),
    (RandomForestModel, {"n_estimators": 10}),
    (GradientBoostingModel, {"n_estimators": 10}),
    (KNNModel, {"n_neighbors": 3}),
])
def test_test_before_train_raises(ModelClass, params, onet_db, student_vector):
    """
    Name: test_test_before_train_raises

    Purpose:
      Verify that calling test() before train() raises RuntimeError for
      models that require training.

    Notes:
      - HeuristicModel is excluded because its train() is a no-op.
    """
    model = _make_model(ModelClass, params)
    with pytest.raises(RuntimeError):
        model.test(student_vector, onet_db)

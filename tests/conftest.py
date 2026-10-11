import matplotlib
import numpy as np
import pandas as pd
import pytest

# Use a non-interactive backend so plot tests also work in CI/headless systems.
matplotlib.use("Agg")


@pytest.fixture
def sample_df():
    """Small mixed dataset with two complete and two incomplete numeric columns."""
    return pd.DataFrame(
        {
            "complete_a": [1, 2, 3, 4, 5, 6, 7, 8],
            "complete_b": [8, 7, 6, 5, 4, 3, 2, 1],
            "missing_a": [1.0, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0],
            "missing_b": [np.nan, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0],
        }
    )


@pytest.fixture
def complete_df():
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6],
            "B": [10, 20, 30, 40, 50, 60],
        }
    )


@pytest.fixture
def model_df():
    """Dataset large enough for stable stratified train/test splits."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    target = 2 * x1 - 0.5 * x2 + rng.normal(scale=0.2, size=n)

    # 25% missingness with a clear dependence on x1.
    target[x1 > np.quantile(x1, 0.75)] = np.nan

    return pd.DataFrame({"x1": x1, "x2": x2, "target": target})


@pytest.fixture
def continuous_imputation_df():
    x = np.linspace(0, 10, 40)
    target = 2.0 * x + 1.0
    target[[3, 8, 15, 22, 31]] = np.nan
    return pd.DataFrame({"x": x, "x2": x**2, "target": target})


@pytest.fixture
def binary_imputation_df():
    x = np.arange(40, dtype=float)
    target = (x % 2).astype(float)
    target[[2, 7, 13, 21, 29, 35]] = np.nan
    return pd.DataFrame({"x": x, "x2": x / 10, "target": target})


@pytest.fixture
def discrete_imputation_df():
    x = np.arange(40, dtype=float)
    target = (2 + (x % 6)).astype(float)
    target[[1, 6, 12, 19, 27, 34]] = np.nan
    return pd.DataFrame({"x": x, "x2": x / 5, "target": target})


@pytest.fixture
def categorical_imputation_df():
    x = np.arange(45, dtype=float)
    target = np.array(["A", "B", "C"] * 15, dtype=object)
    target[[2, 8, 14, 20, 26, 32, 38, 44]] = np.nan
    return pd.DataFrame({"x": x, "x2": x % 5, "target": target})


@pytest.fixture
def little_df():
    """Numeric data satisfying mechanism_viewer's Little-MCAR input requirements."""
    return pd.DataFrame(
        {
            "complete": np.arange(12, dtype=float),
            "miss_a": [1, np.nan, 3, 4, np.nan, 6, 7, 8, 9, 10, 11, 12],
            "miss_b": [1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10, 11, 12],
        }
    )

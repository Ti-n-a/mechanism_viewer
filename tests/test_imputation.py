import numpy as np
import pandas as pd
import pytest

from mechanism_viewer import ColType, plot_imputation_distribution, scatter_imputation_comparison


def _assert_observed_values_unchanged(original, imputed, column):
    observed = original[column].notna()
    pd.testing.assert_series_equal(
        imputed.loc[observed, column],
        original.loc[observed, column],
        check_names=False,
        check_dtype=False,
    )


def test_continuous_imputation_fills_target_and_preserves_observed_values(continuous_imputation_df):
    _, _, imputed = plot_imputation_distribution(
        continuous_imputation_df,
        "target",
        missing_col_type=ColType.CONTINUOUS,
        random_state=42,
        display_plot=False,
    )

    assert imputed["target"].isna().sum() == 0
    _assert_observed_values_unchanged(continuous_imputation_df, imputed, "target")


def test_binary_imputation_fills_with_only_binary_values(binary_imputation_df):
    _, _, imputed = plot_imputation_distribution(
        binary_imputation_df,
        "target",
        missing_col_type=ColType.BINARY,
        random_state=42,
        display_plot=False,
    )

    assert imputed["target"].isna().sum() == 0
    assert set(imputed["target"].unique()).issubset({0, 1, 0.0, 1.0})
    _assert_observed_values_unchanged(binary_imputation_df, imputed, "target")


def test_discrete_imputation_produces_nonnegative_integer_values(discrete_imputation_df):
    _, _, imputed = plot_imputation_distribution(
        discrete_imputation_df,
        "target",
        missing_col_type=ColType.DISCRETE,
        random_state=42,
        display_plot=False,
    )

    values = imputed["target"].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert (values >= 0).all()
    assert np.allclose(values, np.round(values))


def test_categorical_imputation_uses_observed_categories(categorical_imputation_df):
    original_categories = set(categorical_imputation_df["target"].dropna().unique())

    _, _, imputed = plot_imputation_distribution(
        categorical_imputation_df,
        "target",
        missing_col_type=ColType.DISC_CATEGORICAL,
        random_state=42,
        display_plot=False,
    )

    assert imputed["target"].isna().sum() == 0
    assert set(imputed["target"].unique()).issubset(original_categories)


def test_scatter_imputation_comparison_returns_plot(continuous_imputation_df):
    fig, ax = scatter_imputation_comparison(
        continuous_imputation_df,
        column_name="x",
        missing_col="target",
        missing_col_type=ColType.CONTINUOUS,
        random_state=42,
        display_plot=False,
    )

    assert fig is not None
    assert ax is not None


def test_binary_imputation_rejects_non_binary_target(binary_imputation_df):
    df = binary_imputation_df.copy()
    df.loc[df["target"].notna(), "target"] = [
        0 if i % 2 == 0 else 2 for i in range(df["target"].notna().sum())
    ]

    with pytest.raises(ValueError):
        plot_imputation_distribution(
            df,
            "target",
            missing_col_type=ColType.BINARY,
            display_plot=False,
        )


def test_discrete_imputation_rejects_negative_observed_values(discrete_imputation_df):
    df = discrete_imputation_df.copy()
    first_observed = df["target"].first_valid_index()
    df.loc[first_observed, "target"] = -1

    with pytest.raises(ValueError):
        plot_imputation_distribution(
            df,
            "target",
            missing_col_type=ColType.DISCRETE,
            display_plot=False,
        )


def test_noncontinuous_imputation_requires_complete_predictor():
    df = pd.DataFrame(
        {
            "x": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            "target": [0.0, 1.0, np.nan, 1.0, 0.0, np.nan],
        }
    )

    with pytest.raises(ValueError):
        plot_imputation_distribution(
            df,
            "target",
            missing_col_type=ColType.BINARY,
            display_plot=False,
        )

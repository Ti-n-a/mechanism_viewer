import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mechanism_viewer import (
    boxplot_comparison,
    complete_and_misscol_corr,
    missing_rate_matrix,
    missing_rows_matrix,
    missingness_dendrogram,
    missingness_misscol_corr,
    misscol_vs_all_corr,
    parallel_comparison,
    plot_missing_rate,
    plot_missingness_distribution,
    plot_total_missingness,
    scatter_missingness_comparison,
    scatter_missingness_comparison_line,
    upset_missing_rows,
    value_misscol_corr,
)


def assert_fig_ax(result):
    fig, ax = result
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_simple_viewers_return_matplotlib_objects(sample_df):
    assert_fig_ax(plot_total_missingness(sample_df, display_plot=False))
    assert_fig_ax(plot_missing_rate(sample_df, display_plot=False))


def test_missingness_distribution_returns_one_plot_per_other_column(sample_df):
    plots = plot_missingness_distribution(sample_df, "missing_a", display_plot=False)

    assert len(plots) == 2 * (len(sample_df.columns) - 1)
    for i in range(0, len(plots), 2):
        assert isinstance(plots[i], Figure)
        assert isinstance(plots[i + 1], Axes)


def test_matrix_viewers_return_matplotlib_objects(sample_df):
    assert_fig_ax(missing_rows_matrix(sample_df, display_plot=False))
    assert_fig_ax(
        missing_rate_matrix(
            sample_df,
            column_name="complete_a",
            display_plot=False,
        )
    )


def test_missing_rows_matrix_rejects_complete_dataframe(complete_df):
    with pytest.raises(ValueError):
        missing_rows_matrix(complete_df, display_plot=False)


def test_comparison_viewers_return_matplotlib_objects(sample_df):
    assert_fig_ax(
        scatter_missingness_comparison(
            sample_df,
            "complete_a",
            "missing_a",
            display_plot=False,
        )
    )
    assert_fig_ax(
        scatter_missingness_comparison_line(
            sample_df,
            "complete_a",
            "missing_a",
            display_plot=False,
        )
    )
    assert_fig_ax(
        boxplot_comparison(
            sample_df,
            "complete_a",
            "missing_a",
            display_plot=False,
        )
    )
    assert_fig_ax(parallel_comparison(sample_df, "missing_a", display_plot=False))


def test_parallel_comparison_requires_numeric_predictor():
    df = pd.DataFrame(
        {
            "category": ["a", "b", "c", "d"],
            "target": [1.0, None, 3.0, 4.0],
        }
    )

    with pytest.raises(ValueError):
        parallel_comparison(df, "target", display_plot=False)


def test_correlation_viewers_return_matplotlib_objects(sample_df):
    assert_fig_ax(missingness_misscol_corr(sample_df, display_plot=False))
    assert_fig_ax(value_misscol_corr(sample_df, display_plot=False))
    assert_fig_ax(complete_and_misscol_corr(sample_df, display_plot=False))
    assert_fig_ax(misscol_vs_all_corr(sample_df, "missing_a", display_plot=False))


def test_upset_viewer_returns_matplotlib_objects(sample_df):
    assert_fig_ax(upset_missing_rows(sample_df, display_plot=False))


def test_dendrogram_returns_matplotlib_objects(sample_df):
    assert_fig_ax(missingness_dendrogram(sample_df, display_plot=False))


def test_dendrogram_rejects_invalid_linkage_method(sample_df):
    with pytest.raises(ValueError):
        missingness_dendrogram(sample_df, linkage_method="invalid", display_plot=False)


def test_dendrogram_requires_two_varying_missing_columns():
    df = pd.DataFrame(
        {
            "complete": [1, 2, 3, 4],
            "only_missing_col": [1.0, None, 3.0, 4.0],
        }
    )

    with pytest.raises(ValueError):
        missingness_dendrogram(df, display_plot=False)

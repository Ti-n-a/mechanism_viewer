import numpy as np
import pandas as pd
import pytest

import mechanism_viewer.little_mcar as little_module
from mechanism_viewer import (
    interpret_mcar_p_value,
    little_mcar_global,
    little_mcar_pairwise,
    plot_mcar_pairwise,
)


def test_interpret_mcar_p_value_above_alpha_fails_to_reject():
    text = interpret_mcar_p_value(0.20, alpha=0.05)
    assert "Fail to reject" in text
    assert "likely MCAR" in text


def test_interpret_mcar_p_value_below_alpha_rejects():
    text = interpret_mcar_p_value(0.01, alpha=0.05)
    assert "Reject the null hypothesis" in text
    assert "MAR or MNAR" in text


def test_little_mcar_rejects_non_numeric_input(little_df):
    df = little_df.copy()
    df["text"] = ["a", "b"] * 6

    with pytest.raises(ValueError):
        little_mcar_global(df)


def test_little_mcar_requires_two_sufficiently_incomplete_columns():
    df = pd.DataFrame(
        {
            "complete": np.arange(8, dtype=float),
            "only_one_missing_col": [1, np.nan, 3, np.nan, 5, 6, 7, 8],
        }
    )

    with pytest.raises(ValueError):
        little_mcar_global(df)


def test_little_mcar_global_returns_dependency_result(monkeypatch, little_df):
    class FakeMCARTest:
        def __init__(self, method):
            assert method == "little"

        def little_mcar_test(self, df):
            assert set(df.columns) == {"complete", "miss_a", "miss_b"}
            return 0.42

    monkeypatch.setattr(little_module, "MCARTest", FakeMCARTest)

    result = little_mcar_global(little_df)
    assert result == 0.42


def test_little_mcar_pairwise_returns_dependency_matrix(monkeypatch, little_df):
    expected = pd.DataFrame(
        [[np.nan, 0.20], [0.20, np.nan]],
        index=["miss_a", "miss_b"],
        columns=["miss_a", "miss_b"],
    )

    class FakeMCARTest:
        def __init__(self, method):
            assert method == "ttest"

        def mcar_t_tests(self, df):
            return expected.copy()

    monkeypatch.setattr(little_module, "MCARTest", FakeMCARTest)

    result = little_mcar_pairwise(little_df)
    pd.testing.assert_frame_equal(result, expected)


def test_plot_mcar_pairwise_returns_two_figures():
    p_values = pd.DataFrame(
        [[np.nan, 0.01], [0.01, np.nan]],
        columns=["A", "B"],
        index=["A", "B"],
    )

    result = plot_mcar_pairwise(p_values, alpha=0.05, display_plot=False)

    assert len(result) == 4
    fig1, ax1, fig2, ax2 = result
    assert fig1 is not None and ax1 is not None
    assert fig2 is not None and ax2 is not None

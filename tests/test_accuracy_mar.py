import pandas as pd
import pytest

from mechanism_viewer import (
    interpret_mar_abd,
    run_logistic_regression,
    run_random_forest,
)
from mechanism_viewer import test_mar_from_model_accuracy as calculate_mar_abd


def test_random_forest_returns_valid_accuracy(model_df):
    accuracy = run_random_forest(model_df, "target", random_state=42, print_result=False)

    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0


def test_logistic_regression_returns_valid_accuracy(model_df):
    accuracy = run_logistic_regression(model_df, "target", random_state=42, print_result=False)

    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0


def test_random_forest_is_reproducible(model_df):
    first = run_random_forest(model_df, "target", random_state=10, print_result=False)
    second = run_random_forest(model_df, "target", random_state=10, print_result=False)

    assert first == second


def test_model_functions_reject_non_numeric_complete_predictors(model_df):
    df = model_df.copy()
    df["text"] = ["a", "b"] * (len(df) // 2)

    with pytest.raises(ValueError):
        run_logistic_regression(df, "target", print_result=False)


def test_model_functions_reject_dataset_without_complete_predictors():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, None, 4.0, 5.0, 6.0],
            "target": [1.0, None, 3.0, 4.0, None, 6.0],
        }
    )

    with pytest.raises(ValueError):
        run_random_forest(df, "target", print_result=False)


def test_interpret_mar_abd_above_threshold_reports_mar():
    text = interpret_mar_abd(8.0, threshold=5.0)
    assert "likely" in text
    assert "MAR" in text
    assert "MCAR/MNAR" not in text


def test_interpret_mar_abd_at_threshold_reports_mcar_or_mnar():
    text = interpret_mar_abd(5.0, threshold=5.0)
    assert "MCAR/MNAR" in text


def test_calculate_mar_abd_returns_percentage_point_difference(model_df):
    abd = calculate_mar_abd(
        model_df,
        "target",
        threshold=5.0,
        random_state=42,
        print_result=False,
    )

    assert isinstance(abd, float)
    assert -100.0 <= abd <= 100.0

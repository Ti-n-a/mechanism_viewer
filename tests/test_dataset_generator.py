import numpy as np
import pandas as pd
import pytest

from mechanism_viewer import (
    ColType,
    apply_mar,
    apply_mcar,
    apply_missingness,
    apply_mnar,
    generate_dataset_with_missingness,
    generate_synthetic_dataset,
)


def test_generate_synthetic_dataset_shape_and_columns():
    df = generate_synthetic_dataset(
        n_rows=50,
        type_array=[ColType.CONTINUOUS, ColType.DISCRETE, ColType.BINARY],
        random_state=7,
    )

    assert df.shape == (50, 3)
    assert list(df.columns) == ["Col1", "Col2", "Col3"]


def test_generate_synthetic_dataset_is_reproducible():
    types = [ColType.CONTINUOUS, ColType.DISCRETE, ColType.BINARY]
    first = generate_synthetic_dataset(30, types, random_state=123)
    second = generate_synthetic_dataset(30, types, random_state=123)

    pd.testing.assert_frame_equal(first, second)


def test_generate_synthetic_dataset_rejects_unknown_type():
    with pytest.raises(ValueError):
        generate_synthetic_dataset(10, ["not-a-supported-type"])


def test_apply_mcar_creates_expected_number_of_missing_values():
    series = pd.Series(np.arange(100, dtype=float), name="x")
    result = apply_mcar(series, missing_rate=0.20, random_state=42)

    assert result.isna().sum() == 20
    assert series.isna().sum() == 0  # original input is unchanged


def test_apply_mcar_is_reproducible():
    series = pd.Series(np.arange(100, dtype=float), name="x")
    first = apply_mcar(series, missing_rate=0.20, random_state=42)
    second = apply_mcar(series, missing_rate=0.20, random_state=42)

    pd.testing.assert_series_equal(first, second)


def test_apply_mar_removes_rows_with_highest_observable_scores():
    target = pd.Series([10, 20, 30, 40, 50], name="target", dtype=float)
    observable = pd.DataFrame({"obs": [1, 2, 3, 4, 5]})

    result = apply_mar(
        target,
        observable,
        missing_rate=0.40,
        missingness_ascending=True,
    )

    assert result.isna().sum() == 2
    assert result.iloc[[3, 4]].isna().all()
    assert result.iloc[:3].notna().all()


def test_apply_mar_descending_removes_lowest_observable_scores():
    target = pd.Series([10, 20, 30, 40, 50], name="target", dtype=float)
    observable = pd.DataFrame({"obs": [1, 2, 3, 4, 5]})

    result = apply_mar(
        target,
        observable,
        missing_rate=0.40,
        missingness_ascending=False,
    )

    assert result.iloc[[0, 1]].isna().all()
    assert result.iloc[2:].notna().all()


def test_apply_mnar_removes_highest_target_values():
    target = pd.Series([1, 2, 3, 4, 5], name="target", dtype=float)

    result = apply_mnar(target, missing_rate=0.40, missingness_ascending=True)

    assert result.isna().sum() == 2
    assert result.iloc[[3, 4]].isna().all()


def test_apply_mnar_descending_removes_lowest_target_values():
    target = pd.Series([1, 2, 3, 4, 5], name="target", dtype=float)

    result = apply_mnar(target, missing_rate=0.40, missingness_ascending=False)

    assert result.iloc[[0, 1]].isna().all()
    assert result.iloc[2:].notna().all()


def test_zero_missing_rate_leaves_mar_column_unchanged():
    target = pd.Series([1, 2, 3, 4, 5], name="target", dtype=float)
    observable = pd.DataFrame({"obs": [5, 4, 3, 2, 1]})

    result = apply_mar(target, observable, missing_rate=0.0)

    pd.testing.assert_series_equal(result, target)


def test_zero_missing_rate_leaves_mnar_column_unchanged():
    target = pd.Series([1, 2, 3, 4, 5], name="target", dtype=float)

    result = apply_mnar(target, missing_rate=0.0)

    pd.testing.assert_series_equal(result, target)


def test_missing_rate_outside_valid_range_is_rejected():
    target = pd.Series([1, 2, 3], name="target", dtype=float)

    with pytest.raises(ValueError):
        apply_mcar(target, missing_rate=1.1)


def test_mechanism_cannot_be_applied_to_already_missing_column():
    target = pd.Series([1.0, np.nan, 3.0], name="target")

    with pytest.raises(ValueError):
        apply_mcar(target, missing_rate=0.2)


def test_apply_missingness_keeps_complete_columns_complete():
    original = pd.DataFrame(
        {
            "complete": np.arange(20, dtype=float),
            "mcar": np.arange(20, dtype=float),
            "mnar": np.arange(20, dtype=float),
        }
    )

    result = apply_missingness(
        original,
        n_complete_cols=1,
        missing_mechanism_array=["MCAR", "MNAR"],
        missing_rate_array=[0.20, 0.25],
        random_state=42,
    )

    assert result["complete"].isna().sum() == 0
    assert result["mcar"].isna().sum() == 4
    assert result["mnar"].isna().sum() == 5
    assert original.isna().sum().sum() == 0


def test_generate_dataset_with_missingness_applies_requested_rates():
    result = generate_dataset_with_missingness(
        n_rows=100,
        type_array=[ColType.CONTINUOUS, ColType.CONTINUOUS, ColType.CONTINUOUS],
        n_complete_cols=1,
        missing_mechanism_array=["MCAR", "MNAR"],
        missing_rate_array=[0.10, 0.20],
        random_state=42,
    )

    assert result["Col1"].isna().sum() == 0
    assert result["Col2"].isna().sum() == 10
    assert result["Col3"].isna().sum() == 20

import numpy as np
import pandas as pd
import pytest

from mechanism_viewer._validation import (
    validate_column,
    validate_dataframe,
    validate_missing_col,
)


def test_validate_dataframe_rejects_non_dataframe():
    with pytest.raises(TypeError):
        validate_dataframe([1, 2, 3])


def test_validate_dataframe_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        validate_dataframe(pd.DataFrame())


def test_validate_column_rejects_non_string_name(complete_df):
    with pytest.raises(TypeError):
        validate_column(complete_df, 123)


def test_validate_column_rejects_unknown_column(complete_df):
    with pytest.raises(ValueError):
        validate_column(complete_df, "does_not_exist")


def test_validate_missing_col_rejects_complete_column(complete_df):
    with pytest.raises(ValueError):
        validate_missing_col(complete_df, "A")


def test_validate_missing_col_rejects_all_missing_column():
    df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
    with pytest.raises(ValueError):
        validate_missing_col(df, "x")

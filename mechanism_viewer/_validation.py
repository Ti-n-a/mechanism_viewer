"""Shared validation helpers used across mechanism_viewer modules."""

import pandas as pd


def validate_dataframe(
    df: pd.DataFrame
    ) -> None:
    """
    Validate the dataset given as input.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be validated for later visualization.
   
    Returns
    ------- 
    This function does not return anything.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input given is not a pd.DataFrame")
    if df.empty:
        raise ValueError("The function cannot run with an empty pd.DataFrame as input.")
    return


def validate_column(
    df: pd.DataFrame,
    column_name: str
    ) -> None:
    """
    Validate the column given as input.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset that contains the column.
    column_name : str
        The column to be validated for later visualization.
   
    Returns
    ------- 
    This function does not return anything.
    """
    if not isinstance(column_name, str):
        raise TypeError(f"The given column {column_name} must be in a string format.")
    if column_name not in df.columns:
        raise ValueError(f"Could not find {column_name} inside the given dataframe.")
    return


def validate_missing_col(
    df: pd.DataFrame,
    missing_col: str
    ) -> None:
    """
    Validate the missing column given as input.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset that contains the missing column.
    missing_col : str
        The missing column to be validated for later visualization.
   
    Returns
    ------- 
    This function does not return anything.
    """
    validate_column(df, missing_col)
    
    if df[missing_col].notna().all():
        raise ValueError(f"The missing column given, {missing_col}, does not contain any missing value.")
    return
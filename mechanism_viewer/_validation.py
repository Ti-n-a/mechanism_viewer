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


"""This file includes correlation heatmaps that use missingness of missing columns.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ._validation import validate_dataframe


__all__ = [
    "missing_columns_correlation",
    "complete_and_missing_columns_correlation",
]


def _validate_missing_list(
    col_list: list[str]
    ) -> None:
    """
    Validate the list given.

    Parameters
    ----------
    col_list : list[str]
        The list with column names that must be validated
   
    Returns
    ------- 
    This function does not return anything.
    """
    if not col_list:
        raise ValueError("The function cannot run with a pd.DataFrame that has no column with missing values.")
    return


def _validate_complete_list(
    col_list: list[str]
    ) -> None:
    """
    Validate the list given.

    Parameters
    ----------
    col_list : list[str]
        The list with column names that must be validated
   
    Returns
    ------- 
    This function does not return anything.
    """
    if not col_list:
        raise ValueError("The function cannot run with a pd.DataFrame that has no complete column.")
    return


def missing_columns_correlation(
    df: pd.DataFrame,
    display_plot: bool = False
    ) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots a correlation heatmap showing whether there are missing columns with missing
    values in similar rows. Therefore, it only works with columns with missing data.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to plot the missingness correlation heatmap
    display_plot : bool, default = False
        If True, displays figure with ``plt.show()``

    Returns
    -------
    tuple
        (fig_miss_corr, ax_miss_corr) representing the plot available for display.
    """
    validate_dataframe(df)

    data = df.copy()

    columns_with_na = data.columns[data.isna().any()].tolist()
    
    _validate_missing_list(columns_with_na)

    new_df = data[columns_with_na].corr()

    fig_miss_corr, ax_miss_corr = plt.subplots(figsize=(6, 5))

    sns.heatmap(
    new_df,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Missingness Correlation"},
    ax=ax_miss_corr
    )

    ax_miss_corr.set_title("Missingness Correlation Heatmap")
    fig_miss_corr.tight_layout()
    
    if display_plot:
        plt.show()
    else:
        plt.close(fig_miss_corr)
    
    return fig_miss_corr, ax_miss_corr


def complete_and_missing_columns_correlation(
    df: pd.DataFrame,
    display_plot: bool = False
    ) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots a correlation heatmap showing whether there are missing columns with missing
    values in similar rows. Besides, it shows whether the complete columns' values will
    correlate with the missing rows in the missing columns. Therefore, it works with both
    complete columns and missing columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to plot the correlation heatmap
    display_plot : bool, default = False
        If True, displays figure with ``plt.show()``
   
    Returns
    -------
    tuple
        (fig_comp_corr, ax_comp_corr) representing the plot available for display.
    """
    validate_dataframe(df)

    columns_without_na = df.columns[df.notna().all()].tolist()
    columns_with_na = df.columns[df.isna().any()].tolist()

    _validate_complete_list(columns_without_na)
    _validate_missing_list(columns_with_na)

    missing_dataset = df[columns_with_na].isna().astype(int)        # Transform missing columns into 0/1 integer indicators of missingness
    for column in columns_without_na:                               # Maintain values of complete columns
        missing_dataset[column] = df[column]

    corr = missing_dataset.corr()           

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)             # Mask the upper triangle

    fig_comp_corr, ax_comp_corr = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax_comp_corr)
    ax_comp_corr.set_title("Correlation Heatmap")
    fig_comp_corr.tight_layout()
    
    if display_plot:
        plt.show()
    else:
        plt.close(fig_comp_corr)
    
    return fig_comp_corr, ax_comp_corr
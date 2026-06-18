"""This file includes correlation heatmaps that use missingness of missing columns.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ._validation import validate_dataframe, validate_missing_col


__all__ = [
    "missingness_misscol_corr",
    "value_misscol_corr",
    "complete_and_misscol_corr",
    "misscol_vs_all_corr",
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


def missingness_misscol_corr(
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
        (fig_mssng_corr, ax_mssng_corr) representing the plot available for display.
    """
    validate_dataframe(df)

    data = df.copy()

    columns_with_na = data.columns[data.isna().any()].tolist()
    
    _validate_missing_list(columns_with_na)

    missing_dataset = df[columns_with_na].isna().astype(int)        # Transform missing columns into 0/1 integer indicators of missingness

    new_df = missing_dataset.corr()

    _ , n_cols = new_df.shape

    fig_mssng_corr, ax_mssng_corr = plt.subplots(figsize=(max(4,n_cols), max(4,n_cols)))

    sns.heatmap(new_df, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"label": "Missingness correlation"},
                ax=ax_mssng_corr)

    ax_mssng_corr.set_title("Missingness correlation between missing columns", pad=40)
    fig_mssng_corr.tight_layout()
    
    if display_plot:
        plt.show()
    else:
        plt.close(fig_mssng_corr)
    
    return fig_mssng_corr, ax_mssng_corr


def value_misscol_corr(
    df: pd.DataFrame,
    display_plot: bool = False
    ) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots a correlation heatmap for the values of the missing columns.
    It is focused on only plotting the correlation for columns with missing data.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to plot the value correlation heatmap of missing
        columns
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

    _ , n_cols = new_df.shape

    fig_miss_corr, ax_miss_corr = plt.subplots(figsize=(max(4,n_cols), max(4,n_cols)))

    sns.heatmap(new_df, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"label": "Value correlation"},
                ax=ax_miss_corr)

    ax_miss_corr.set_title("Value correlation between missing columns", pad=40)
    fig_miss_corr.tight_layout()
    
    if display_plot:
        plt.show()
    else:
        plt.close(fig_miss_corr)
    
    return fig_miss_corr, ax_miss_corr


def complete_and_misscol_corr(
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

    n_cols = len(columns_without_na) + len(columns_with_na)

    _validate_complete_list(columns_without_na)
    _validate_missing_list(columns_with_na)

    missing_dataset = df[columns_with_na].isna().astype(int)        # Transform missing columns into 0/1 integer indicators of missingness
    for column in columns_without_na:                               # Maintain values of complete columns
        missing_dataset[column] = df[column]

    corr = missing_dataset.corr()           

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)             # Mask the upper triangle

    fig_comp_corr, ax_comp_corr = plt.subplots(figsize=( max(4,n_cols),  max(4,n_cols)))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax_comp_corr)
    ax_comp_corr.set_title("Correlation heatmap", pad=20)
    fig_comp_corr.tight_layout()
    
    if display_plot:
        plt.show()
    else:
        plt.close(fig_comp_corr)
    
    return fig_comp_corr, ax_comp_corr


def misscol_vs_all_corr(
    df: pd.DataFrame,
    missing_col: str,
    display_plot: bool = False
    ) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots a correlation heatmap showing whether the missing column has its missingness
    correlated with values from the other columns, no matter if they are complete or
    missing.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to plot the correlation heatmap
    missing_col : str
        The missing column name that will be used for correlation
    display_plot : bool, default = False
        If True, displays figure with ``plt.show()``
   
    Returns
    -------
    tuple
        (fig_vs_corr, ax_vs_corr) representing the plot available for display.
    """
    validate_dataframe(df)
    validate_missing_col(df, missing_col)

    missing_dataset = df.copy()
    missing_dataset[missing_col] = df[missing_col].isna().astype(int)

    corr = missing_dataset.corrwith(missing_dataset[missing_col]).to_frame(name=missing_col)   

    n_rows = len(corr)     

    fig_vs_corr, ax_vs_corr = plt.subplots(figsize=(3, max(4, n_rows * 0.6)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax_vs_corr)
    ax_vs_corr.set_title(f"Correlation Heatmap of {missing_col}", pad=20)
    fig_vs_corr.tight_layout()
    
    if display_plot:
        plt.show()
    else:
        plt.close(fig_vs_corr)
    
    return fig_vs_corr, ax_vs_corr
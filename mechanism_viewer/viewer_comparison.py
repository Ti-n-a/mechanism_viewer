"""The visualization tools described below can be used to compare observed columns with
the missingness of a column. For best visualization, the first two should be used with
an observed column with continuous data type.
"""

import pandas as pd
from pandas.api.types import is_numeric_dtype
import seaborn as sns
import matplotlib.pyplot as plt

from ._validation import validate_dataframe, validate_column, validate_missing_col


__all__ = [
    "scatter_missingness_comparison",
    "scatter_missingness_comparison_line",
    "boxplot_comparison",
]


def validate_numeric_col(
    df: pd.DataFrame,
    numeric_col: str
    ) -> None:
    """
    Validate the numeric column given as input.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset that contains the numeric column.
    numeric_col : str
        The numeric column to be validated for later visualization.
   
    Returns
    ------- 
    This function does not return anything.
    """
    validate_column(df, numeric_col)
    
    if not is_numeric_dtype(df[numeric_col]):
        raise ValueError(f"The column given, {numeric_col}, is not of numeric type.")
    if df[numeric_col].isna().all():       
        raise ValueError(f"{numeric_col} column is full of missing values.")
    return



def scatter_missingness_comparison(
    df: pd.DataFrame,
    column_name: str,
    missing_col: str,
    display_plot: bool = False
    ) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots a scatterplot for the data points (column_name, is_not_missing(missing_col))

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to plot.
    column_name : str
        The observed column name that will be used for comparison
    missing_col : str
        The missing column name that will be used for comparison
    display_plot : bool, default = False
        If True, displays figure with ``plt.show()``

    Returns
    -------
    tuple
        (fig_comp, ax_comp) representing the plot available for display.
    """
    validate_dataframe(df)
    validate_column(df, column_name)
    validate_missing_col(df, missing_col)

    missing_col_missingness = df[missing_col].notna().astype(int)

    fig_comp, ax_comp = plt.subplots(figsize=(10, 2))

    sns.scatterplot(x=df[column_name], y=missing_col_missingness, alpha=0.5, ax=ax_comp)

    ax_comp.set_title(f"Plotting of {column_name} using missingness of {missing_col}")
    ax_comp.set_xlabel(column_name)
    ax_comp.set_ylabel(f"Missingness of {missing_col}")

    ax_comp.set_yticks(range(0, 2), ["Missing","Not missing"] )

    if display_plot:
        plt.show()
    else:
        plt.close(fig_comp)
    
    return fig_comp, ax_comp


def scatter_missingness_comparison_line(
    df: pd.DataFrame,
    column_name: str,
    missing_col: str,
    display_plot: bool = False
    ) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots a scatterplot for the data points (column_name, is_not_missing(missing_col)) in
    a single line.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to plot.
    column_name : str
        The observed column name that will be used for comparison
    missing_col : str
        The missing column name that will be used for comparison
    display_plot : bool, default = False
        If True, displays figure with ``plt.show()``

    Returns
    -------
    tuple
        (fig_comp_line, ax_comp_line) representing the plot available for display.
    """
    validate_dataframe(df)
    validate_column(df, column_name)
    validate_missing_col(df, missing_col)

    missing_col_missingness = df[missing_col].notna().astype(int)
    missing_labels = missing_col_missingness.map({0: "Missing", 1: "Not missing"})

    fig_comp_line, ax_comp_line = plt.subplots(figsize=(10, 2))

    sns.scatterplot(x=df[column_name], y=[1] * len(df), hue=missing_labels, 
                    hue_order=["Missing", "Not missing"], palette={"Missing": "red", "Not missing": "blue"},
                    alpha=0.5, ax=ax_comp_line)

    ax_comp_line.set_title(f"Plotting of {column_name} using missingness of {missing_col}")
    ax_comp_line.set_xlabel(column_name)
    ax_comp_line.set_ylabel("")

    ax_comp_line.set_yticks([])

    if display_plot:
        plt.show()
    else:
        plt.close(fig_comp_line)
    
    return fig_comp_line, ax_comp_line


def boxplot_comparison(
    df: pd.DataFrame,
    column_name: str,
    missing_col: str,
    display_plot: bool = False
    ) -> tuple[plt.Figure, plt.Axes]:
    """
    Shows three boxplots: the boxplot of column_name when missing_col has values, the
    boxplot of column_name when the values of missing_col are missing, and the general
    boxplot of column_name no matter the missingness of missing_col.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to plot.
    column_name : str
        The observed column name that will be used for comparison
    missing_col : str
        The missing column name that will be used for comparison
    display_plot : bool, default = False
        If True, displays figure with ``plt.show()``

    Returns
    -------
    tuple
        (fig_boxplot, ax_boxplot) representing the plot available for display.
    """
    validate_dataframe(df)
    validate_numeric_col(df, column_name)
    validate_missing_col(df, missing_col)

    not_missing_series = df.loc[df[missing_col].notna(), column_name]
    missing_series = df.loc[df[missing_col].isna(), column_name]
    all_series = df[column_name]

    not_missing_str = f"Values of {missing_col} are not missing"
    missing_str = f"Values of {missing_col} are missing"
    all_str = f"All values of {column_name}"

    plot_df = pd.concat([pd.DataFrame({"group": not_missing_str, "value": not_missing_series}),
                         pd.DataFrame({"group": missing_str, "value": missing_series}),
                         pd.DataFrame({"group": all_str, "value": all_series})
                        ], ignore_index=True)

    group_order = [not_missing_str, missing_str, all_str]

    fig_boxplot, ax_boxplot = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="group", y="value", data=plot_df, order=group_order, ax=ax_boxplot)
    ax_boxplot.set_title(f"Comparison of values of {column_name} with missingness in {missing_col}")
    ax_boxplot.set_xlabel('')
    ax_boxplot.set_ylabel(f"Values of {column_name}")
    
    if display_plot:
        plt.show()
    else:
        plt.close(fig_boxplot)
    
    return fig_boxplot, ax_boxplot
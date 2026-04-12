"""The functions below plot simple visualization content, as a way to aid in the profiling 
of the dataset in the early stages of identifying the missing data mechanism.
"""

import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns

from ._validation import validate_dataframe, validate_missing_col


__all__ = [
    "plot_missing_rate",
    "build_distribution_of_missingness",
]


def plot_missing_rate(
    df: pd.DataFrame,
    display_plot: bool = False
    ) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots the missing rate of each column.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used
    display_plot : bool, default = False
        If True, displays figure with ``plt.show()``
   
    Returns
    -------
    tuple
        (fig_missing_rate, ax_missing_rate) representing the plot available for display.
    """
    validate_dataframe(df)

    missing_rate_dataset = df.isna().mean().round(2)

    df_missing_rate = missing_rate_dataset.to_frame().T             # Transform DataSeries into Dataframe and rotate it

    fig_missing_rate, ax_missing_rate = plt.subplots(figsize=(len(df_missing_rate.columns) * 1.5, 2.25))  # To create square boxes
    sns.heatmap(df_missing_rate, annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1, cbar=False, linewidths=0.75, ax=ax_missing_rate)
    ax_missing_rate.set_yticks([], [])
    ax_missing_rate.set_xlabel("Columns", labelpad=15)
    ax_missing_rate.set_title("Missing rate per column")
    fig_missing_rate.tight_layout()
    
    if display_plot:
        plt.show()
    else:
        plt.close(fig_missing_rate)
    
    return fig_missing_rate, ax_missing_rate


def build_distribution_of_missingness(
    df: pd.DataFrame,
    missing_col: str,
    display_plot: bool = False
    ) -> tuple[plt.Figure | plt.Axes, ...]:
    """
    Creates multiple plots showing the distribution of each column based on the missingness of missing_col.
    If the existence of red bars appear throughout the plot, then missing_col does not see any relationship
    with the column at hand, in regards to the way its missing values appear. In the other hand, if most
    red bars appear in a zone of the plot, then it is likely that missing_col has a MAR missing mechanism, and
    depends on the column at hand.

    If a column has non-numeric data, then the distribution plot is created using ``countplot`` instead of ``histplot``

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe that will be used for plotting
    missing_col : str
        The column with missing values, that will be used to group the other columns into 2 groups:
        rows that have NaN in missing_col, and rows that have a value in missing_col.
    display_plot : bool, default = False
        If True, displays figure with ``plt.show()``
        
    Returns
    -------
    tuple
        (fig1, ax1, fig2, ax2, ..., figN, axN) representing all plots available for display,
        Order follows ``df.columns``, skipping ``missing_col``.
    """
    validate_dataframe(df)
    validate_missing_col(df, missing_col)

    data = df.copy()

    missing_label = f"Missing value on {missing_col}"
    observed_label = f"Observed value on {missing_col}"
    palette = {missing_label: "#C92C3E", observed_label: "#029911"}

    missingness_col = "Missingness"

    # Make sure the Missingness column has a unique name that no column in df has by adding a "_"
    while missingness_col in data.columns:
        missingness_col += "_"

    # Map True/False missingness into 2 strings to appear in the legend
    data[missingness_col] = data[missing_col].isna().map({True: missing_label, False: observed_label})
    
    plots = []  # To store every pair of fig and ax

    for col in df.columns:
        if col != missing_col:
            fig_distribution, ax_distribution = plt.subplots(figsize=(8, 6))
            if is_numeric_dtype(data[col]):
                sns.histplot(
                    data=data,
                    x=col,
                    hue=missingness_col,
                    multiple="dodge",
                    edgecolor="grey",
                    palette=palette,
                    ax=ax_distribution,
                )
            else:
                sns.countplot(
                    data=data,
                    x=col,
                    hue=missingness_col,
                    edgecolor="grey",
                    palette=palette,
                    ax=ax_distribution,
                )

            ax_distribution.set_title(f"Distribution of {col} based on the missingness of {missing_col}")
            ax_distribution.set_xlabel(f"{col}")
            ax_distribution.set_ylabel("Count")

            plots.extend([fig_distribution, ax_distribution])
            
            if display_plot:
                plt.show()
            else:
                plt.close(fig_distribution)
    
    return tuple(plots)  # Flat tuple (fig1, ax1, fig2, ax2,...)
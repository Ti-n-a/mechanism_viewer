"""This module provides a dendrogram for grouping columns with similar missingness patterns."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from ._validation import validate_dataframe


__all__ = [
    "missingness_dendrogram",
]

def validate_linkage_method(
    linkage_method: Literal[
        "single",
        "complete",
        "average",
        "weighted",
    ]
    ) -> None:
    """
    Validate the linkage method given as input.

    Parameters
    ----------
    linkage_method : {"single", "complete", "average", "weighted"}
        The linkage method used to calculate the distance between clusters.
   
    Returns
    ------- 
    This function does not return anything.
    """
    allowed_methods = {"single", "complete", "average", "weighted"}

    if linkage_method not in allowed_methods:
        raise ValueError(f"Unknown chosen linkage method: {linkage_method}. "
                         f"Expected one of {allowed_methods}.")
    return


def missingness_dendrogram(
    df: pd.DataFrame,
    linkage_method: Literal[
        "single",
        "complete",
        "average",
        "weighted",
    ] = "average",
    display_plot: bool = False,
    ) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a dendrogram that groups columns with similar missingness patterns.

    Each column is converted into a binary missingness indicator, where 1
    represents a missing value and 0 represents an observed value. The
    correlation between these indicators is used to calculate the distance
    between columns. Columns that tend to contain missing values in the same
    rows are therefore joined closer together in the dendrogram.

    Complete columns and columns containing only missing values are excluded
    because their missingness does not vary.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used for plotting.
    linkage_method : {"single", "complete", "average", "weighted"},
        default = "average"
        The linkage method used to calculate the distance between clusters.
    display_plot : bool, default = False
        If True, displays the figure with ``plt.show()``.

    Returns
    -------
    tuple
        (fig_dendrogram, ax_dendrogram) representing the plot available for
        display.
    """
    validate_dataframe(df)
    validate_linkage_method(linkage_method)

    missing_matrix = df.isna().astype(int)

    missing_columns = [column for column in missing_matrix.columns if missing_matrix[column].nunique() > 1]

    if len(missing_columns) < 2:
        raise ValueError("At least two columns containing both missing and observed values are required to create the dendrogram.")

    missing_matrix = missing_matrix[missing_columns]

    correlation = missing_matrix.corr()
    distance = 1.0 - correlation
    np.fill_diagonal(distance.values, 0.0)

    condensed_distance = squareform(distance.values, checks=False)
    linkage_matrix = linkage(condensed_distance, method=linkage_method)

    fig_dendrogram, ax_dendrogram = plt.subplots(figsize=(max(8, len(missing_columns) * 0.55), 6))

    dendrogram(linkage_matrix, labels=missing_columns, leaf_rotation=45, leaf_font_size=10, ax=ax_dendrogram)

    ax_dendrogram.set_title("Dendrogram of missingness similarity")
    ax_dendrogram.set_xlabel("Columns")
    ax_dendrogram.set_ylabel("Distance")
    fig_dendrogram.tight_layout()

    if display_plot:
        plt.show()
    else:
        plt.close(fig_dendrogram)

    return fig_dendrogram, ax_dendrogram

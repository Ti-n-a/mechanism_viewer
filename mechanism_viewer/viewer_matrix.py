"""The methods plot matrices with multiple missing and complete columns, with a focus on the missing
rows and the missing rate.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ._validation import validate_dataframe, validate_column


__all__ = [
    "visualize_column_dependencies",
    "missing_rate_matrix",
]


def _validate_input(
    df: pd.DataFrame
    ) -> None:
    """
    Validate the dataset given as input.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be validated for later visualization
   
    Returns
    ------- 
    This function does not return anything.
    """
    validate_dataframe(df)
    
    if not df.isna().values.any():
         raise ValueError(f"The dataset has no missing values.")
    return


def visualize_column_dependencies(
    df: pd.DataFrame,
    sort_by_complete: bool = False,
    display_plot: bool = False,
    ) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots a matrix of all missing rows. It enables to check which rows depend on which values
    of complete columns. Besides, rows with the same attributes missing can as well be viewed.

    Columns with missing rows that have mostly high/low values for complete columns, indicate
    a missingness dependency on that complete column, giving indication for a MAR mechanism.

    Complete columns will be displayed in green hues (light green = low value, dark green = high
    value), while missing columns will have their missingness displayed in red (white = not
    missing, red = missing).

    Perfect for continuous or numeric complete columns, and for multivariate missing columns.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to plot.
    sort_by_complete : bool, default = False
        To sort the complete columns for better visualization
    display_plot : bool, default = False
        If True, displays figure with ``plt.show()``

    Returns
    -------
    tuple
        (fig_dep_matrix, ax_dep_matrix) representing the plot available for display.
    """
    validate_dataframe(df)

    data = df.copy()

    fig_dep_matrix, ax_dep_matrix = plt.subplots(figsize=(8, 12))

    cmap = plt.cm.Greens

    if sort_by_complete:
        columns_complete = data.columns[data.notna().all()]
        if len(columns_complete) > 0:
            data = data.sort_values(by=columns_complete.tolist(), ascending=False)

    mask = data.isna().any(axis=1)    # Missingness mask

    if not mask.any():
        raise ValueError("The function cannot run with a pd.DataFrame that has no column with missing values.")

    filtered_data = data[mask].copy()     #Focusing only in rows that have at least one missing attribute

    for i, column in enumerate(data.columns):
        if data[column].notna().all():
            cmap = plt.cm.Greens
            # Scale complete columns to [0, 1] so color intensity is comparable.
            col_values = data[column]
            if pd.api.types.is_numeric_dtype(col_values):
                col_min = col_values.min()
                col_max = col_values.max()
                if col_max > col_min:
                    scaled_series = (col_values - col_min) / (col_max - col_min)
                else:
                    scaled_series = pd.Series(0.0, index=col_values.index)
            else:
                codes, _ = pd.factorize(col_values, sort=True)
                codes_series = pd.Series(codes, index=col_values.index)
                max_code = codes_series.max()
                if max_code > 0:
                    scaled_series = codes_series / max_code
                else:
                    scaled_series = pd.Series(0.0, index=col_values.index)

            filtered_data[column] = scaled_series.loc[filtered_data.index].astype(float)
        else:
            cmap = plt.cm.OrRd
            filtered_data[column] = filtered_data[column].isna().astype(int)

        ax_dep_matrix.imshow(
        filtered_data[[column]].values,
        cmap=cmap,
        aspect='auto',
        vmin=0,
        vmax=1,
        interpolation='nearest',
        extent=[i, i+1, 0, len(filtered_data)]
        )


    ax_dep_matrix.set_xticks(range(len(filtered_data.columns)))
    ax_dep_matrix.set_xticklabels(filtered_data.columns)
    ax_dep_matrix.set_yticks([])
    ax_dep_matrix.set_ylabel('Rows')
    ax_dep_matrix.set_xlabel('Columns')
    ax_dep_matrix.set_title('Observable Column Dependency')

    sm = ScalarMappable(
        cmap=plt.cm.Greens, 
        norm=Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])

    fig_dep_matrix.colorbar(sm, ax=ax_dep_matrix, label='Value')

    fig_dep_matrix.tight_layout()

    if display_plot:
        plt.show()
    else:
        plt.close(fig_dep_matrix)

    return fig_dep_matrix, ax_dep_matrix


def missing_rate_matrix(
    df: pd.DataFrame,
    column_name: str,
    sort_by_column: bool = True,
    display_plot: bool = False,
    ) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots a matrix of missing rate. It uses column_name to group the values of the other columns,
    and checks for each group the missing rate of the column. The darker the cells are, the higher
    the missing rate will be of a column for that particular value of column_name. If the color
    of a column is similar all over column_name, then the column is likely to not depend on
    column_name for its missing values. In contrast, if there is a darker region, the column is
    likely having MAR mechanism.

    Its best to use this plot when column_name is not continuous or when it does not contain multiple
    values.  Perfect for multivariate missing columns.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to plot.
    column_name : str
        The observed column name that will be used for comparison
    sort_by_column : bool, default = True
        To sort the column_name values in ascending order for better visualization
    display_plot : bool, default = False
        If True, displays figure with ``plt.show()``

    Returns
    -------
    tuple
        (fig_miss_matrix, ax_miss_matrix) representing the plot available for display.
    """
    _validate_input(df)
    validate_column(df, column_name)

    data = df.copy()

    norm_scaler = MinMaxScaler()

    if sort_by_column:
        data = data.sort_values(by=column_name)

    # Add Complete and Incomplete so that normalization fits well between 0 and 1, without columns exagerating values on their comparisons 
    data["Complete"] = 0
    data["Incomplete"] = np.nan

    missingness_per_value_df= data.groupby(column_name).agg(lambda x: x.isna().sum())

    df_scaled = norm_scaler.fit_transform(missingness_per_value_df.T)         # Normalize the data (transform table to rotate it on horizontal axis)

    # To remove complete and incomplete columns
    df_scaled = df_scaled[:-2]
    data.drop(['Complete', 'Incomplete'], axis=1, inplace=True)

    fig_miss_matrix, ax_miss_matrix = plt.subplots(figsize=(10, 6))
    im = ax_miss_matrix.imshow(df_scaled, aspect="auto", cmap="binary", interpolation="nearest", vmin=0, vmax=1)

    # Print the column names we are comparing against column_name on y axis
    only_missing_cols_df = data.drop(column_name, axis=1, inplace=False)
    ax_miss_matrix.set_yticks(range(len(only_missing_cols_df.columns)), only_missing_cols_df.columns)

    # Print unique values of the column_name on x axis
    unique_values_arr = data[column_name].unique()
    ax_miss_matrix.set_xticks(range(len(unique_values_arr)), unique_values_arr)
    ax_miss_matrix.set_xlabel(f"Unique {column_name} Values")
    
    ax_miss_matrix.set_title("Missing Rate Matrix (The darker a cell is, the higher the missing rate on that column for a specific observed value)")
    fig_miss_matrix.colorbar(im, ax=ax_miss_matrix, label="Missing Rate")
    fig_miss_matrix.tight_layout()
    
    if display_plot:
        plt.show()
    else:
        plt.close(fig_miss_matrix)
    
    return fig_miss_matrix, ax_miss_matrix
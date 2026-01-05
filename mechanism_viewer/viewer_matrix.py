import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

"""
The methods plot matrices with multiple missing and complete columns, with a focus on the missing
rows and the missing rate.
"""


def visualize_column_dependencies(df: pd.DataFrame, sort_complete: bool = False):
    """
    Plots a matrix of all missing rows. It enables to check which rows depend on which values
    of complete columns. Besides, rows with the same attributes missing can as well be viewed.

    Columns with missing rows that have mostly high/low values for complete columns, indicate
    a missingness dependency on that complete column, givind indication for a MAR mechanism.

    Complete colums will be displayed in green hues (light green = low value, dark green = high
    value), while missing columns will have their missingness displayed in red (white = not
    missing, red = missing).

    Perfect for continuous or numeric complete columns, and for multivariate missing columns.

    Parameters:
    df (pd.DataFrame): The dataset to be used to plot.
    sort_complete (bool): To sort the complete columns for better visualization

    Returns:
    This function does not return anything.
    """
    _, ax = plt.subplots(figsize=(8, 12))

    cmap = plt.cm.Greens

    if sort_complete:
        columns_complete = df.columns[df.notna().all()]
        df = df.sort_values(by=columns_complete.tolist(), ascending=False)

    mask = df.isna().any(axis=1)    # Missingness mask

    filtered_data = df[mask].copy()     #Focusing only in rows that have at least one missing attribute

    for i, column in enumerate(df.columns):
        if df[column].notna().all():
            cmap = plt.cm.Greens
        else:
            cmap = plt.cm.OrRd
            filtered_data[column] = filtered_data[column].isna().astype(int)

        ax.imshow(
        filtered_data[[column]].values,
        cmap=cmap,
        aspect='auto',
        vmin=0,
        vmax=1,
        interpolation='nearest',
        extent=[i, i+1, 0, len(filtered_data)]
        )


    ax.set_xticks(range(len(filtered_data.columns)))
    ax.set_xticklabels(filtered_data.columns)
    ax.set_yticks([])
    ax.set_ylabel('Rows')
    ax.set_xlabel('Columns')
    ax.set_title('Observable Column Dependency')

    sm = ScalarMappable(
        cmap=plt.cm.Greens, 
        norm=Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])

    plt.colorbar(sm, ax=ax, label='Value')

    plt.tight_layout()
    plt.show()


def missing_rate_matrix(df: pd.DataFrame, column_name_x: str, sort_column_x: bool = True):
    """
    Plots a matrix of missing rate. It uses column_name_x to group the values of the other columns,
    and checks for each group the missing rate of the column. The darker the cells are, the higher
    the missing rate will be of a column for that particular value of column_name_x. If the color
    of a column is similar all over column_name_x, then the column is likely to not depend on
    column_name_x for its missing values. In contrast, if there is a darker region, the column is
    likely having MAR mechanism.

    Its best to use this plot when column_name_x is not continuos or when it does not contain multiple
    values.  Perfect for multivariate missing columns.

    Parameters:
    df (pd.DataFrame): The dataset to be used to plot.
    column_name_x (str): The observed column name that will be used for comparison
    sort_column_x (bool): To sort the column_name_x values in ascending order for better visualization

    Returns:
    This function does not return anything.
    """
    norm_scaler = MinMaxScaler()

    if sort_column_x:
        df = df.sort_values(by=column_name_x)

    #Add Complete and Imcomplete so that normalization fits well between 0 and 1, without columns exagerating values on their comparisons 
    df["Complete"] = 0
    df["Incomplete"] = np.nan

    missingness_per_value_df= df.groupby(column_name_x).agg(lambda x: x.isna().sum())

    df_scaled = norm_scaler.fit_transform(missingness_per_value_df.T)         # Normalize the data (transform table to rotate it on horizontal axis)

    #To remove complete and imcomplete columns
    df_scaled = df_scaled[:-2]
    df.drop(['Complete', 'Incomplete'], axis=1, inplace=True)

    plt.figure(figsize=(10,6))    
    plt.imshow(df_scaled, aspect="auto", cmap="binary", interpolation="nearest", vmin=0, vmax=1)

    # Print the column names we are compraing against column_name_x on y axis
    only_missing_cols_df = df.drop(column_name_x, axis=1, inplace=False)
    plt.yticks(range(len(only_missing_cols_df.columns)), only_missing_cols_df.columns)

    # Print column_name_x unique values on x axis
    unique_ages_arr = df[column_name_x].unique()
    plt.xticks(range(len(unique_ages_arr)), unique_ages_arr)
    plt.xlabel(f"Unique {column_name_x} Values")
    
    plt.title("Missing Rate Matrix (The darker a cell is, the higher the missing rate on that column for a specific observed value)")
    plt.colorbar(label="Missing Rate")
    plt.tight_layout()
    plt.show()
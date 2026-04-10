"""The visualization tool on this module focuses on plotting a similar UpSet plot for rows with the same
attributes missing.
"""

import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    "rows_with_similar_missing",
]


def _validate_input(
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


def _get_combination_columns(
    tuple_combination: tuple[bool,...],
    column_name_array: list[str]
    ) -> str:
    """
    From the existing tuples, knowing that each tuple indicates the columns that have
    same rows with missing values, creates a string with every column name.

    Parameters
    ----------
    tuple_combination : tuple[bool,...] 
        A tuple symbolizing the combination of columns with the same rows missing
    column_name_array : list[str]
        Every column name, for easy retrieval of the name
   
    Returns
    -------
    A string with all the names of the columns with same missing rows
    """

    true_columns = [column_name_array[i] for i, value in enumerate(tuple_combination) if value]    # Create a list of column names where the combination is True

    if not true_columns:
        return "No column values missing"   # If no True values, which are the rows with all attributes complete
    return ', '.join(true_columns)          # Join column names with commas if multiple columns are True


def rows_with_similar_missing(
    df: pd.DataFrame,
    display_plot: bool = False,
    ) -> tuple[plt.Figure, plt.Axes]:
    """
    Creates a plot similar to UpSet plot. It retrieves which rows have the same attributes missing, 
    by indicating the count of those rows, and which are the missing attributes.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to plot.
    display_plot : bool, default = False
        If True, displays figure with ``plt.show()``

    Returns
    -------
    tuple
        (fig_similar_miss, ax_similar_miss) representing the plot available for display.
    """
    _validate_input(df)
    
    isna_df = df.isna()
    missing_combinations = isna_df.apply(lambda row: tuple(row), axis=1)        # Convert rows into tuples of missing values (True = missing, False = not missing). E.g.: (True,False,False,True)

    combination_counts = missing_combinations.value_counts().reset_index()      # Count how many rows have the same combination of missing column values

    combination_counts.columns = ['Combination', 'Count']                       # Rename dataframe for visualization

    combinations = combination_counts['Combination'].apply(lambda tuple_combination: _get_combination_columns(tuple_combination, list(df.columns)))

    fig_similar_miss, ax_similar_miss = plt.subplots(figsize=(10, 5))
    ax_similar_miss.bar(combinations, combination_counts["Count"], color='gray')

    ax_similar_miss.set_title('Number of rows with same missing patterns')
    ax_similar_miss.set_xlabel('Column combination (missing values at same rows)')
    ax_similar_miss.set_ylabel('Number of rows')
    ax_similar_miss.tick_params(axis='x', rotation=90)     # Rotating the labels on x axis so they fit better

    # To add count text on top of each bar
    for i, count in enumerate(combination_counts["Count"]):
        ax_similar_miss.text(i, count + 0.4, str(count), ha='center')

    highest_count = max(combination_counts["Count"]) + 5
    ax_similar_miss.set_yticks(range(0, highest_count+1, int((highest_count)/5) ))     # Set y axis ticks as integers (previously floats)

    # To hide the ugly border around the graph
    ax_similar_miss.spines['top'].set_visible(False)     # Top border
    ax_similar_miss.spines['right'].set_visible(False)   # Right border

    if display_plot:
        plt.show()
    else:
        plt.close(fig_similar_miss)

    return fig_similar_miss, ax_similar_miss
"""The visualization tool on this module focuses on plotting a similar UpSet plot for rows with the same
attributes missing.
"""

import pandas as pd
import matplotlib.pyplot as plt


def _get_combination_columns(tuple_combination, column_name_array: list[str]):
    """
    From the existing tupples, knowing that each tupple indicates the columns that have
    same rows with missing values, creates a string with every column name.

    Parameters
    ----------
    tuple_combination : tuple[bool] 
        A tupple simbolizing the combination of columns with the same rows misssing
    column_name_array : list[str]
        Every column name, for easy retrival of the name
   
    Returns
    -------
    A string with all the names of the columns with same missing rows
    """

    true_columns = [column_name_array[i] for i, value in enumerate(tuple_combination) if value]    # Create a list of column names where the combination is True

    if not true_columns:
        return "No column values missing"   # If no True values, which are the rows with all attributes complete
    return ', '.join(true_columns)          # Join column names with commas if multiple columns are True


def rows_with_similar_missing(df: pd.DataFrame):
    """
    Creates a plot similar to UpSet plot. It retrives which rows have the same attributes missing, 
    by indicating the count of those rows, and which are the missing attributes.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to plot.

    Returns
    -------
    This function does not return anything.
    """

    isna_df = df.isna()
    missing_combinations = isna_df.apply(lambda row: tuple(row), axis=1)        # Convert rows into tuples of missing values (True = missing, False = not missing). E.g.: (True,False,False,True)

    combination_counts = missing_combinations.value_counts().reset_index()      # Count how many rows have the same combination of missing column values

    combination_counts.columns = ['Combination', 'Count']                       # Rename dataframe for visualization

    combinations = combination_counts['Combination'].apply(lambda tuple_combination: _get_combination_columns(tuple_combination, df.columns))

    plt.bar(combinations, combination_counts["Count"], color='gray')

    plt.title('Number of rows with same missing patterns')
    plt.xlabel('Column combination (missing values at same rows)')
    plt.ylabel('Number of rows')
    plt.xticks(rotation=45, ha='right')     # Rotating the labels on x axis so they can fit better 

    # To add count text on top of each bar
    for i, count in enumerate(combination_counts["Count"]):
        plt.text(i, count + 0.1, str(count), ha='center')

    highest_count = max(combination_counts["Count"]) + 5
    plt.yticks(range(0, highest_count+1, int((highest_count)/5) ))     # Set y axis ticks as integers (previously floats)

    # To hide the ugly border around the graph
    ax = plt.gca()
    ax.spines['top'].set_visible(False)     # Top border
    ax.spines['right'].set_visible(False)   # Right border

    plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
The visualization tools described below can be used to compare observed columns with
the missingness of a column. For best visualization, the first two should be used with
an observed column with continuos data type.
"""

def scatter_missingness_comparison(df: pd.DataFrame, column_name_x: str, column_name_y: str):
    """
    Plots a scatterplot for the data points (column_name_x, is_not_missing(column_name_y))

    Parameters:
    df (pd.DataFrame): The dataset to be used to plot.
    column_name_x (str): The observed column name that will be used for comparison
    column_name_y (str): The missing column name that will be used for comparison

    Returns:
    This function does not return anything.
    """
    missing_col = df[column_name_y].notna().astype(int)

    plt.figure(figsize=(10, 2))

    sns.scatterplot(x=df[column_name_x], y=missing_col, alpha=0.5)

    plt.title('Missing Values vs Not Missing Values')
    plt.xlabel(column_name_x)
    plt.ylabel(f"Missingness of {column_name_y}")

    plt.yticks(range(0, 2), ["missing","not missing"] )

    plt.show()


def scatter_missingness_comparison_line(df: pd.DataFrame, column_name_x: str, column_name_y: str):
    """
    Plots a scatterplot for the data points (column_name_x, is_not_missing(column_name_y)) in
    a single line.

    Parameters:
    df (pd.DataFrame): The dataset to be used to plot.
    column_name_x (str): The observed column name that will be used for comparison
    column_name_y (str): The missing column name that will be used for comparison

    Returns:
    This function does not return anything.
    """
    missing_col = df[column_name_y].notna().astype(int)

    plt.figure(figsize=(10, 2))

    sns.scatterplot(x=df[column_name_x], y=[1]*len(df), hue=missing_col, palette={0: 'blue', 1: 'red'}, alpha=0.5)

    plt.title('Missing Values vs Not Missing Values')
    plt.xlabel(column_name_x)
    plt.ylabel(f"Missingness of {column_name_y}")

    plt.yticks([])

    plt.show()


def boxplot_comparison(df: pd.DataFrame, column_name_x: str, column_name_y: str):
    """
    Shows three boxplots: the boxplot of column_name_x when column_name_y has values, the
    boxplot of column_name_x when the values of column_name_y are missing, and the general
    boxplot of column_name_x no matter the missingness of column_name_y.

    Parameters:
    df (pd.DataFrame): The dataset to be used to plot.
    column_name_x (str): The observed column name that will be used for comparison
    column_name_y (str): The missing column name that will be used for comparison

    Returns:
    This function does not return anything.
    """
    missingness = df[column_name_y].isna().astype(int)

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=missingness, y=column_name_x, data=df)
    sns.boxplot(y=column_name_x, data=df)
    plt.title(f"Comparison of values of {column_name_x} with Missingness in {column_name_y}")
    plt.xticks(range(3), [f"Values of {column_name_y} are not missing", f"Values of {column_name_y} are missing", f"All values of {column_name_x}"])
    plt.xlabel('')
    plt.ylabel(f"Values of {column_name_x}")
    plt.show()
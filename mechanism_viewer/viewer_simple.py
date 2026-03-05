"""The functions below plot simple visualization content, as a way to aid in the profiling 
of the dataset in the early stages of identifying the missing data mechanism.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_missing_rate(df: pd.DataFrame):
    """
    Plots the missing rate of each column.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used
   
    Returns
    -------
    This function does not return anything.
    """
    missing_rate_dataset = round(df.isna().astype(int).sum() / len(df), 2)

    df_missing_rate = missing_rate_dataset.to_frame().T             # Transform DataSeries into Dataframe and rotate it

    plt.figure(figsize=(len(df_missing_rate.columns) * 1.5, 2.25))  # To create square boxes
    sns.heatmap(df_missing_rate, annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1, cbar=False, linewidths=0.75)

    plt.yticks([], [])
    plt.xlabel("Columns", labelpad=15)
    plt.title("Missing rate per column")
    plt.tight_layout()
    plt.show()


def build_distribution_of_missingness(df: pd.DataFrame, missing_col: str):
    """
    Creates multiple plots showing the distribution of each column based on the missingness of missing_col.
    If the existence of red bars appear throughout the plot, then missing_col does not see any relationship
    with the column at hand, in regards to the way its missing values appear. In the other hand, if most
    red bars appear in a zone of the plot, then it is likely for missing_col have a MAR missing mechanism, and
    depend on the column at hand.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe that will be used for plotting
    missing_col : str
        The column with missing values, that will be used to group the other columns into 2 groups:
        rows that have NAN in missing_col, and rows that have a value in missing_col.
    
    Returns
    -------
    This function does not return anything.
    """
    
    if missing_col not in df.columns:
        raise ValueError(f"Could not find {missing_col} inside dataframe.")

    data = df.copy()

    # Map True/False missingness into 2 strings to appear in the legend
    data["Missingness"] = data[missing_col].isna().map({True: f"Missing value on {missing_col}", False: f"Observed value on {missing_col}"})

    for col in data.columns:
        if (col != missing_col) and (col != "Missingness"):
            plt.figure(figsize=(8, 6))
            sns.histplot(data=data, x=col, hue="Missingness",
                        multiple="dodge", edgecolor="grey", palette={f"Missing value on {missing_col}": "#C92C3E", f"Observed value on {missing_col}": "#029911"})

            plt.title(f"Distribution of {col} based on the missingness of {missing_col}")
            plt.xlabel(f"{col}")
            plt.ylabel("Count")
            plt.show()
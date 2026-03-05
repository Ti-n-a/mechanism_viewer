"""This file includes correlation heatmaps that use missingness of missing columns.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def missing_columns_correlation(df: pd.DataFrame):
    """
    Plots a correlation heatmap showing whether there are missing columns with missing
    values in similar rows. Therefore, it only works with columns with missing data.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to plot the missingness correlation heatmap
   
    Returns
    -------
    This function does not return anything.
    """
    data = df.copy()
    columns_with_na = data.columns[data.isna().any()].tolist()
    new_df = data[columns_with_na].corr()

    plt.figure(figsize=(6, 5))

    sns.heatmap(
    new_df,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Missingness Correlation"}
    )

    plt.title("Missingness Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def complete_and_missing_columns_correlation(df: pd.DataFrame):
    """
    Plots a correlation heatmap showing whether there are missing columns with missing
    values in similar rows. Besides, it shows whether the complete columns' values will
    correlate with the missing rows in the missing columns. Therefore, it works with both
    complete columns and missing columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to plot the correlation heatmap
   
    Returns
    -------
    This function does not return anything.
    """
    columns_without_na = df.columns[df.notna().all()].tolist()
    columns_with_na = df.columns[df.isna().any()].tolist()

    missing_dataset = df[columns_with_na].isna().astype(int)        # Transform missing columns into bool (1/0 instead of true/false) of missingness
    for column in columns_without_na:                               # Maintain values of complete columns
        missing_dataset[column] = df[column]

    corr = missing_dataset.corr()           

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)             # Mask the upper triangle

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
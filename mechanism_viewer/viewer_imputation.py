"""The below code implements imputation models to generate values for missing data in a specific
missing column, depending on its data type. The imputed data can then be used to find hidden
missing patterns, when comparing to other observed columns. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PoissonRegressor


def _impute_database(df: pd.DataFrame):
    """
    Imputes the database using MICE. MICE models correlations across variables, therefore,
    it is perfect to view MAR mechanism.
    
    Use when:
     - Column_name_y is continuous
     - Impute multiple columns at same time (even if they have NAN values)

    Do NOT use when:
     - Column_name_y is binary
     - Column_name_y is categorical
     - Column_name_y is discrete 

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe that will be used for imputation. Can have multiple columns
        with missing data.
       
    Returns
    -------
    new_data (pd.DataFrame): The dataframe with imputed values.
    """
    imputer = IterativeImputer(max_iter=10, random_state=42, sample_posterior=True)
    imputed = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed, columns=df.columns)

    return imputed_df


def _impute_database_binary(df: pd.DataFrame, column_name_missing: str):
    """
    Imputes the database using logistic regression for binary data. It trains the model to 
    calculate the probability of being positive class (e.g. "1"). Then, selects which imputed
    value the row will have, based of the probability obtained for each row (depends on the 
    other column).

    Use when:
    - column_name_missing is binary (0 or 1)

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe that will be used for imputation. Can not have columns
        with missing data.
    column_name_missing : str
        The name of the column with the missing data.
       
    Returns
    -------
    new_data (pd.DataFrame): The dataframe with imputed values.
    """
    mask_obs = df[column_name_missing].notna()
    mask_mis = df[column_name_missing].isna()

    X_obs = df.loc[mask_obs].drop(columns=[column_name_missing])
    y_obs = df.loc[mask_obs, column_name_missing]

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_obs, y_obs)

    probs = logreg.predict_proba(df.loc[mask_mis].drop(columns=[column_name_missing]))[:, 1]
    
    random_gener = np.random.default_rng()
    df.loc[mask_mis, column_name_missing] = random_gener.binomial(1, probs)
    return df


def _impute_database_discrete(df: pd.DataFrame, column_name_missing: str):
    """
    Imputes the database using Poisson regressor for discrete data that is not negative.
    It trains the model to predict a lambda_hat value for each row (the expected average number 
    based on other column values). Then, it picks a random value for each row using lambda_hat.
    This way, imputed values follow the predicted distribution of the model, instead of filling
    the rows using the mean.

    Use when:
    - column_name_missing is discrete and it does not have negative values

    Do NOT use when:
     - column_name_missing is binary (might predict values > 1)
     - column_name_missing is continuous
     - column_name_missing is categorical

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe that will be used for imputation. Can not have columns
        with missing data.
    column_name_missing : str
        The name of the column with the missing data.
       
    Returns
    -------
    new_data (pd.DataFrame): The dataframe with imputed values.
    """

    mask_obs = df[column_name_missing].notna()
    mask_mis = df[column_name_missing].isna()

    X_obs = df.loc[mask_obs].drop(columns=[column_name_missing])
    y_obs = df.loc[mask_obs, column_name_missing]

    model = PoissonRegressor(alpha=0.0, max_iter=1000)
    model.fit(X_obs, y_obs)

    lambda_hat = model.predict(df.loc[mask_mis].drop(columns=[column_name_missing]))
    df.loc[mask_mis, column_name_missing] = np.random.poisson(lambda_hat)
    return df


def _impute_database_categorical(df: pd.DataFrame, column_name_missing: str):
    """
    Imputes the database using logistic regression for categorical data. It computes for
    every row the probability distribution of all possible categories. Then, selects randomly
    a category based on its probability. This assures the natural variance of the data, instead
    of picking the highest category.

    Use when:
    - column_name_missing is categorical

    Do NOT use when:
     - column_name_missing is ordinal

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe that will be used for imputation. Can not have columns
        with missing data.
    column_name_missing : str
        The name of the column with the missing data.
       
    Returns
    -------
    new_data (pd.DataFrame): The dataframe with imputed values.
    """
    df = df.copy()

    mask_obs = df[column_name_missing].notna()
    mask_mis = df[column_name_missing].isna()

    X_obs = df.loc[mask_obs].drop(columns=[column_name_missing])
    y_obs = df.loc[mask_obs, column_name_missing]

    X_mis = df.loc[mask_mis].drop(columns=[column_name_missing])

    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X_obs, y_obs)

    probs = model.predict_proba(X_mis)
    classes = model.classes_

    df.loc[mask_mis, column_name_missing] = [np.random.choice(classes, p=p) for p in probs]

    return df


def scatterplot_imputation_comparison(df: pd.DataFrame, column_name_x: str, column_name_y: str, y_column_type: str = "normal"):
    """
    Plot a scatterplot that compares the values of a column with missing values, and the imputed values of those missing values. 
    A different column (an observed column) will be used to compare the values.
    All complete columns of the dataset will help the imputation algorithm guess the missing values.

    If the imputed values appear consistently distributed uniformly, then the missing data mechanism is plausible to be MCAR/MNAR.
    If the imputed values appear in a specific zone of the plot, then the column_name_y is most likely to have a MAR mechanism
    and be dependent on column_name_x.

    Note: If column_name_y does not depend on column_name_x to have its values missing, then, even if column_name_y has a MAR
    mechanism, the imputed values might appear to have an uniform distribution. Therefore, it is advisable to run more tests,
    before concluding the column missing mechanism.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataframe that will be used for imputation
    column_name_x : str
        The name of the complete column that will be on the x axis
    column_name_y : str
        The name of the column with missing data, that will have its values imputed
    y_column_type : str
        The type of data that column_name_y is, to better select the imputation model
    
    Returns
    -------
    This function does not return anything.
    """

    if column_name_y not in df.columns:
        raise ValueError(f"{column_name_y} not found in dataframe.")

    columns_without_na = df.columns[df.notna().all()].tolist()
    new_df = df[columns_without_na].copy()
    new_df[column_name_y] = df[column_name_y]

    if y_column_type == "binary":
        imputed_df = _impute_database_binary(new_df, column_name_y)
    elif y_column_type == "discrete_categorical":
        imputed_df = _impute_database_categorical(new_df, column_name_y)
    elif y_column_type == "discrete":
        imputed_df = _impute_database_discrete(new_df, column_name_y)
    else:
        imputed_df = _impute_database(df)

    plt.figure(figsize=(8, 6))

    plt.scatter(imputed_df[column_name_x], imputed_df[column_name_y], color='red', label='Data points')
    plt.scatter(df[column_name_x], df[column_name_y], color='blue', label='Data points')

    plt.title(f"Scatterplot of {column_name_y} with imputated values compared to {column_name_x}")
    plt.xlabel(f"{column_name_x}")
    plt.ylabel(f"{column_name_y}")

    plt.grid(True)  #Just to show grid

    plt.show()



def plot_imputation_distribution(df: pd.DataFrame, column_name: str, column_type: str = "normal"):
    """
    Plots a kdeplot comparing the distribution of a column with the distribution of the same column with imputed values.

    This does not offer much aid in figuring the data missing mechanism. It only serves the purpose of showcasing each
    imputation model working for the correct data type.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataframe that will be used for imputation
    column_name : str
        The name of the column with missing data, that will have its values imputed
    column_type : str
        The type of data that column_name is, to better select the imputation model
    
    Returns
    -------
    imputed_df (pd.DataFrame): The dataframe with imputed values on column_name, along with the complete data used for imputation.
    """

    if column_name not in df.columns:
        raise ValueError(f"{column_name} not found in dataframe.")
    if df[column_name].notna().all():
        raise ValueError(f"{column_name} is complete. This method requires the desired column to have missing values.")

    
    columns_without_na = df.columns[df.notna().all()].tolist()
    new_df = df[columns_without_na].copy()
    new_df[column_name] = df[column_name]

    if column_type == "binary":
        imputed_df = _impute_database_binary(new_df, column_name)
    elif column_type == "discrete_categorical":
        imputed_df = _impute_database_categorical(new_df, column_name)
    elif column_type == "discrete":
        imputed_df = _impute_database_discrete(new_df, column_name)
    else:
        imputed_df = _impute_database(df)

    plt.figure(figsize=(8, 5))
    sns.kdeplot(df[column_name].dropna(), label="Original", linewidth=2)
    sns.kdeplot(imputed_df[column_name], label="Imputed", linewidth=2)
    plt.title(f"Distribution of {column_name}: Original vs Imputed")
    plt.legend()
    plt.show()

    return imputed_df
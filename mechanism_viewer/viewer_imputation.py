"""The below code implements imputation models to generate values for missing data in a specific
missing column, depending on its data type. The imputed data can then be used to find hidden
missing patterns, when comparing to other observed columns. 
"""

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PoissonRegressor

from .column_data_types import ColType

from ._random import DEFAULT_RANDOM_STATE

from ._utils import get_single_class

from ._validation import validate_dataframe, validate_column, validate_missing_col


__all__ = [
    "scatterplot_imputation_comparison",
    "plot_imputation_distribution",
]


def _validate_observed_predictor(
    X_obs: pd.DataFrame,
    missing_col: str
    ) -> None:
    """
    Validate predictor dataset used by imputers with regression.

    Parameters
    ----------
    X_obs : pd.DataFrame
        Predictor dataset with only the rows where missing_col is observed
    missing_col : str
        The name of the column with missing data, that will have its values imputed

    Returns
    -------
    This function does not return anything.
    """
    if X_obs.shape[1] == 0:
        raise ValueError(f"Cannot run imputation for {missing_col} since no complete predictor columns were available.")
    non_numeric_cols = X_obs.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        raise TypeError(f"Cannot run imputation for {missing_col} since the predictors must be numeric. "
                        f"The existent non-numeric columns found: {non_numeric_cols}.")
    return


def _validate_observed_target_poisson(
    y_obs: pd.Series,
    missing_col: str
    ) -> None:
    """
    Validate observed target values for Poisson imputation.

    Parameters
    ----------
    y_obs : pd.Series
        Observed target values from rows where missing_col is not missing
    missing_col : str
        The name of the column with missing data, that will have its values imputed

    Returns
    -------
    This function does not return anything.
    """
    if not is_numeric_dtype(y_obs):
        raise TypeError(f"{missing_col} must be numeric for Poisson imputation.")
    if (y_obs < 0).any():
        raise ValueError(f"{missing_col} contains negative values, but Poisson imputation requires non-negative values.")
    return


def _validate_observed_target_binary(
    y_obs: pd.Series,
    missing_col: str
    ) -> None:
    """
    Validate observed target values for binary imputation.

    Parameters
    ----------
    y_obs : pd.Series or array-like
        Observed target values from rows where missing_col is not missing.
    missing_col : str
        The name of the column with missing data, that will have its values imputed

    Returns
    -------
    This function does not return anything.
    """
    unique_classes = pd.Series(y_obs).dropna().unique().tolist()
    if not set(unique_classes).issubset({0, 1}):
        raise ValueError(f"{missing_col} must contain only 0/1 values for binary imputation, but the classes are: {unique_classes}")
    return


def _impute_database_mice(
    df: pd.DataFrame,
    random_state: int = DEFAULT_RANDOM_STATE
    ) -> pd.DataFrame:
    """
    Imputes the database using MICE. MICE models correlations across variables, therefore,
    it is perfect to view MAR mechanism.
    
    Use when:
     - missing_col is continuous
     - Impute multiple columns at same time (even if they have NAN values)

    Do NOT use when:
     - missing_col is binary
     - missing_col is categorical
     - missing_col is discrete 

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe that will be used for imputation. Can have multiple columns
        with missing data.
    random_state : int, default = 42
        Seed used in stochastic imputation routines for reproducible results
       
    Returns
    -------
    pd.DataFrame
        The dataframe with imputed values.
    """
    imputer = IterativeImputer(max_iter=10, random_state=random_state, sample_posterior=True)
    imputed = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed, columns=df.columns)

    return imputed_df


def _impute_database_binary(
    df: pd.DataFrame,
    missing_col: str,
    random_state: int = DEFAULT_RANDOM_STATE
    ) -> pd.DataFrame:
    """
    Imputes the database using logistic regression for binary data. It trains the model to 
    calculate the probability of being positive class (e.g. "1"). Then, selects which imputed
    value the row will have, based of the probability obtained for each row (depends on the 
    other column).

    Use when:
    - missing_col is binary (0 or 1)

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe that will be used for imputation. Can not have columns
        with missing data
    missing_col : str
        The name of the column with the missing data
    random_state : int, default = 42
        Seed used in stochastic imputation routines for reproducible results
       
    Returns
    -------
    pd.DataFrame
        The dataframe with imputed values.
    """
    imputed_df = df.copy()
    
    mask_obs = df[missing_col].notna()
    mask_mis = df[missing_col].isna()

    X_obs = df.loc[mask_obs].drop(columns=[missing_col])
    y_obs = df.loc[mask_obs, missing_col]

    _validate_observed_predictor(X_obs, missing_col)
    _validate_observed_target_binary(y_obs, missing_col)

    single_class = get_single_class(y_obs)

    if single_class is not None:
        # To impute with the only class available, instead of making the model fail
        imputed_df.loc[mask_mis, missing_col] = single_class

        return imputed_df
    else:
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_obs, y_obs)

        probs = lr_model.predict_proba(df.loc[mask_mis].drop(columns=[missing_col]))[:, 1]
        
        random_gener = np.random.default_rng(random_state)

        imputed_df.loc[mask_mis, missing_col] = random_gener.binomial(1, probs)

        return imputed_df


def _impute_database_discrete(
    df: pd.DataFrame,
    missing_col: str,
    random_state: int = DEFAULT_RANDOM_STATE
    ) -> pd.DataFrame:
    """
    Imputes the database using Poisson regressor for discrete data that is not negative.
    It trains the model to predict a lambda_hat value for each row (the expected average number 
    based on other column values). Then, it picks a random value for each row using lambda_hat.
    This way, imputed values follow the predicted distribution of the model, instead of filling
    the rows using the mean.

    Use when:
    - missing_col is discrete and it does not have negative values

    Do NOT use when:
     - missing_col is binary (might predict values > 1)
     - missing_col is continuous
     - missing_col is categorical

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe that will be used for imputation. Can not have columns
        with missing data
    missing_col : str
        The name of the column with the missing data
    random_state : int, default = 42
        Seed used in stochastic imputation routines for reproducible results
       
    Returns
    -------
    pd.DataFrame
        The dataframe with imputed values.
    """
    imputed_df = df.copy()

    mask_obs = df[missing_col].notna()
    mask_mis = df[missing_col].isna()

    X_obs = df.loc[mask_obs].drop(columns=[missing_col])
    y_obs = df.loc[mask_obs, missing_col]

    _validate_observed_predictor(X_obs, missing_col)
    _validate_observed_target_poisson(y_obs, missing_col)

    model = PoissonRegressor(alpha=0.0, max_iter=1000)
    model.fit(X_obs, y_obs)

    lambda_hat = model.predict(df.loc[mask_mis].drop(columns=[missing_col]))

    random_gener = np.random.default_rng(random_state)
    imputed_df.loc[mask_mis, missing_col] = random_gener.poisson(lambda_hat)
    
    return imputed_df


def _impute_database_categorical(
    df: pd.DataFrame,
    missing_col: str,
    random_state: int = DEFAULT_RANDOM_STATE
    ) -> pd.DataFrame:
    """
    Imputes the database using logistic regression for categorical data. It computes for
    every row the probability distribution of all possible categories. Then, selects randomly
    a category based on its probability. This assures the natural variance of the data, instead
    of picking the highest category.

    Use when:
    - missing_col is categorical

    Do NOT use when:
     - missing_col is ordinal

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe that will be used for imputation. Can not have columns
        with missing data
    missing_col : str
        The name of the column with the missing data
    random_state : int, default = 42
        Seed used in stochastic imputation routines for reproducible results
       
    Returns
    -------
    pd.DataFrame
        The dataframe with imputed values.
    """
    imputed_df = df.copy()

    mask_obs = df[missing_col].notna()
    mask_mis = df[missing_col].isna()

    X_obs = df.loc[mask_obs].drop(columns=[missing_col])
    y_obs = df.loc[mask_obs, missing_col]

    X_mis = df.loc[mask_mis].drop(columns=[missing_col])

    _validate_observed_predictor(X_obs, missing_col)

    single_class = get_single_class(y_obs)

    if single_class is not None:
        # To impute with the only class available, instead of making the model fail
        imputed_df.loc[mask_mis, missing_col] = single_class
        
        return imputed_df
    else:
        lr_model = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr_model.fit(X_obs, y_obs)

        probs = lr_model.predict_proba(X_mis)
        classes = lr_model.classes_

        random_gener = np.random.default_rng(random_state)
        imputed_df.loc[mask_mis, missing_col] = [random_gener.choice(classes, p=p) for p in probs]

        return imputed_df


def _impute_database(
    df: pd.DataFrame,
    missing_col: str,
    missing_col_type: str = "normal",
    random_state: int = DEFAULT_RANDOM_STATE
    ) -> pd.DataFrame:
    """
    Imputes the given missing_col with the appropriate imputation model.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataframe that will be used for imputation
    missing_col : str
        The name of the column with missing data, that will have its values imputed
    missing_col_type : str, default = "normal"
        The type of data that missing_col is, to better select the imputation model
    random_state : int, default = 42
        Seed used in stochastic imputation routines for reproducible results
    
    Returns
    -------
    pd.DataFrame
        The dataframe with imputed values on missing_col.
    """
    if missing_col_type == ColType.BINARY:
        imputed_df = _impute_database_binary(df, missing_col, random_state)
    elif missing_col_type == ColType.DISC_CATEGORICAL:
        imputed_df = _impute_database_categorical(df, missing_col, random_state)
    elif missing_col_type == ColType.DISCRETE:
        imputed_df = _impute_database_discrete(df, missing_col, random_state)
    else:
        imputed_df = _impute_database_mice(df, random_state)

    return imputed_df
    

def _prepare_missingness_dataset(
    df: pd.DataFrame,
    missing_col: str
    ) -> pd.DataFrame:
    """
    Prepare the dataset used for the imputation.
    
    The function selects the complete columns and the missing_col to be used for imputation.

    Parameters
    ----------
    df : pd.DataFrame,
        The dataset from which complete columns and the missing_col will be obtained
    missing_col : str
        The name of the column with missing data, that will have its values imputed

    Returns
    -------
    pd.DataFrame
        The dataframe containing only the complete columns and the missing_col.
    """
    columns_without_na = df.columns[df.notna().all()].tolist()
    prepared_df = df[columns_without_na].copy()
    prepared_df[missing_col] = df[missing_col]

    return prepared_df


def scatterplot_imputation_comparison(
    df: pd.DataFrame,
    column_name: str,
    missing_col: str,
    missing_col_type: str = "normal",
    random_state: int = DEFAULT_RANDOM_STATE,
    display_plot: bool = False
    ) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a scatterplot that compares the values of a column with missing values, and the imputed values of those missing values. 
    A different column (an observed column) will be used to compare the values.
    All complete columns of the dataset will help the imputation algorithm guess the missing values.

    If the imputed values appear consistently distributed uniformly, then the missing data mechanism is plausible to be MCAR/MNAR.
    If the imputed values appear in a specific zone of the plot, then the missing_col is most likely to have a MAR mechanism
    and be dependent on column_name.

    Note: If missing_col does not depend on column_name to have its values missing, then, even if missing_col has a MAR
    mechanism, the imputed values might appear to have an uniform distribution. Therefore, it is advisable to run more tests,
    before concluding the column missing mechanism.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataframe that will be used for imputation
    column_name : str
        The name of the complete column that will be on the x axis
    missing_col : str
        The name of the column with missing data, that will have its values imputed
    missing_col_type : str, default = "normal"
        The type of data that missing_col is, to better select the imputation model
    random_state : int, default = 42
        Seed used in stochastic imputation routines for reproducible results.
    display_plot : bool, default = False
        If True, displays figure with ``plt.show()``
    
    Returns
    -------
    tuple
        (fig_imput_comp, ax_imput_comp) representing the plot available for display.
    """
    validate_dataframe(df)
    validate_column(df, column_name)
    validate_missing_col(df, missing_col)

    prepared_df = _prepare_missingness_dataset(df, missing_col)

    imputed_df = _impute_database(prepared_df, missing_col, missing_col_type, random_state)

    # Warn about the possibility column_name was not used for imputation 
    # TODO: Review later due to MICE being 'useless'. Currently, verbose code.
    # Saved warning comment: f"Currently, using the original values of {column_name} for x-axis in the imputed scatter."
    if column_name in imputed_df.columns:
        imputed_x = imputed_df[column_name]
    else:
        imputed_x = df[column_name]
        warnings.warn((f"{column_name} is not a complete column in the original dataset. "
                        "Thus, it was not used for the imputation process. "
                      ), stacklevel=2)

    # To make the plt.scatter() process of displaying the data points more transparent
    plottable_points_before_imput = df[column_name].notna() & df[missing_col].notna()
    plottable_points_after_imput = imputed_x.notna() & imputed_df[missing_col].notna()

    dropped_after_imput = int((~plottable_points_after_imput).sum())
    if dropped_after_imput > 0:
        warnings.warn((f"Some data points were not plotted due to existence of a missing value in either {column_name} or {missing_col}. "
                      f"Total dropped after imputation: {dropped_after_imput}."
                      ), stacklevel=2)

    fig_imput_comp, ax_imput_comp = plt.subplots(figsize=(8, 6))
    ax_imput_comp.scatter(imputed_x[plottable_points_after_imput], imputed_df.loc[plottable_points_after_imput, missing_col], color='red', label='Imputed values')   # Data points
    ax_imput_comp.scatter(df.loc[plottable_points_before_imput, column_name], df.loc[plottable_points_before_imput, missing_col], color='blue', label='Observed values')
    ax_imput_comp.set_title(f"Scatterplot of {missing_col} with imputed values compared to {column_name}")
    ax_imput_comp.set_xlabel(f"{column_name}")
    ax_imput_comp.set_ylabel(f"{missing_col}")
    ax_imput_comp.grid(True)  # Just to show grid
    ax_imput_comp.legend()

    if display_plot:
        plt.show()
    else:
        plt.close(fig_imput_comp)
    
    return fig_imput_comp, ax_imput_comp


def plot_imputation_distribution(
    df: pd.DataFrame,
    missing_col: str,
    missing_col_type: str = "normal",
    random_state: int = DEFAULT_RANDOM_STATE,
    display_plot: bool = False
    ) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Plots a kdeplot comparing the distribution of a column with the distribution of the same column with imputed values.

    This does not offer much aid in figuring the data missing mechanism. It only serves the purpose of showcasing each
    imputation model working for the correct data type.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataframe that will be used for imputation
    missing_col : str
        The name of the column with missing data, that will have its values imputed
    missing_col_type : str, default = "normal"
        The type of data that missing_col is, to better select the imputation model
    random_state : int, default = 42
        Seed used in stochastic imputation routines for reproducible results.
    display_plot : bool, default = False
        If True, displays figure with ``plt.show()``
    
    Returns
    -------
    tuple
        (fig_imput_dist, ax_imput_dist, imputed_df) representing the plot available for display, and the dataframe with imputed
        values on missing_col, along with the complete data used for imputation.
    """
    validate_dataframe(df)
    validate_missing_col(df, missing_col)
    
    prepared_df = _prepare_missingness_dataset(df, missing_col)

    imputed_df = _impute_database(prepared_df, missing_col, missing_col_type, random_state)

    fig_imput_dist, ax_imput_dist = plt.subplots(figsize=(8, 5))

    if missing_col_type in (ColType.BINARY, ColType.DISC_CATEGORICAL):
        original_series = df[missing_col].dropna().astype(str)
        imputed_series = imputed_df[missing_col].astype(str)
        plot_df = pd.concat([pd.DataFrame({"value": original_series, "source": "Original"}),
                             pd.DataFrame({"value": imputed_series, "source": "Imputed"}),
                            ], ignore_index=True,)
        
        sns.countplot(data=plot_df, x="value", hue="source", ax=ax_imput_dist)
        ax_imput_dist.set_xlabel(missing_col)
        ax_imput_dist.set_ylabel("Count")
    elif missing_col_type == ColType.DISCRETE:
        sns.histplot(df[missing_col].dropna(), label="Original", alpha=0.5, stat="density", discrete=True, ax=ax_imput_dist)
        sns.histplot(imputed_df[missing_col], label="Imputed", alpha=0.5, stat="density", discrete=True, ax=ax_imput_dist)
    else:
        sns.kdeplot(df[missing_col].dropna(), label="Original", linewidth=2, ax=ax_imput_dist)
        sns.kdeplot(imputed_df[missing_col], label="Imputed", linewidth=2, ax=ax_imput_dist)

    ax_imput_dist.set_title(f"Distribution of {missing_col}: Original vs Imputed")
    ax_imput_dist.legend()
    
    if display_plot:
        plt.show()
    else:
        plt.close(fig_imput_dist)

    return fig_imput_dist, ax_imput_dist, imputed_df
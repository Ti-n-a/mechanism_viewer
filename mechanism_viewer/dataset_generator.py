"""This module provides tools for generating synthetic datasets and applying various
missing data mechanisms.

It allows the generation of complete datasets with customizable properties, such as
the number of rows, columns, and data types.

In addition to defining the missing data mechanism for each column of the dataset, users
can specify each column's missing rates and dependencies.

This module enables users to simulate datasets with missing data patterns for visualizing
missing data and testing the capabilities of other module tools.
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder

from .column_data_types import ColType

from ._random import DEFAULT_RANDOM_STATE

__all__ = [
    "generate_synthetic_dataset",
    "apply_mcar",
    "apply_mar",
    "apply_mnar",
    "apply_missing_data",
    "generate_dataset_with_missing_data",
]


def _validate_and_convert_column_to_numeric(
    column: pd.Series
    ) -> pd.Series:
    """
    Validates if the given column has numerical data. If the column is
    non-numeric, then convert the data to numeric. This conversion is
    important for MAR and MNAR mechanism, since they rely on orderable
    data to remove row values using argsort.
    
    Parameters
    ----------
    column: pd.Series
        The column given to validate if it is numerical,
        and convert it to numerical otherwise.
 
    Returns
    -------
    pd.Series
        The column data as numerical data, if the original data was non-numeric.
    """
    if not is_numeric_dtype(column):
        label_encoder = LabelEncoder()                              
        new_column = label_encoder.fit_transform(column)
        return pd.Series(new_column, index=column.index, name=column.name)
    else:
        return column
    

def _validate_and_convert_to_numeric(
    data: pd.Series | pd.DataFrame
    ) -> pd.Series | pd.DataFrame:
    """
    Applies the given input to a validation function depending
    on the data type of the input.
    
    Parameters
    ----------
    data : pd.Series | pd.DataFrame
        The data given to validate if it is numerical,
        and convert it to numerical otherwise.
 
    Returns
    -------
    pd.Series | pd.DataFrame
        The input data as numerical data, if the original data was non-numeric 
    """

    if isinstance(data, pd.Series):
        return _validate_and_convert_column_to_numeric(data)

    if isinstance(data, pd.DataFrame):
        return data.apply(_validate_and_convert_column_to_numeric)

    raise TypeError("The given data must be a pandas Series or a pandas DataFrame")


def _validate_missing_rate(
    missing_rate: float
    ) -> None:
    """
    Validates the value of the missing rate given as input.
    
    Parameters
    ----------
    missing_rate : float
        The missing rate value the dataset must have.
 
    Returns
    -------
    This function does not return anything.
    """
    if (missing_rate < 0) or (missing_rate > 1):
        raise ValueError(f"The given missing rate {missing_rate} is out of the range [0,1]")
    
    return


def _validate_rows(
    n_rows: int
    ) -> None:
    """
    Validates the shape of a dataset given as input.
    
    Parameters
    ----------
    n_rows : int
        The number of rows the dataset must have.
 
    Returns
    -------
    This function does not return anything.
    """
    if n_rows < 0:
        raise ValueError(f"The given number of rows {n_rows} is a negative value")
        
    return
    

def _validate_n_complete_cols(
        n_complete_cols: int,
        n_cols: int,
    ) -> None:
    """
    Validates the number of complete columns given as input.
    
    Parameters
    ----------
    n_complete_cols : int
        The number of complete columns the dataset must have.
    n_cols : int
        The number of columns the dataset has.
 
    Returns
    -------
    This function does not return anything.
    """
    if n_complete_cols < 0:
        raise ValueError(f"The number of complete columns ({n_complete_cols}) is a negative value")

    if n_complete_cols > n_cols:
        raise ValueError(f"There are more complete columns than existing columns: {n_complete_cols} vs {n_cols}")
    
    return


def _validate_missing_mechanism_array(
    missing_mechanism_array: list[str],
    n_cols: int,
    ) -> None:
    """
    Validates the data missing mechanism array given as input.
    
    Parameters
    ----------
    missing_mechanism_array : list[str]
        The array with the data missing_mechanism of each column.
    n_cols : int
        The number of columns the dataset must have.
 
    Returns
    -------
    This function does not return anything.
    """
    if len(missing_mechanism_array) != n_cols:
        raise ValueError(f"There is a mismatch between the number of missing mechanisms ({missing_mechanism_array}) and the number of wanted missing columns ({n_cols})")

    allowed_mechanisms = {"MCAR", "MAR", "MNAR"}

    for missing_mechanism in missing_mechanism_array:
        if missing_mechanism not in allowed_mechanisms:
            raise ValueError(f"There is an unknown missing mechanism: {missing_mechanism}")

    return


def _validate_type_array(
    type_array: list[str]
    ) -> None:
    """
    Validates the type array given as input.
    
    Parameters
    ----------
    type_array : list[str]
        The array with the type of each column.

    Returns
    -------
    This function does not return anything.
    """
    allowed_types = {
        ColType.CONTINUOUS,
        ColType.DISC_CATEGORICAL,
        ColType.DISCRETE,
        ColType.BINARY,
    }

    for column_type in type_array:
        if column_type not in allowed_types:
            raise ValueError(f"There is an unknown type of column: {column_type}")

    return


def _validate_column_is_complete(
    column_data: pd.Series
    ) -> None:
    """
    Validates whether the given column has no missing values to proceed
    with applying a missing data mechanism.
    
    Parameters
    ----------
    column_data : pd.Series
        The column where it will be applied the the missing
        data mechanism.
 
    Returns
    -------
    This function does not return anything.
    """
    if column_data.isna().any():
        raise ValueError(f"The column {column_data.name} has already missing values. The missing mechanism needs to be applied on a complete dataset. This way, a column with a mixture of missing data mechanisms is not returned.")

    return


def _prepare_apply_mechanism(
    column_data: pd.Series,
    missing_rate: float = 0.1
    ) -> tuple[int, pd.Series]:
    """
    Validates the missing rate and column data given as input, calculates
    the number of missing rows to remove on column_data, and creates a 
    copy of the column.
    
    Parameters
    ----------
    column_data : pd.Series
        The column where it will be applied the the missing
        data mechanism.
    missing_rate : float = 0.1
        The missing rate the new column should have after
        applying the missing data mechanism.
 
    Returns
    -------
    tuple[int, pd.Series]
        The total number of missing_rows and a copy of column_data.
    """
    _validate_missing_rate(missing_rate)

    _validate_column_is_complete(column_data)

    total_missing_rows = int(len(column_data) * missing_rate)

    new_column = column_data.copy()

    return total_missing_rows, new_column


def generate_synthetic_dataset(
    n_rows: int,
    type_array: list[str],
    random_state: int = DEFAULT_RANDOM_STATE
    ) -> pd.DataFrame:
    """
    Generates a synthetic dataset according to the properties given.
    
    Parameters
    ----------
    n_rows : int
        The number of rows the dataset must have.
    type_array : list[str]
        The array with the type of each column.
    random_state : int, default = 42
        Seed used in stochastic routines for reproducible results.
   
    Returns
    -------
    data (pd.DataFrame): A dataframe with the given properties
    """
    _validate_rows(n_rows)

    _validate_type_array(type_array)

    n_cols = len(type_array)

    rng = np.random.default_rng(random_state)

    col_names = [f"Col{i+1}" for i in range(n_cols)]     # Names columns as Col1, Col2, Col3...

    data = pd.DataFrame()

    for i, col_name in enumerate(col_names):
        if type_array[i] == ColType.CONTINUOUS:
            data[col_name] = rng.normal(0, 1, n_rows)  # Normal (Gaussian) Distribution
        elif type_array[i] == ColType.DISC_CATEGORICAL:
            K = 10  # number of categories
            data[col_name] = rng.integers(0, K, size=n_rows)       # Discrete uniform distribution
            data[col_name] = data[col_name].astype("Int64")                             # Transforms column to nullable integer type, because pandas normally transforms data into floats when missing values are applied
        elif type_array[i] == ColType.DISCRETE:
            data[col_name] = rng.poisson(lam=5, size=n_rows)  # Poisson Distribution (Counts)
            data[col_name] = data[col_name].astype("Int64")  
        elif type_array[i] == ColType.BINARY:
            data[col_name] = rng.choice([0, 1], size=n_rows)
            data[col_name] = data[col_name].astype("Int64")  
        else:
            raise ValueError(f"There is an unknown type of data: {type_array[i]}")
    
    return data


def apply_mcar(
    column_data: pd.Series,
    missing_rate: float = 0.1,
    random_state: int = DEFAULT_RANDOM_STATE
    ) -> pd.Series:
    """
    Applies MCAR mechanism to one column.
    
    Parameters
    ----------
    column_data : pd.Series
        The column to transform
    missing_rate : float = 0.1
        The missing rate of the column
    random_state : int, default = 42
        Seed used in stochastic routines for reproducible results.
  
    Returns
    -------
    new_column (pd.Series): A copy of the column with the MCAR mechanism applied
    """
    n_rows = len(column_data)
    total_missing_rows, new_column = _prepare_apply_mechanism(column_data, missing_rate)

    rng = np.random.default_rng(random_state)
    missing_indices = rng.choice(n_rows, size=total_missing_rows, replace=False) # Samples missing rows randomly

    new_column.iloc[missing_indices] = np.nan

    return new_column 


def apply_mar(
    column_data: pd.Series,
    observable_df: pd.DataFrame,
    missing_rate: float = 0.1,
    missingness_ascending: bool = True
    ) -> pd.Series:
    """
    Applies MAR mechanism to one column.
    
    Parameters
    ----------
    column_data : pd.Series
        The column to transform
    observable_df : pd.DataFrame
        The column(s) that are observable and will be used as missingness dependency 
    missing_rate : float = 0.1
        The missing rate of the column
    missingness_ascending : bool = True
        Indicates whether the highest/lowest values must become missing 
    
    Returns
    -------
    new_column (pd.Series): A copy of the column with the MAR mechanism applied
    """
    total_missing_rows, new_column = _prepare_apply_mechanism(column_data, missing_rate)

    n_complete_cols = len(observable_df.columns)
    if n_complete_cols < 1:
        raise ValueError(f"There must be at least 1 observable column for MAR: {n_complete_cols}")

    new_observable_df = _validate_and_convert_to_numeric(data=observable_df)

    score = new_observable_df.sum(axis=1) # Calculates a score of each row by summing the values on the complete columns

    if missingness_ascending:
        sorted_idx = np.argsort(score)  # Makes rows with the highest values last (to make them more likely missing)
    else:
        sorted_idx = np.argsort(-score) # Makes rows with the lowest values last (descending order)

    missing_indices = sorted_idx[-total_missing_rows:] # Marks the last rows of the sorted_idx to become missing

    new_column.iloc[missing_indices] = np.nan
    
    return new_column


def apply_mnar(
    column_data: pd.Series,
    missing_rate: float = 0.1,
    missingness_ascending: bool = True
    ) -> pd.Series:
    """
    Applies MNAR mechanism to one column.
    
    Parameters
    ----------
    column_data : pd.Series
        The column to transform
    missing_rate : float = 0.1
        The missing rate of the column
    missingness_ascending : bool = True
        Indicates whether the highest/lowest values must become missing 
    
    Returns
    -------
    new_column (pd.Series): A copy of the column with the MNAR mechanism applied
    """
    total_missing_rows, new_column = _prepare_apply_mechanism(column_data, missing_rate)

    new_column_int = _validate_and_convert_to_numeric(data=new_column)

    if missingness_ascending:
        sorted_idx = np.argsort(new_column_int) # Makes rows with the highest values last (to make them more likely missing) of that missing column
    else:
        sorted_idx = np.argsort(-new_column_int) # Makes rows with the lowest values last (descending order)

    missing_indices = sorted_idx[-total_missing_rows:] # Marks the last rows of the sorted_idx to become missing
    
    new_column.iloc[missing_indices] = np.nan

    return new_column


def apply_missing_data(
    data: pd.DataFrame,
    n_complete_cols: int,
    missing_mechanism_array: list[str],
    missing_rate_array: list[float],
    missingness_ascending: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE
    ) -> pd.DataFrame:
    """
    Applies multiple missing data mechanisms to a dataframe
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataframe that will be transformed
    n_complete_cols : int
        The number of the first columns of the dataframe that won't have the missing mechanism applied
    missing_mechanism_array : list[str]
        The list containing the missing mechanism that needs to be applied for each column (excluding the complete columns)
    missing_rate_array : list[float]
        The missing rate of each column with missing mechanism applied
    missingness_ascending : bool = True
        Indicates whether the highest/lowest values must become missing 
    random_state : int, default = 42
        Seed used in stochastic routines for reproducible results.
    
    Returns
    -------
    new_data (pd.DataFrame): A copy of the dataframe with the missing data mechanisms applied
    """
    _, n_cols = data.shape
    new_data = data.copy()

    _validate_n_complete_cols(n_complete_cols, n_cols)

    n_incomplete_cols = n_cols-n_complete_cols

    _validate_missing_mechanism_array(missing_mechanism_array, n_incomplete_cols)

    if len(missing_rate_array) != n_incomplete_cols:
        raise ValueError(f"There is a mismatch between the number of missing rates ({missing_rate_array}) and the number of wanted missing columns ({n_cols-n_complete_cols})")

    missing_cols = data.columns[n_complete_cols:]
    for i, col in enumerate(missing_cols):  
        if missing_mechanism_array[i] == "MCAR":     # Missingness is random
            # Not using the same random_state for every MCAR column.
            # Although the same random_state would be simpler, it could correlate patterns across columns
            new_data[col] = apply_mcar(new_data[col], missing_rate_array[i], random_state + i)

        elif missing_mechanism_array[i] == "MAR":    # Missingness depends on complete features
            new_data[col] = apply_mar(new_data[col], new_data.iloc[:,:n_complete_cols], missing_rate_array[i], missingness_ascending)

        elif missing_mechanism_array[i] == "MNAR":   # Missingness depends on the values of the column (higher/lower values are more likely to be missing)       
            new_data[col] = apply_mnar(new_data[col], missing_rate_array[i], missingness_ascending)
        else:
            raise ValueError(f"There is an unknown missing mechanism: {missing_mechanism_array[i]}")

    return new_data


def generate_dataset_with_missing_data(
    n_rows: int,
    type_array: list[str],
    n_complete_cols: int,
    missing_mechanism_array: list[str],
    missing_rate_array: list[float],
    missingness_ascending: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE
    ) -> pd.DataFrame:
    """
    Generates a dataset with missing data mechanisms. Applies the above functions for user usability.
    
    Parameters
    ----------
    n_rows : int
        Indicates the number of rows the dataset must have
    type_array : list[str]
        Indicates the type of each column
    n_complete_cols : int
        The number of the first columns of the dataframe that won't have the missing mechanism applied
    missing_mechanism_array : list[str]
        The list containing the missing mechanism that needs to be applied for each column (excluding the complete columns)
    missing_rate_array : list[float]
        The missing rate of each column with missing mechanism applied
    missingness_ascending : bool = True
        Indicates whether the highest/lowest values must become missing 
    random_state : int, default = 42
        Seed used in stochastic routines for reproducible results.
    
    Returns
    -------
    new_data (pd.DataFrame): A copy of the dataframe with the missing data mechanisms applied
    """
    data = generate_synthetic_dataset(n_rows, type_array, random_state)
    missing_dataset = apply_missing_data(data, n_complete_cols, missing_mechanism_array, missing_rate_array, missingness_ascending, random_state)
    
    return missing_dataset
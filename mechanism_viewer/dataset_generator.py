import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder

def generate_synthetic_dataset(n_rows: int, n_cols: int, type_array: list[str]):
    """
    Generates a synthetic dataset according to the properties given
    
    Parameters:
    n_rows (int): Indicates the number of rows the dataset must have
    n_cols (int): Indicates the number of columns the dataset must have
    type_array (list[str]): Indicates the type of each column
   
    Returns:
    data (pd.DataFrame): A dataframe with the given properties
    """
    np.random.seed(42)

    if len(type_array) != n_cols:
        raise ValueError(f"There is a mismatch between the number of column types ({type_array}) and the number of columns ({n_cols})")

    col_names = [f"Col{i+1}" for i in range(n_cols)]     # Names columns as Col1, Col2, Col3...

    data = pd.DataFrame()
    random_gener = np.random.default_rng()

    for i, col_name in enumerate(col_names):
        if type_array[i] == "continuous":
            data[col_name] = random_gener.normal(0, 1, n_rows)  # Normal (Gaussian) Distribution
        elif type_array[i] == "discrete_categorical":
            K = 10  # number of categories
            data[col_name] = random_gener.integers(0, K, size=n_rows, endpoint=True)    # Discrete uniform distribution. Endpoint to make K inclusive
            data[col_name] = data[col_name].astype("Int64")                             # Transforms column to nullable integer type, because pandas normally transforms data into floats when missing values are applied
        elif type_array[i] == "discrete":
            data[col_name] = random_gener.poisson(lam=5, size=n_rows)  # Poisson Distribution (Counts)
            data[col_name] = data[col_name].astype("Int64")  
        elif type_array[i] == "binary":
            data[col_name] = random_gener.choice([0, 1], size=n_rows)
            data[col_name] = data[col_name].astype("Int64")  
        else:
            raise ValueError(f"There is an unknown type of data: {type_array[i]}")
    
    return data


def apply_MCAR(column_data: pd.Series, missing_rate: float = 0.1) -> pd.Series:
    """
    Applies MCAR mechanism to one column.
    
    Parameters:
    column_data (pd.Series): The column to transform
    missing_rate (float): The missing rate of the column
  
    Returns:
    new_column (pd.Series): A copy of the column with the MCAR mechanism applied
    """
    n_rows = len(column_data)
    new_data = column_data.copy()

    total_missing_rows = int(n_rows * missing_rate)  # Calculates the number of missing rows to update

    missing_cells = [ row for row in range(n_rows)] # Creates all possible cells that can have missing value
    missing_indices = random.sample(missing_cells, total_missing_rows) # Samples X (total_missing_values) random cells

    for row in missing_indices:
        new_data.iloc[row] = np.nan

    return new_data 


def apply_MAR(column_data: pd.Series, observable_df: pd.DataFrame, missing_rate: float = 0.1, missingness_ascending: bool = True) -> pd.Series:
    """
    Applies MAR mechanism to one column.
    
    Parameters:
    column_data (pd.Series): The column to transform
    observable_df (pd.DataFrame): The column(s) that are observable and will be used as missingness dependency 
    missing_rate (float): The missing rate of the column
    missingness_ascending (bool): Indicates whether the highest/lowest values must become missing 
    
    Returns:
    new_column (pd.Series): A copy of the column with the MAR mechanism applied
    """

    n_rows = len(column_data)
    new_data = column_data.copy()

    n_complete_cols = len(observable_df.columns)
    if n_complete_cols < 1:
        raise ValueError(f"There must be at least 1 observable column for MAR: {n_complete_cols}")

    total_missing_rows = int(n_rows * missing_rate)

    label_encoder = LabelEncoder()
    for col in observable_df.columns:
        # Check if the column is numeric, and transform it, if it is string
        if not pd.api.types.is_numeric_dtype(observable_df[col]):
            observable_df[col] = label_encoder.fit_transform(observable_df[col])

    score = observable_df.sum(axis=1) # Calculates a score of each row by summing the values on the complete columns

    if missingness_ascending:
        sorted_idx = np.argsort(score) # Sorts rows by the highest score (to make them more likely missing)
    else:
        sorted_idx = np.argsort(-score) # Sorts rows by the lowest score (descending order)

    rows_to_miss = sorted_idx[-total_missing_rows:] # Make rows with highest score become missing (last X rows of the sorted index)

    new_data[rows_to_miss] = np.nan
    return new_data


def apply_MNAR(column_data: pd.Series, missing_rate: float = 0.1, missingness_ascending: bool = True) -> pd.Series:
    """
    Applies MNAR mechanism to one column.
    
    Parameters:
    column_data (pd.Series): The column to transform
    missing_rate (float): The missing rate of the column
    missingness_ascending (bool): Indicates whether the highest/lowest values must become missing 
    
    Returns:
    new_column (pd.Series): A copy of the column with the MNAR mechanism applied
    """
    n_rows = len(column_data)
    new_column = column_data.copy()

    total_missing_rows = int(n_rows * missing_rate)

    if not pd.api.types.is_numeric_dtype(new_column):
        label_encoder = LabelEncoder()                              
        new_column_int = label_encoder.fit_transform(new_column)   # Convert values to numeric so they can be ordered by argsort
    else:
        new_column_int = new_column

    if missingness_ascending:
        sorted_idx = np.argsort(new_column_int) # Sorts rows by the highest values (to make them more likely missing) of that missing column
    else:
        sorted_idx = np.argsort(-new_column_int) # Sorts rows by the lowest values (descending order)

    rows_to_miss = sorted_idx[-total_missing_rows:] # Make rows with highest/lowest value become missing (last X rows of the sorted index)
    
    new_column[rows_to_miss] = np.nan

    return new_column


def apply_missing_data(data: pd.DataFrame, missing_mechanism_array: list[str], missing_rate_array: list[float], n_complete_cols: int = 1, missingness_ascending: bool = True):
    """
    Applies multiple missing data mechanisms to a dataframe
    
    Parameters:
    data (pd.DataFrame): The dataframe that will be transformed
    missing_mechanism_array (list[str]): The list containing the missing mechanism that needs to be applied for each column (excluding the complete columns)
    missing_rate_array (list[float]): The missing rate of each column with missing mechanism applied
    n_complete_cols (int): The number of the first columns of the dataframe that won't have the missing mechanism applied
    missingness_ascending (bool): Indicates whether the highest/lowest values must become missing 
    
    Returns:
    new_data (pd.DataFrame): A copy of the dataframe with the missing data mechanisms applied
    """
    _, n_cols = data.shape
    new_data = data.copy()

    if n_complete_cols > n_cols:
        raise ValueError(f"There are more complete columns than existing columns: {n_complete_cols} vs {n_cols}")

    if len(missing_mechanism_array) != n_cols-n_complete_cols:
        raise ValueError(f"There is a mismatch between the number of missing mechanisms ({missing_mechanism_array}) and the number of wanted missing columns ({n_cols-n_complete_cols})")

    if len(missing_rate_array) != n_cols-n_complete_cols:
        raise ValueError(f"There is a mismatch between the number of missing rates ({missing_rate_array}) and the number of wanted missing columns ({n_cols-n_complete_cols})")


    missing_cols = data.columns[n_complete_cols:]
    
    for i, col in enumerate(missing_cols):  
        if missing_mechanism_array[i] == "MCAR":     # Missingness is random
            new_data[col] = apply_MCAR(new_data[col], missing_rate_array[i])

        elif missing_mechanism_array[i] == "MAR":    # Missingness depends on complete features
            new_data[col] = apply_MAR(new_data[col], new_data.iloc[:,:n_complete_cols], missing_rate_array[i], missingness_ascending)

        elif missing_mechanism_array[i] == "MNAR":   # Missingness depends on the values of the column (higher/lower values are more likely to be missing)       
            new_data[col] = apply_MNAR(new_data[col], missing_rate_array[i], missingness_ascending)
        else:
            raise ValueError(f"There is an unknown missing mechanism: {missing_mechanism_array[i]}")

    return new_data


def generate_dataset_with_missing_data(n_rows: int, n_cols: int, type_array: list[str], missing_mechanism_array: list[str], missing_rate_array: list[float], n_complete_cols: int = 1, missingness_ascending: bool = True):
    """
    Generates a dataset with missing data mechanisms. Applies the above functions for user usability.
    
    Parameters:
    n_rows (int): Indicates the number of rows the dataset must have
    n_cols (int): Indicates the number of columns the dataset must have
    type_array (list[str]): Indicates the type of each column
    missing_mechanism_array (list[str]): The list containing the missing mechanism that needs to be applied for each column (excluding the complete columns)
    missing_rate_array (list[float]): The missing rate of each column with missing mechanism applied
    n_complete_cols (int): The number of the first columns of the dataframe that won't have the missing mechanism applied
    missingness_ascending (bool): Indicates whether the highest/lowest values must become missing 
    
    Returns:
    new_data (pd.DataFrame): A copy of the dataframe with the missing data mechanisms applied
    """
    data = generate_synthetic_dataset(n_rows, n_cols, type_array)
    missing_dataset = apply_missing_data(data, missing_mechanism_array, missing_rate_array , n_complete_cols, missingness_ascending)
    
    return missing_dataset
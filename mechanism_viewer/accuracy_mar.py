"""These algorithms run models to obtain a prediction accuracy value.

By splitting the dataset into training and testing sets, and then training a predictive model,
we can evaluate how accurately the model predicts the missing values. As a consequence, an 
high prediction accuracy suggests that the missingness can be explained by the observed
variables, which indicates that the underlying mechanism is likely MAR.

For an average accuracy value that is 5% higher than the baseline ( max(1-missing_rate, missing_rate) ),
it is likely that underlying mechanism is MAR. In contrast, if the difference between the
average accuracy and the baseline is close to 0 or negative, then it is likely to be MCAR/MNAR.

Common Pitfalls and Limitations
The models works best at identifying MAR/MCAR/MNAR when the datasets have continuous data.
Other types of data may have a smaller difference between the missing mechanisms.
Non-numeric data cannot be used due to the nature of the models.
"""

import warnings
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

__all__ = [
    "run_random_forest",
    "run_logistic_regression",
    "detect_mar_from_model_accuracy",
]


def _validate_inputs(
    df: pd.DataFrame,
    column_name: str
    ) -> None:
    """
    Validate the dataset and column_name given as inputs.

    Parameters
    ----------
    df : pd.DataFrame,
        The dataset to be validated for training and testing a model.
    column_name : str
        The missing column to be used on the model.

    Returns
    -------
    This function does not return anything.
    """
    # Column_name (target column) is not in the dataset
    if column_name not in df.columns:       
        raise ValueError(f"{column_name} column was not found in dataframe.")
    # Column_name has no missing values
    if df[column_name].notna().all():       
        raise ValueError(f"{column_name} column does not have any missing values.")
    # Column_name has only missing values
    if df[column_name].isna().all():       
        raise ValueError(f"{column_name} column is full of missing values.")
    
    complete_columns_list = df.columns[df.notna().all()].tolist()
    non_numeric_cols = df[complete_columns_list].select_dtypes(exclude=["number"]).columns.tolist()

    # There are no complete columns (predictors)
    if not complete_columns_list:       
        raise ValueError(f"The given dataset does not have any complete column.")
    # There are predictors that have non-numeric data that the model cannot fit directly    
    if non_numeric_cols:       
        raise ValueError(f"The given dataset has complete column that have non-numeric data. \
                         Please remove the complete columns with non-numerical data, before using \
                         the function, since the model cannot fit the data directly.")

    return

def _validate_data_splitting(
    y_train: pd.Series,
    y_test: pd.Series
    ) -> None:
    """
    Warns the user if the dataset has too few examples of the minority class
    in the target column.

    A very small number of samples in any class can reduce the stability and
    reliability of the model's results.

    Parameters
    ----------
    y_train : pd.Series,
        The training samples of the target (y) that will be used to train
        the model.
    y_test : pd.Series
        The testing samples  of the target (y) that will be used to test
        the model's performance.

    Returns
    -------
    This function does not return anything.
    """
    min_elements_per_class = 5

    if y_train.value_counts().min() < min_elements_per_class:
        warnings.warn("Training set may be too small for stable training! \
                      Increase the dataset size to produce more reliable results.", UserWarning)

    if y_test.value_counts().min() < min_elements_per_class:
        warnings.warn("Test set may be too small for reliable evaluation! \
                      Increase the dataset size to produce more reliable results.", UserWarning)
    
    return


def _prepare_missingness_dataset(
    df: pd.DataFrame,
    column_name: str
    ) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare the dataset used to train and test a missingness prediction model.
    
    The function selects columns without missing values to be used as predictor
    variables (X) and creates a target variable (y) representing the missingness
    of the specified column.

    Parameters
    ----------
    df : pd.DataFrame,
        The dataset from which complete columns and the missingness of the
        specified column will be obtained.
    column_name : str
        The name of the column whose missingness will be used as the target
        variable in the model.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        The predictor variables (X) and the target variable (y).
    """
    complete_columns_list = df.columns[df.notna().all()].tolist()

    X = df[complete_columns_list].copy()
    y = df[column_name].isna().astype(int)

    return (X,y)


def _train_test_missingness_model(
    X: pd.DataFrame,
    y: pd.Series,
    model: BaseEstimator,
    model_name: str,
    print_result: bool,
    ) -> float:
    """
    Train and evaluate a model to predict missingness.

    The function splits the dataset into training and testing sets,
    fits the provided model on the training data, and evaluates its
    accuracy on the test data.
    
    Parameters
    ----------
    X : pd.DataFrame
        The predictor variables used to train the model.
    y : pd.Series
        The target variable representing the missingness indicator.
    model : BaseEstimator
        The machine learning model used to predict missingness.
    model_name : str
        The name of the model used for output message.
    print_result : bool
        The boolean indicating whether to print the result.

    Returns
    -------
    float
        The prediction accuracy of the model when train with the dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)       # Split the data into training set and testing set, that uses stratification so that both train and test sets contain all classes (0/1).

    _validate_data_splitting(y_train,y_test)

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)                      # Obtain the accuracy of the model
    
    if print_result:
        print(f"{model_name} Model Accuracy: {accuracy:.4f}")

    return accuracy


def run_random_forest(
    df: pd.DataFrame,
    column_name: str,
    print_result: bool = True,
    ) -> float:
    """
    Trains a Random Forest model to obtain the test accuracy value. The model uses
    100 estimators.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to train and test the model.
    column_name : str
        The missing column to be used on the model.
    print_result : bool = True
        The boolean indicating whether to print the result.
   
    Returns
    -------
    float
        The prediction accuracy of the model when train with the dataset.
    """
    _validate_inputs(df, column_name)

    X,y = _prepare_missingness_dataset(df, column_name)

    # Train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_accuracy = _train_test_missingness_model(X, y, model=rf_model, model_name="Random Forest", print_result=print_result)

    return rf_accuracy


def run_logistic_regression(
    df: pd.DataFrame,
    column_name: str,
    print_result: bool = True,
    ) -> float:
    """
    Trains a Logistic Regression model to obtain the test accuracy value.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to train and test the model.
    column_name : str
        The missing column to be used on the model.
    print_result : bool = True
        The boolean indicating whether to print the result.
   
    Returns
    -------
    float
        The prediction accuracy of the model when train with the dataset.
    """
    _validate_inputs(df, column_name)

    X,y = _prepare_missingness_dataset(df, column_name)

    # Train the Logistic Regression model
    lr_model = LogisticRegression(max_iter=1000)

    lr_accuracy = _train_test_missingness_model(X, y, model=lr_model, model_name="Logistic Regression", print_result=print_result)

    return lr_accuracy


def detect_mar_from_model_accuracy(
    df: pd.DataFrame,
    column_name: str,
    print_result: bool = True,
    ) -> float:
    """
    Calculates the Accuracy Baseline Difference (ABD), an heuristic that
    estimates whether the missingness mechanism of a column is likely MAR
    (Missing At Random) by comparing the average model accuracy with a baseline.

    The function trains a Random Forest and a Logistic Regression model to
    predict the missingness of the specified column. It then compares the
    average accuracy of these models with a baseline based on the missing
    rate of the column.

    If the model accuracy exceeds the baseline by more than 5 percentage
    points, it suggests that the missingness can be predicted from other
    variables, indicating that the mechanism is likely MAR. Otherwise, the
    missingness is more likely MCAR/MNAR.

    Formula
    -------
        ABD = ((RF_accuracy + LR_accuracy) / 2 - baseline) * 100

    where
    -----
        baseline = max(p, 1 - p)
        p = missing_rate(column_name)

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing the column with missing values.
    column_name : str
        The name of the column whose missingness mechanism will be assessed.
    print_result : bool = True
        The boolean indicating whether to print the results of the models and
        print the accuracy result of the most likely missing data mechanism.

    Returns
    -------
    float
        The Accuracy Baseline Difference (ABD), which represents the difference
        (in percentage points) between the average model accuracy and the 
        baseline accuracy.
        Result rounded to 2 decimal places.
    """
    rf_accuracy = run_random_forest(df, column_name, print_result)
    lr_accuracy = run_logistic_regression(df, column_name, print_result)

    missing_rate = df[column_name].isna().mean()
    baseline = max(1- missing_rate, missing_rate)

    accuracy_baseline_diff = round((((rf_accuracy + lr_accuracy)/2) - baseline)*100,2)

    if print_result:
        print(f"The target column {column_name} with missing rate of {missing_rate} gives an Accuracy Baseline Difference of {accuracy_baseline_diff}.")

        if accuracy_baseline_diff > 5:
            print(f"Thus, it is likely that the underlying mechanism of {column_name} is MAR.")
        else:
            print(f"Thus, it is likely that the underlying mechanism of {column_name} is MCAR/MNAR.")

    return accuracy_baseline_diff
"""These algorithms run models to obtain a prediction accuracy value.

By splitting the dataset into training and testing sets, and then training a predictive model,
we can evaluate how accurately the model predicts the missing values. As a consequence, an 
high prediction accuracy suggests that the missingness can be explained by the observed
variables, which indicates that the underlying mechanism is likely MAR.

For an average accuracy value that is 5% (which can be changed in parameter `threshold`) higher
than the baseline ( max(1-missing_rate, missing_rate) ), it is likely that underlying mechanism
is MAR. In contrast, if the difference between the average accuracy and the baseline is close
to 0 or negative, then it is likely to be MCAR/MNAR.

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

from ._random import DEFAULT_RANDOM_STATE

from ._utils import get_single_class

from ._validation import validate_dataframe, validate_missing_col


__all__ = [
    "run_random_forest",
    "run_logistic_regression",
    "detect_mar_from_model_accuracy",
    "interpret_accuracy_baseline_diff",
]


def _validate_inputs(
    df: pd.DataFrame,
    missing_col: str
    ) -> None:
    """
    Validate the dataset and missing_col given as inputs.

    Parameters
    ----------
    df : pd.DataFrame,
        The dataset to be validated for training and testing a model.
    missing_col : str
        The missing column to be used on the model.

    Returns
    -------
    This function does not return anything.
    """
    validate_dataframe(df)
    validate_missing_col(df, missing_col)
    
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


def _validate_single_class(
    y: pd.Series
    ) -> None:
    """
    Validate class composition before stratified train/test split.

    Parameters
    ----------
    y : pd.Series
        The binary target indicating the missingness (0 = observed, 1 = missing) of the missing column

    Returns
    -------
    This function does not return anything.
    """
    single_class = get_single_class(y)
    if single_class is not None:
        raise ValueError(f"Cannot perform stratified train/test split because there is only one class: {single_class}")

    class_counts = y.value_counts()
    if class_counts.min() < 2:
        raise ValueError("Cannot perform stratified train/test split because at least one class in the missingness target has fewer than 2 samples. "
                        f"Class counts: {class_counts.to_dict()}. Increase dataset size or adjust missingness rate.")
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
    missing_col: str
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
    missing_col : str
        The name of the column whose missingness will be used as the target
        variable in the model.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        The predictor variables (X) and the target variable (y).
    """
    complete_columns_list = df.columns[df.notna().all()].tolist()

    X = df[complete_columns_list].copy()
    y = df[missing_col].isna().astype(int)

    return (X,y)


def _train_test_missingness_model(
    X: pd.DataFrame,
    y: pd.Series,
    model: BaseEstimator,
    model_name: str,
    random_state: int = DEFAULT_RANDOM_STATE,
    print_result: bool = True,
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
    random_state : int, default = 42
        Seed used in stochastic model routines for reproducible results.
    print_result : bool = True
        The boolean indicating whether to print the result.

    Returns
    -------
    float
        The prediction accuracy of the model when train with the dataset.
    """

    _validate_single_class(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)       # Split the data into training set and testing set, that uses stratification so that both train and test sets contain all classes (0/1).

    _validate_data_splitting(y_train,y_test)

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)                      # Obtain the accuracy of the model
    
    if print_result:
        print(f"{model_name} Model Accuracy: {accuracy:.4f}")

    return accuracy


def run_random_forest(
    df: pd.DataFrame,
    missing_col: str,
    random_state: int = DEFAULT_RANDOM_STATE,
    print_result: bool = True,
    ) -> float:
    """
    Trains a Random Forest model to obtain the test accuracy value. The model uses
    100 estimators.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to train and test the model.
    missing_col : str
        The missing column to be used on the model.
    random_state : int, default = 42
        Seed used in stochastic model routines for reproducible results.
    print_result : bool = True
        The boolean indicating whether to print the result.
   
    Returns
    -------
    float
        The prediction accuracy of the model when train with the dataset.
    """
    _validate_inputs(df, missing_col)

    X,y = _prepare_missingness_dataset(df, missing_col)

    # Train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state)

    rf_accuracy = _train_test_missingness_model(X, y, model=rf_model, model_name="Random Forest", random_state=random_state, print_result=print_result)

    return rf_accuracy


def run_logistic_regression(
    df: pd.DataFrame,
    missing_col: str,
    random_state: int = DEFAULT_RANDOM_STATE,
    print_result: bool = True,
    ) -> float:
    """
    Trains a Logistic Regression model to obtain the test accuracy value.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to train and test the model.
    missing_col : str
        The missing column to be used on the model.
    random_state : int, default = 42
        Seed used in stochastic model routines for reproducible results.
    print_result : bool = True
        The boolean indicating whether to print the result.
   
    Returns
    -------
    float
        The prediction accuracy of the model when train with the dataset.
    """
    _validate_inputs(df, missing_col)

    X,y = _prepare_missingness_dataset(df, missing_col)

    # Train the Logistic Regression model
    lr_model = LogisticRegression(max_iter=1000)

    lr_accuracy = _train_test_missingness_model(X, y, model=lr_model, model_name="Logistic Regression", random_state=random_state ,print_result=print_result)

    return lr_accuracy


def interpret_accuracy_baseline_diff(
    accuracy_baseline_diff: float,
    threshold: float = 5.0
    ) -> str:
    """
    Return the interpretation of an Accuracy Baseline Difference (ABD) value
    in a human readable way.

    Parameters
    ----------
    accuracy_baseline_diff : float
        The Accuracy Baseline Difference (ABD) value
    threshold : float, default = 5.0
        Threshold (in percentage points) used to classify likely MAR vs MCAR/MNAR

    Returns
    -------
    str
        The string containing a human readable interpretation of the ABD value.
    """
    abd_str = (f"Since the obtained Accuracy Baseline Difference is {round(accuracy_baseline_diff, 2)}, "
          f"and the given threshold is {threshold}, ")

    if accuracy_baseline_diff > threshold:
        return(abd_str + f"thus, it is likely that the underlying mechanism of the missing column is MAR.")
    else:
        return(abd_str + f"thus, it is likely that the underlying mechanism of the missing column is MCAR/MNAR.")


def detect_mar_from_model_accuracy(
    df: pd.DataFrame,
    missing_col: str,
    threshold: float = 5.0,
    random_state: int = DEFAULT_RANDOM_STATE,
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

    If the model accuracy exceeds the baseline by more than the threshold,
    (in percentage points), it suggests that the missingness can be predicted
    from other variables, indicating that the mechanism is likely MAR. Otherwise,
    the missingness is more likely MCAR/MNAR.

    Formula
    -------
        ABD = ((RF_accuracy + LR_accuracy) / 2 - baseline) * 100

    where
    -----
        baseline = max(p, 1 - p)
        p = missing_rate(missing_col)

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing the column with missing values
    missing_col : str
        The name of the column whose missingness mechanism will be assessed
    threshold : float, default = 5.0
        Threshold (in percentage points) used to classify likely MAR vs MCAR/MNAR
    random_state : int, default = 42
        Seed used in stochastic model routines for reproducible results
    print_result : bool = True
        The boolean indicating whether to print the results of the models and
        print the accuracy result of the most likely missing data mechanism

    Returns
    -------
    float
        The Accuracy Baseline Difference (ABD), which represents the difference
        (in percentage points) between the average model accuracy and the 
        baseline accuracy.
        Result rounded to 2 decimal places.
    """
    rf_accuracy = run_random_forest(df, missing_col, random_state, print_result)
    lr_accuracy = run_logistic_regression(df, missing_col, random_state, print_result)

    missing_rate = df[missing_col].isna().mean()
    baseline = max(1- missing_rate, missing_rate)

    accuracy_baseline_diff = round((((rf_accuracy + lr_accuracy)/2) - baseline)*100,2)

    if print_result:
        print(f"The target column {missing_col} with missing rate of {missing_rate} gives an Accuracy Baseline Difference of {accuracy_baseline_diff}.")
        print(interpret_accuracy_baseline_diff(accuracy_baseline_diff, threshold))

    return accuracy_baseline_diff
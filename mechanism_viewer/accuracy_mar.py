import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

"""
These algorithms run modelts to obtain an prediction accuracy value.

By splitting the dataset into training and testing sets, and then training a predictive model,
we can evaluate how accurately the model predicts the missing values. As a consequence, an 
high prediction accuracy suggests that the missingness can be explained by the observed
variables, which indicates that the underlying mechanism is likely MAR.

For an accuracy that is 5% higher than the baseline ( max(1-missing_rate, missing_rate) ),
it is likely that underlying mechanism is MAR. If a negative accuracy value is achieved, or 
a value close to 0, then it is likely to be MCAR/MNAR.

Common Pitfalls and Limitations
The models works best at identifying MAR/MCAR/MNAR when the datasets has continuous data.
Other types of data may have a smaller diference between the missing mechanisms.
"""

def run_random_forest(df: pd.DataFrame, column_name: str):
    """
    Trains a Random Forest model to obtain the test accuracy value. The model uses
    100 estimators.
    
    Parameters:
    df (pd.DataFrame): The dataset to be used to train and test the model
    column_name (str): The missing column to be used on the model.
   
    Returns:
    accuracy (int): The prediction accuracy of the model when train with the dataset.
    It is rounded to the four decimal places.
    """
    if column_name not in df.columns:
        raise ValueError(f"{column_name} not found in dataframe.")

    columns_without_na = df.columns[df.notna().all()].tolist()
    new_df = df[columns_without_na].copy()

    new_df["column_name_missingness"] = df[column_name].isna().astype(int)

    X = new_df[columns_without_na]
    y = new_df["column_name_missingness"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   # Split the data into training set and testing set

    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)          # Obtain the accuracy of the model
    print(f"Random Forest Model Accuracy: {accuracy:.4f}")

    return round(accuracy, 4)


def run_logistic_regression(df: pd.DataFrame, column_name: str):
    """
    Trains a Logistic Regression model to obtain the test accuracy value.
    
    Parameters:
    df (pd.DataFrame): The dataset to be used to train and test the model
    column_name (str): The missing column to be used on the model.
   
    Returns:
    accuracy (int): The prediction accuracy of the model when train with the dataset.
    It is rounded to the four decimal places.
    """

    if column_name not in df.columns:
        raise ValueError(f"{column_name} not found in dataframe.")

    columns_without_na = df.columns[df.notna().all()].tolist()
    new_df = df[columns_without_na].copy()

    new_df["column_name_missingness"] = df[column_name].isna().astype(int)

    X = new_df[columns_without_na]
    y = new_df["column_name_missingness"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)    # Split the data into training set and testing set

    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)          # Obtain the accuracy of the model
    print(f"Logistic Regression Model Accuracy: {accuracy:.4f}")

    return round(accuracy, 4)


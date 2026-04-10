"""These algorithms run the Little's MCAR Test.

Little's MCAR (Missing Completely At Random) test evaluates whether the probability of data being 
missing is unrelated to observed or unobserved data. When data is MCAR, missingness does not
introduce bias, making it safe to use techniques like listwise deletion.

Key Assumptions of Little's MCAR Test
    - Data missingness is random across all variables.
    - The sample is representative of the population under study.
    - Sufficient sample size exists to compute reliable statistics.

Common Pitfalls and Limitations
    - Low Power: Small sample sizes or subtle data patterns can lead to inconclusive results.
    - Misinterpretation: A non-significant result does not confirm data is MCAR;
    it only indicates insufficient evidence against the null hypothesis.
    - Numerical Data Dependency: The test is designed for numerical data, limiting its
    applicability to categorical datasets.

Best Practices
    - Use covariates relevant to the context of missing data.
    - Ensure adequate sample size to boost test reliability.
    - Use visualization tools like missingness heatmaps for preliminary insights.

Reference
---------
https://www.linkedin.com/pulse/understanding-littles-mcar-test-comprehensive-guide-data-debasish-deb-fbmpf

Documentation: https://rianneschouten.github.io/pyampute/build/html/pyampute.exploration.html
"""

import warnings
import pandas as pd
from pyampute.exploration.mcar_statistical_tests import MCARTest
import matplotlib.pyplot as plt
import seaborn as sns

from ._validation import validate_dataframe

__all__ = [
    "little_mcar_test",
    "interpret_mcar_p_value",
    "little_mcar_pairwise",
    "plot_mcar_pairwise",
]


def _validate_input(
    df: pd.DataFrame
    ) -> None:
    """
    Validate the dataset given as input.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be validated for running the Little's MCAR Test.
   
    Returns
    ------- 
    This function does not return anything.
    """
    validate_dataframe(df)
    
    non_numeric_cols = df.select_dtypes(exclude="number").columns
    if len(non_numeric_cols) > 0:
        non_num_cols = ", ".join(map(str, non_numeric_cols))
        raise ValueError("The dataset has non-numeric data, which can hinder the test results. " \
                        "Please remove the columns with non-numerical data, before using \
                         the function.\n"
                        f"Existing non-numeric columns inside the dataframe: {non_num_cols}.")
    return


def _prepare_missingness_dataset(
    df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Prepare the given dataset to be used for running the Little's MCAR Test.

    Since Little's MCAR Test tries to detect whether the missingness of the columns is due
    to random chance or not, this function keeps the complete columns and the columns that 
    have at least two missing values and two non-missing values. The Little's MCAR Test
    requires two columns minimum that fit the last criteria. In other words, having at least
    two columns that fit that criteria ensures the model can evaluate the missingness effectively. 

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be validated for running the Little's MCAR Test.
   
    Returns
    ------- 
    pd.DataFrame
        The filtered dataset with only columns that have missing values.
    """
    min_group = 2
    n_rows = len(df)
    missing_counts = df.isnull().sum()

    valid_cols = missing_counts[
        (missing_counts >= min_group) & (missing_counts <= n_rows - min_group)
    ].index.tolist()

    complete_cols = missing_counts[
        missing_counts == 0
    ].index.tolist()

    if len(valid_cols) < 2:
        raise ValueError("Not enough columns with sufficient missing values to run Little's MCAR test. "
                         "At least two columns must each contain >=2 missing and >=2 non-missing values.")

    eligible_cols = complete_cols + valid_cols

    return df[eligible_cols]
    

def _validate_output(
    p_values: pd.DataFrame
    ) -> None:
    """
    Validate the pairwise Little's MCAR output matrix.

    Parameters
    ----------
    p_values : pd.DataFrame
        The output obtained from the Little's MCAR Test.
   
    Returns
    ------- 
    This function does not return anything.
    """
    if p_values.empty:
        raise ValueError("The resulting p_values matrix is empty. No valid comparisons could be made.")
    
    return


def interpret_mcar_p_value(
    p_value: float,
    alpha: float = 0.05
    ) -> str:
    """
    Return the results of the Little's MCAR chi-square test in a human readable way,
    including the potential missing data mechanism in the dataset.

    Parameters
    ----------
    p_value : float
        The p-value obtained from the Little's MCAR chi-square test
    alpha : float, default = 0.05
        The alpha value that will be used for the rejection of the null hypothesis
    
    Returns
    ------- 
    The string containing a human readable interpretation of a Little's MCAR p-value.
    """
    p_value_str = (f"\nObtained p_value is {round(p_value,4)}.\n\nInterpretation: ")

    if p_value > alpha:
        return(p_value_str + "Fail to reject the null hypothesis. Data is likely MCAR")
    else:
        return(p_value_str + "Reject the null hypothesis. Data is not MCAR (likely MAR or MNAR)")


def plot_mcar_pairwise(
    p_values: pd.DataFrame,
    alpha: float = 0.05,
    display_plot: bool = False,
    ) -> tuple[plt.Figure, plt.Axes, plt.Figure, plt.Axes]:
    """
    Return the plots of p-values obtained from Little's MCAR t-test, including the
    potential missing data mechanism in the dataset. The plots are available for display.

    Parameters
    ----------
    p_values : pd.DataFrame
        The output obtained from the Little's MCAR t-test
    alpha : float, default = 0.05
        The alpha value that will be used for the rejection of the null hypothesis
    display_plot : bool, default = False
        If True, displays figures with ``plt.show()``
   
    Returns
    ------- 
    tuple
        (fig_p_values, ax_p_values, fig_reject, ax_reject) representing the 2 plots available for display.
    """
    _validate_output(p_values)

    fig_p_values, ax_p_values = plt.subplots(figsize=(6, 6))
    sns.heatmap(p_values, annot=True, fmt=".4f", cmap="coolwarm_r", center=alpha, cbar_kws={"label": "p-value"}, ax=ax_p_values)
    ax_p_values.set_title("p_value of t-test for every pair of columns")
    ax_p_values.set_xlabel("Note: white square = No p-value\n(column is complete or square belongs to diagonal)\n grey square = reject the null hypothesis (Data is not MCAR)", labelpad=15)
    fig_p_values.tight_layout()

    reject = p_values <= alpha

    fig_reject, ax_reject = plt.subplots(figsize=(8, 5))
    sns.heatmap(reject, cmap=["#d26256", "#2ecc71"], cbar=False, linewidths=0.75, ax=ax_reject)
    ax_reject.set_title("Pairwise MCAR test rejections")
    ax_reject.set_xlabel("Note: green square = evidence against MCAR\nred square = fail to reject null hypothesis (likely MCAR)", labelpad=15)
    fig_reject.tight_layout()

    if display_plot:
        plt.show()
    else:
        plt.close(fig_p_values)
        plt.close(fig_reject)

    return fig_p_values, ax_p_values, fig_reject, ax_reject


def little_mcar_test(
    df: pd.DataFrame
    ) -> float:
    """
    Runs the Little's MCAR Test. It performs a chi-square test on the entire dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to run the test

    Returns
    -------
    float 
        The p-value of the test based on the entire dataset
    """
    _validate_input(df)

    df_filtered = _prepare_missingness_dataset(df)

    mt = MCARTest(method="little")

    p_value = mt.little_mcar_test(df_filtered)

    return p_value


def little_mcar_pairwise(
    df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Runs the Little's MCAR Test. It performs a separate t-test for every pair of columns possible.
    Since each t-test returns a p-value, the function returns every pair combination and its p-value.
    
    Notes
    -----
    This function does not plot or print. Use ``plot_mcar_pairwise`` to visualize
    the returned matrix.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to run the test
   
    Returns
    -------
    p_values (pd.DataFrame): A matrix dataframe with every p-value of each pair combination
    """
    _validate_input(df)

    df_filtered = _prepare_missingness_dataset(df)

    mt = MCARTest(method="ttest")
    with warnings.catch_warnings():
        warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module=r"pyampute\..*",
        )
        p_values = mt.mcar_t_tests(df_filtered)

    _validate_output(p_values)

    return p_values
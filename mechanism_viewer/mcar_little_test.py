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
import numpy as np

__all__ = [
    "run_little_test",
    "run_little_test_pairs"
]



def run_little_test(
    df: pd.DataFrame
    ) -> np.float64:
    """
    Runs the Little's MCAR Test. It performs a chi-square test on the entire dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to run the test
   
    Returns
    -------
    p_value (np.float64): The p-value of the test based on the entire dataset
    """

    mt = MCARTest(method="little")

    p_value = mt.little_mcar_test(df)

    print(f"\nObtained p_value is {round(p_value,4)}.\n\nConclusion:")

    if p_value > 0.05:
        print("Fail to reject the null hypothesis: Data is likely MCAR")
    else:
        print("Reject the null hypothesis: Data is not MCAR (likely MAR or MNAR)")

    return p_value


def run_little_test_pairs(
    df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Runs the Little's MCAR Test. It performs a separate t-tests for every pair of columns possible.
    Since each t-test returns a p-value, the function returns every pair combination and its p-value.
    The function also displays two plots to better visualize the p-value matrix. The first one is an
    heatmap presenting every p-value number in the matrix. The second one illustrates the matrix by
    only using the colors to represent whether each p-value makes the null hypothesis rejected or not.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used to run the test
   
    Returns
    -------
    p_values (pd.DataFrame): A matrix dataframe with every p-value of each pair combination
    """

    if df.empty:
        print("The function run_little_test_pairs cannot run with an empty pd.DataFrame as input.")
        return pd.DataFrame()

    # Keep only columns that have at least 2 missing AND 2 non-missing values,
    # otherwise ttest_ind receives an empty/single-element group and returns NaN
    # with a SmallSampleWarning.
    min_group = 2
    n = len(df)
    missing_counts = df.isnull().sum()
    valid_cols = missing_counts[
        (missing_counts >= min_group) & (missing_counts <= n - min_group)
    ].index.tolist()

    if len(valid_cols) < 2:
        print("Not enough columns with sufficient missing values to run pairwise t-tests.")
        return pd.DataFrame()

    df_filtered = df[valid_cols]

    mt = MCARTest(method="ttest")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_values = mt.mcar_t_tests(df_filtered)

    if p_values.empty:
        print("The resulting p_values matrix is empty. No valid comparisons could be made.")
        return p_values

    plt.figure(figsize=(6, 6))
    sns.heatmap(p_values, annot=True, fmt=".4f", cmap="coolwarm_r", center=0.05, cbar_kws={"label": "p-value"})
    plt.title("p_value of t-test for every pair of columns")
    plt.xlabel("Note: white square = No p-value\n(column is complete or square belongs to diagonal)\n grey square = reject the null hypothesis (Data is not MCAR)", labelpad=15)
    plt.tight_layout()
    plt.show()

    reject = p_values <= 0.05

    plt.figure(figsize=(8, 5))
    sns.heatmap(reject, cmap=["#d26256", "#2ecc71"], cbar=False, linewidths=0.75)
    plt.title("MCAR test rejections")
    plt.xlabel("Note: green square = evidence against MCAR\nred square = fail to reject null hypothesis (likely MCAR)", labelpad=15)
    plt.tight_layout()
    plt.show()

    return p_values
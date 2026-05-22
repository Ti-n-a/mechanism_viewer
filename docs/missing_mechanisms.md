# Missing-Data Mechanisms

Understanding why data is missing is critical before choosing an imputation strategy.
Rubin discussed this issue in 1976[^1] and explored how missing values arise.
He distinguished three patterns of missingness, which became known as missing-data mechanisms.
Those are:

[^1]: D. B. Rubin, Biometrika 63, 581 (1976).

- MCAR (Missing Completely At Random)
- MAR (Missing At Random)
- MNAR (Missing Not At Random)

!!! warning "Important caveat"

    Missing-data mechanisms are assumptions about how the missing values were generated, based on the available data and knowledge domain. Nevertheless, in real datasets, the missing patterns can be mixture of different missing-data mechanisms. For example, columns can be partly MAR and partly MNAR. Thus, treat conclusions as interpretations of the available information rather than absolute truth.

## MCAR

A MCAR pattern indicates that the missing values do not depend on any observed or unobserved data.

Definition:
$$
P(R \mid X_{\text{obs}}, X_{\text{miss}}) = P(R)
$$

Intuition:

- Missing values occur randomly throughout the dataset.
- Rows with missing values are similar to rows without missing values.

Example:

*A sensor randomly fails to record data 5% of the time, regardless of the values of any variables.*

Implications:

- Using only rows without missing values is generally less prone to bias, although it reduces the amount of available data.
- Simple imputation methods can be acceptable depending on the task.


## MAR

MAR means missingness depends on the observed variables, but does not depend on the missing values themselves.

Definition:
$$
P(R \mid X_{\text{obs}}, X_{\text{miss}}) = P(R \mid X_{\text{obs}})
$$


Intuition:

- Missingness is predictable from the columns existing in the dataset.

Example:

*In social surveys, personal information such as age, education, and occupation is collected. The income of a person is often missing for younger respondents because many may not yet be employed.*

Implications:

- Imputation methods that predict missing values from observed data estimate them reasonably well. However, it is important to select the variables that help explain why the missingness occurs.
- Using the relevant observed features with the missing column improves the quality of the data analysis. 
- Domain knowledge can be used to detect observed variables that explain the missingness.


## MNAR

MNAR describes the missingness that depends on unobserved data. For example, features that were not recorded, or even the actual values of the missing column where entries are missingt.

Definition:
$$
P(R \mid X_{\text{obs}}, X_{\text{miss}}) \neq P(R \mid X_{\text{obs}})
$$

Intuition:

- The observed columns do not help understand how the missingness occurs.
- The missigness was not generated due to random events or measurement issues.

Example:

*People with very high income are less likely to report income.*

*In a health survey, patients with high stress levels may be less likely to answer questions about sleep quality. If stress level was not recorded in the dataset, then the missingness in sleep quality depends on an unobserved variable.*

Implications:

- Imputations algorithms are very proned to bias when not taking into consideration the unobserved data.
- It is essential to test different assumptions and use domain knowledge, because the missing values may not behave like the observed values.


## Mechanism Comparison

| Mechanism | Depends on observed data? | Depends on unobserved features or its missing values? | How to process the data? |
|-----------|---------------------------|-------------------------------------------------------|---|
| MCAR | No | No | Simpler methods to address missing values are often acceptable |
| MAR | Yes | No | Imputation should be based on relevant observed features |
| MNAR | No | Yes | Using domain knowledge in sensitivity analysis is advised before making decisions |


## How ``mechanism_viewer`` helps

The package provides diagnostics tools for:

- Inspecting the rate, structure and pattern of missingness.
- Evaluate MCAR plausibility on numeric data.
- Check whether missingness is predictable from observed data.
- Explore whether imputation shifts the distribution of the missing column.
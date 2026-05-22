# Missing-Data Mechanisms

Understanding why data is missing is critical before choosing an imputation strategy.
In practice, the missingness of a column is usually discussed through three mechanisms:

- MCAR (Missing Completely At Random)
- MAR (Missing At Random)
- MNAR (Missing Not At Random)

## MCAR

MCAR means the probability that a value is missing does not depend on any observed or unobserved data.

Intuition:

- Missing values occur randomly throughout the dataset.
- Rows with missing values are similar to rows without missing values.

Example:

- A sensor randomly fails to record data 5% of the time, regardless of the values of any variables.

Implications:

- Using only rows without missing values is generally less prone to bias, although it reduces the amount of available data.
- Simple imputation methods can be acceptable depending on the task.

## MAR

MAR means missingness can depend on observed variables, but not on the missing value itself after conditioning on observed data.

Intuition:

- Missingness is predictable from columns you already have.

Example:

- Income is more often missing for younger respondents, and age is observed.

Implications:

- Model-based imputations (for example, MICE) are often reasonable.
- Including informative observed predictors is important.

## MNAR

MNAR means missingness depends on unobserved information, often including the missing value itself.

Intuition:

- Even after using observed columns, missingness still relates to what you do not see.

Example:

- People with very high income are less likely to report income.

Implications:

- Standard MAR-based imputations may be biased.
- Sensitivity analysis and domain assumptions become essential.

## Quick Comparison

| Mechanism | Depends on observed data? | Depends on unobserved/missing value? | Typical handling |
|---|---|---|---|
| MCAR | No | No | Simpler methods often acceptable |
| MAR | Yes | Not after conditioning on observed data | Model-based imputation (for example, MICE) |
| MNAR | Yes/No | Yes | Sensitivity analysis and explicit assumptions |

## How mechanism_viewer helps

The package provides complementary diagnostics; no single test is definitive.

- Visual tools: inspect structure and pattern of missingness.
- Little's MCAR test: evaluate MCAR plausibility on numeric data.
- MAR heuristic (ABD): check whether missingness is predictable from observed data.
- Imputation comparison plots: inspect whether imputations preserve plausible distributions.

Recommended approach:

1. Start with visual diagnostics.
2. Run Little's test for MCAR plausibility.
3. Run MAR heuristic for key missing columns.
4. Combine evidence with domain knowledge.

## Important caveat

Mechanisms are assumptions about the data-generating process. In real datasets, missingness can be mixed (for example, partly MAR and partly MNAR across columns). Treat conclusions as evidence-based, not absolute truth.
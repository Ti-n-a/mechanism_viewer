# Workflow

This page explains the typical workflow when using `mechanism_viewer` to diagnose a dataset with missing values.



## Typical Workflow

1. Load or generate a dataset.
2. Inspect missing rates and row-level patterns.
3. Explore how missingness relates to observed columns.
4. Form hypotheses about whether the observed missingness patterns are MCAR, MAR, or MNAR.
5. Test hypotheses about the missingness mechanism.
6. Use imputation diagnostics as additional evidence for difficult cases.
7. Combine all evidences before concluding.


## Workflow Example

### 1. Build a synthetic dataset

Use ColType values to define column behavior.

```python
import mechanism_viewer as mv
from mechanism_viewer import ColType

df = mv.generate_synthetic_dataset(
    n_rows=500,
    type_array=[
        ColType.CONTINUOUS,
        ColType.DISCRETE,
        ColType.BINARY,
        ColType.CONTINUOUS,
    ],
    random_state=7,
)
```

### 2. Inject missingness by mechanism

The first n_complete_cols remain complete and can be used as predictors/dependencies.

```python
df_missing = mv.apply_missing_data(
    df=df,
    n_complete_cols=2,
    missing_mechanism_array=["MAR", "MNAR"],
    missing_rate_array=[0.20, 0.25],
    missingness_ascending=True,
    random_state=7,
)
```

### 3. Read missingness structure visually

Start with global views:

```python
mv.plot_missing_rate(df_missing, display_plot=True)
mv.rows_with_similar_missing(df_missing, display_plot=True)
```

Then inspect dependencies:

```python
mv.visualize_column_dependencies(df_missing, sort_by_complete=True, display_plot=True)
mv.missing_rate_matrix(df_missing, column_name="Col1", display_plot=True)
```

### 4. Estimate MAR plausibility

The package provides an Accuracy Baseline Difference (ABD) heuristic.

```python
abd = mv.test_mar_from_model_accuracy(df_missing, missing_col="Col3")
print("ABD:", abd)
print(mv.interpret_mar_abd(abd, threshold=5.0))
```

Interpretation rule:

- ABD > threshold: likely MAR.
- ABD <= threshold: likely MCAR or MNAR.

### 5. Test MCAR hypothesis (numeric data)

Little's test operates on numeric data and needs enough rows per pattern.

```python
numeric_df = df_missing.select_dtypes(include="number")
p_value = mv.little_mcar_test(numeric_df)
print(mv.interpret_mcar_p_value(p_value, alpha=0.05))
```

Pairwise matrix version:

```python
pvals = mv.little_mcar_pairwise(numeric_df)
mv.plot_mcar_pairwise(pvals, alpha=0.05, display_plot=True)
```

### 6. Inspect imputation behavior

Use complete columns as predictors and compare observed vs imputed values.

```python
mv.scatterplot_imputation_comparison(
    df_missing,
    column_name="Col1",
    missing_col="Col3",
    missing_col_type="continuous",
    display_plot=True,
)
```

Distribution-level check:

```python
fig, ax, imputed_df = mv.plot_imputation_distribution(
    df_missing,
    missing_col="Col3",
    missing_col_type="continuous",
    display_plot=True,
)
```
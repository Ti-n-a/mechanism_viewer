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

### 1. Generate a synthetic dataset

Use ColType values to define the column type.

```python
import mechanism_viewer as mv
from mechanism_viewer import ColType

df = mv.generate_synthetic_dataset(n_rows=500, type_array=[ColType.CONTINUOUS, ColType.BINARY,
                                ColType.DISCRETE, ColType.CONTINUOUS], random_state=7)
```

### 2. Inject missingness onto the dataset

The first `n_complete_cols` remain complete and can be used as dependencies for applying MAR missingness to columns.

```python
df_missing = mv.apply_missingness(df=df, n_complete_cols=2, missing_mechanism_array=["MAR", "MCAR"],
                                    missing_rate_array=[0.20, 0.25], missingness_ascending=True,
                                    random_state=7)
```

### 3. View missingness structure

Start with general diagnostic tools.

```python
mv.plot_missing_rate(df_missing, display_plot=True)
mv.upset_missing_rows(df_missing, display_plot=True)
```

Then, inspect the missingness dependencies.

```python
mv.plot_missingness_distribution(df_missing["Col1","Col2","Col4"], missing_col="Col3",
                                    display_plot=True)
mv.plot_missingness_distribution(df_missing["Col1","Col2","Col3"], missing_col="Col4",
                                    display_plot=True)
mv.missing_rate_matrix(df_missing, column_name="Col1", display_plot=True)
mv.missing_rate_matrix(df_missing, column_name="Col2", display_plot=True)
```

### 4. Estimate MAR plausibility

The package provides an Accuracy Baseline Difference (ABD) heuristic, where:

- ABD > threshold: likely MAR.
- ABD <= threshold: likely MCAR or MNAR.

```python
abd = mv.test_mar_from_model_accuracy(df_missing, missing_col="Col3")
print(mv.interpret_mar_abd(abd, threshold=5.0))
```

### 5. Test MCAR hypothesis

Little's test operates on numeric data and needs enough rows per pattern.

```python
p_value = mv.little_mcar_global(df_missing)
interpretation_str = mv.interpret_mcar_p_value(p_value, alpha=0.05)
print(interpretation_str)
```

There is also the pairwise version of the Little's test, where each
pair of colums is tested.

```python
p_value_matrix = mv.little_mcar_pairwise(df_missing)
mv.plot_mcar_pairwise(p_value_matrix, alpha=0.05, display_plot=True)
```

### 6. Inspect imputation behavior

Use complete columns as predictors and compare observed vs imputed values.

```python
mv.scatter_imputation_comparison(df_missing, column_name="Col1", missing_col="Col3",
                                    missing_col_type="continuous", display_plot=True)
```

Visualize distribution shift with imputed values added. Moreover, the imputed dataset can
be obtained for further analysis.

```python
fig, ax, imputed_df = mv.plot_imputation_distribution(df_missing, missing_col="Col3",
                                    missing_col_type="continuous", display_plot=True)
```

### 7. Check evidences

Gather insights from the results and formulate conclusions that align with both domain knowledge and the observed patterns.

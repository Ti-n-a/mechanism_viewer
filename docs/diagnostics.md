# Diagnostic Overview

This page summarizes the main diagnostic tools in `mechanism_viewer`.

## General Missingness Diagnostics

- `plot_missing_rate`: heatmap with the missing rate of each column.
- `upset_missing_rows`: UpSet similar plot counting the row-level missing combinations.

## Missingness Compared to Complete Columns

- `plot_missingness_distribution`: plot distribution of an observed column based on the missingness of a column.
- `visualize_column_dependencies`: visualization of the value itensity of complete columns and the missingness indicator of missing rows.
- `missing_rate_matrix`: plot missig rates of columns based on the values of one observed column.
- `scatter_missingness_comparison` and `scatter_missingness_comparison_line`: plot the missingness indicator against another column.
- `boxplot_comparison`: use the missingness indicator to compare the distribution of an observed column.

## Correlation Plots

- `missingness_misscol_corr`: correlation of the missingness indicator among columns with missing values.
- `value_misscol_corr`: value correlation among columns that contain missing values.
- `complete_and_misscol_corr`: correlation between the values of complete columns and the missingness indicator of missing columns.
- `misscol_vs_all_corr`: correlation between the missingness indicator of a missing column and the values of all other columns.

## Statistical Tests and MAR Heuristic

- `little_mcar_global` and `interpret_mcar_p_value`: global Little's MCAR test via `pyampute` and its text interpretation.
- `little_mcar_pairwise` and `plot_mcar_pairwise`: pairwise MCAR checks and visual summaries.
- `run_random_forest` and `run_logistic_regression`: compute model accuracy in predicting the missingness.
- `test_mar_from_model_accuracy`: heuristic that calculates Accuracy Baseline Difference (ABD).
- `interpret_mar_abd`: text interpretation for ABD values.

## Imputation Diagnostics

- `scatter_imputation_comparison`: compare observed and imputed values against an observed column.
- `plot_imputation_distribution`: compare original and imputed distributions.

## Function Requirements

- Public functions expect a pandas DataFrame where applicable.
- Many diagnostics require at least one column containing missing values.
- MAR accuracy and several imputers require complete predictor columns to be numeric.
- `little_mcar_global` and `little_mcar_pairwise` require numeric columns and sufficient missing/non-missing counts.
- Imputation helpers require the correct `missing_col_type` using `ColType`.

If the input requirements are not met, a ValueError or TypeError may be raised during execution.

## Final Notes

No single plot or test proves the missingness is based on MCAR, MAR, or MNAR. Use the package to build a conclusion step by step, by checking whether different diagnostics point toward the same missing-data mechanism, while keeping in mind that each method has assumptions, limitations, and potential sources of bias.

`mechanism_viewer` is an exploratory package, and interpretations of its results should always be made with caution.

Furthermore, users can also use other plots, analyses, or domain knowledge when they are useful. The goal should be to combine multiple sources of evidence and reach the most reliable conclusion possible.

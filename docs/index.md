# mechanism_viewer

**`mechanism_viewer` is a Python package for diagnosing missing-data mechanisms in tabular datasets.**

Use it to investigate whether missingness is likely:

- MCAR (Missing Completely At Random)
- MAR (Missing At Random)
- MNAR (Missing Not At Random)

The package combines visualization tools, diagnostics based on models, Little's MCAR test, and synthetic data generation.

## Why use it

Imputation choices and downstream model validity depend on the missingness mechanism.
`mechanism_viewer` gives a practical diagnostic toolbox to help decide how to handle missing data.

## Main features

- Generate synthetic datasets with configurable column types and missingness patterns.
- Apply MCAR, MAR, and MNAR mechanisms in individual columns.
- Visualize missingness structure, dependence, and row-level patterns.
- Compare observed and imputed distributions.
- Estimate whether MAR is plausible using model accuracy heuristics.
- Run Little's MCAR test and pairwise MCAR checks.

## Installation

1. Download the package from `https://github.com/Ti-n-a/mechanism_viewer`

2. Place the package in any folder, as long as you remember the path.

> Note: The path to be taken is the first folder from the package, not `mechanism_viewer` folder that is inside the package next to README.

3. Open the terminal and execute the following command.

> Note: The command will install the package in non-editable mode. In other words, if any changes are made to the package, you need to reinstall the package again.

```python
python -m pip install <path_of_package>/mechanism_viewer
```

```python
# If you are already inside the python environment
pip install <path_of_package>/mechanism_viewer
```

As an alternative, you may install the package in editable mode, if you wish to edit the package internally.

```python
# To execute command outside the path
python -m pip install -e <path_of_package>/mechanism_viewer
```

```python
# For example, inside a Jupyter Notebook file on Windows
%pip install -e "<path_of_package>\mechanism_viewer"
```

> Note: In this mode, it is only necessary to reload the kernel to have the most updated version of the package.

## Quick Start

```python
import mechanism_viewer as mv
from mechanism_viewer import ColType

# 1) Generate a complete synthetic dataset
df = mv.generate_synthetic_dataset( n_rows=300, type_array=[ColType.CONTINUOUS, ColType.CONTINUOUS,
															ColType.DISCRETE, ColType.BINARY,
															ColType.CONTINUOUS])

# 2) Inject missingness in the last 3 columns
df_missing = mv.apply_missingness( df=df, n_complete_cols=2,
									missing_mechanism_array=["MAR", "MCAR", "MNAR"],
									missing_rate_array=[0.20, 0.15, 0.25],
									missingness_ascending=True)

# 3) Run diagnostics
fig, ax = mv.plot_missing_rate(df_missing, display_plot=True)
fig_corr, ax_corr = mv.complete_and_misscol_corr(df_missing, display_plot=True)

# 4) Test MCAR hypothesis on whole dataset
p_value = mv.little_mcar_global(df_missing)
interpretation = mv.interpret_mcar_p_value(p_value)
print(interpretation)

# 5) Test MAR plausibility for Col3
abd = mv.test_mar_from_model_accuracy(df_missing, missing_col="Col3")
interpretation_abd = mv.interpret_mar_abd(abd)
print(interpretation_abd)
```

## Next Steps

1. Read [Missing Mechanisms](missing_mechanisms.md) for an introduction to the concepts of MCAR, MAR, and MNAR.
2. Understand the [Workflow](workflow.md) to diagnose the missing patterns of datasets.
3. Check [Diagnostic Overview](diagnostics.md) for a general categorization of the available tools.
4. Browse the [Function Documentation](documentation.md) to see all available tools.

## Reproducibility

Most stochastic functions provide a `random_state` parameter (default=42). Set it explicitly to ensure reproducible results.

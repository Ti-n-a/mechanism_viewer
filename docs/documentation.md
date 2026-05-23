# Function Documentation

This page provides reference documentation for the available tools and functions in `mechanism_viewer`.

<div class="documentation-page" markdown>

## Data Generation

### `mechanism_viewer.dataset_generator`

::: mechanism_viewer.dataset_generator
	options:
		show_root_heading: false
		show_source: false
		heading_level: 4


## Data Types

### `mechanism_viewer.ColType`

Column data types supported by the synthetic dataset generator
and imputation diagnostics.

It specifies the type of synthetic column to generate and the
imputation strategy to apply.

| Type                       | Meaning                                     | Example use                     |
| -------------------------- | ------------------------------------------- | ------------------------------- |
| `ColType.CONTINUOUS`       | Numeric values with continuous variation    | height, age, temperature        |
| `ColType.DISC_CATEGORICAL` | Discrete values representing categories     | score group, stage, class label |
| `ColType.DISCRETE`         | Integer numeric values                      | number of visits, count values  |
| `ColType.BINARY`           | Binary values represented as ``0`` and `1`  | yes/no, positive/negative       |

	

## Missingness Visualization

### `mechanism_viewer.viewer_simple`

::: mechanism_viewer.viewer_simple
	options:
		show_root_heading: false
		show_source: false
		heading_level: 4

### `mechanism_viewer.viewer_matrix`

::: mechanism_viewer.viewer_matrix
	options:
		show_root_heading: false
		show_source: false
		heading_level: 4

### `mechanism_viewer.viewer_upset`

::: mechanism_viewer.viewer_upset
	options:
		show_root_heading: false
		show_source: false
		heading_level: 4

### `mechanism_viewer.viewer_correlation`

::: mechanism_viewer.viewer_correlation
	options:
		show_root_heading: false
		show_source: false
		heading_level: 4

### `mechanism_viewer.viewer_comparison`

::: mechanism_viewer.viewer_comparison
	options:
		show_root_heading: false
		show_source: false
		heading_level: 4


## Imputation Diagnostics

### `mechanism_viewer.viewer_imputation`

::: mechanism_viewer.viewer_imputation
	options:
		show_root_heading: false
		show_source: false
		heading_level: 4

### Parameter ``missing_col_type`` for Imputation

The imputation model to be used in `missing_col` is determined by `missing_col_type`. In other words, the function selects the model that is best suited for the data type of the missing column. However, the default value of the parameter, `missing_col_type="normal"`, can be used for a MICE model imputation. 

Moreover, the columns used for imputation depend on the selected `missing_col_type`. For continuous/MICE imputation, numeric columns are used, including numeric columns with missing values. This is useful for real-world datasets, where incomplete numeric columns may still help explain each other. For binary, discrete, and categorical imputation, only complete columns are used, because the current models cannot handle missing values.

Ultimately, always select the appropriate `ColType` value when using the Imputation module.

```python
from mechanism_viewer import ColType
mv.plot_imputation_distribution(df_missing, "Col3", missing_col_type=ColType.DISCRETE)
```


## Statistical Diagnostics

### `mechanism_viewer.accuracy_mar`

::: mechanism_viewer.accuracy_mar
	options:
		show_root_heading: false
		show_source: false
		heading_level: 4

### `mechanism_viewer.mcar_little_test`

::: mechanism_viewer.mcar_little_test
	options:
		show_root_heading: false
		show_source: false
		heading_level: 4

</div>

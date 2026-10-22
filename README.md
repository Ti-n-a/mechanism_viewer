# mechanism_viewer

`mechanism_viewer` is a Python package for exploring and diagnosing missing data mechanisms in tabular datasets.

It includes visualization tools, statistical diagnostics, imputation diagnostics, and synthetic data generation for investigating MCAR, MAR, and MNAR missingness.

**Documentation:** [https://ti-n-a.github.io/mechanism_viewer/](https://ti-n-a.github.io/mechanism_viewer/)

**PyPI:** [https://pypi.org/project/mechanism-viewer/](https://pypi.org/project/mechanism-viewer/)

**Repository:** [https://github.com/Ti-n-a/mechanism_viewer/](https://github.com/Ti-n-a/mechanism_viewer/)

## Missing Data Mechanisms

There are 3 types of missing data mechanism, which are MCAR, MAR, and MNAR.

### Missing Completely At Random (MCAR)

Missing values depend neither on observed variables nor on the missing values themselves.

### Missing At Random (MAR)

Missing values depend on observed variables but do not depend on the missing values themselves.

### Missing Not At Random (MNAR)

Missing values depend on the missing values themselves or on unobserved variables, but do not depend on the observed variables.

## Project Structure

```bash
mechanism_viewer/
├── .github/                  # GitHub Actions workflows
├── mechanism_viewer/         # Package source code
│   ├── __init__.py
│   ├── _random.py
│   ├── _utils.py
│   ├── _validation.py
│   ├── accuracy_mar.py
│   ├── column_data_types.py
│   ├── dataset_generator.py
│   ├── little_mcar.py
│   ├── viewer_comparison.py
│   ├── viewer_correlation.py
│   ├── viewer_dendrogram.py
│   ├── viewer_imputation.py
│   ├── viewer_matrix.py
│   ├── viewer_simple.py
│   └── viewer_upset.py
├── tests/                    # Automated test suite
├── examples/                 # Reproducible package examples
├── use_cases/                # Real-world case studies
├── sample_data/              # Example datasets used by notebooks
├── docs/                     # MkDocs documentation source
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── mkdocs.yml
└── pyproject.toml
```

## Installation

`mechanism_viewer` requires Python 3.10 or later.

1. Install the package from [PyPI](https://pypi.org/project/mechanism-viewer/):

   ```bash
   pip install mechanism_viewer
   ```

2. Install documentation dependencies if necessary:

   ```bash
   pip install "mechanism_viewer[docs]"
   ```

## Example

1. Generate a synthetic dataset with missingness:

   ```python
   from mechanism_viewer import (
       generate_dataset_with_missingness,
       plot_missing_rate,
       ColType)

   df = generate_dataset_with_missingness(
       n_rows=1000,
       type_array=[
           ColType.CONTINUOUS,
           ColType.CONTINUOUS,
           ColType.DISCRETE
       ],
       n_complete_cols=1,
       missing_mechanism_array=["MAR", "MCAR"],
       missing_rate_array=[0.20, 0.15])

   fig, ax = plot_missing_rate(df)
   fig.show()
   ```

2. Run Little’s MCAR Test:

   ```python
   from mechanism_viewer import (
       little_mcar_global,
       interpret_mcar_p_value)

   p_value = little_mcar_global(df)

   print(interpret_mcar_p_value(p_value))
   ```

## Notebook Examples

- Example notebooks demonstrating package usage are available inside `examples/`

- Real-world use cases are available inside `use_cases/`

## Sources of Use Case Datasets

The real-world examples in this repository use publicly available datasets from the following sources:

- [Cancer Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
  - Used in: `use_cases/cancer.ipynb`

- [Student Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
  - Used in: `use_cases/student.ipynb`

- [Contraceptive Dataset](https://archive.ics.uci.edu/dataset/30/contraceptive+method+choice)
  - Used in: `use_cases/contraceptive.ipynb`

- [Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
  - Used in: `use_cases/diabetes.ipynb`

- [Horses Dataset](https://www.kaggle.com/datasets/uciml/horse-colic/code)
  - Used in: `use_cases/horses.ipynb`

## Contributing

Please read `CONTRIBUTING.md` for more information.

## License

`mechanism_viewer` is distributed under the BSD 3-Clause "New" or "Revised" License. See [LICENSE](LICENSE) for details.

## Limitations

Missing-data mechanisms generally cannot be proven using a single method.

The best practice is to combine multiple tools with domain knowledge.

> This package should be used as supporting evidence rather than definitive proof of missing-data mechanisms.

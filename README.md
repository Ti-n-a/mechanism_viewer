# mechanism_viewer

mechanism_viewer is a python package that can be used to help diagnose the missing data mechanism of a dataset.

It includes visualization tools, statistical testing, imputation tools, and synthetic dataset generation.


## Missing Data Mechanisms

There are 3 types of missing data mechanism, which are MCAR, MAR, and MNAR.

### Missing Completely At Random (MCAR)

Missing values depend neither on observed variables nor on the missing values themselves.

### Missing At Random (MAR)

Missing values depend on observed variables but do not depend on the missing values themselves.

### Missing Not At Random (MNAR)

Missing values depend on the missing values themselves or on unobserved variables, but do not depend on the observed variables.


## Features

### Visualization Tools

* Missing-rate heatmaps
* Missingness dependency matrices
* Missingness correlation heatmaps
* Missing-pattern visualizations
* Missing-rate matrices
* Missingness comparison plots
* Imputation comparison plots

### Statistical Tools

* Little’s MCAR test
* Pairwise MCAR testing
* Missingness correlation analysis
* Accuracy-based MAR detection

### Synthetic Dataset Generation

Generate datasets with:

* MCAR mechanisms
* MAR mechanisms
* MNAR mechanisms
* Multiple data types
* Controlled missing rates


## Installation

1) Clone the repository:

    ```bash
    git clone https://github.com/Ti-n-a/mechanism_viewer.git
    cd mechanism_viewer
    ```

2) Create a virtual environment.

    Create a python virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

    Or create a conda environment:
    ```bash
    conda create -n mechanism-viewer python=3.11
    conda activate mechanism-viewer
    ```

3) Install the package:

    ```bash
    pip install -e .
    ```

4) Install documentation dependencies if necessary:

    ```bash
    pip install -e ".[docs]"
    ```


## Example

1) Generate a synthetic dataset with missingness:

    ```python
    from mechanism_viewer import (
        generate_dataset_with_missing_data,
        plot_missing_rate,
        ColType)

    df = generate_dataset_with_missing_data(
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

2) Run Little’s MCAR Test:

    ```python
    from mechanism_viewer import (
        little_mcar_test,
        interpret_mcar_p_value)

    p_value = little_mcar_test(df)

    print(interpret_mcar_p_value(p_value))
    ```

## Notebook Examples

- Example notebooks demonstrating package usage are available inside ``examples/``

- Real-world use cases are available inside ``tests_real_datasets/``

## Contributing

Please read ``CONTRIBUTING.md`` for more information.

## Limitations

Missing-data mechanisms generally cannot be proven using a single method. 

The best practice is to combine multiple tools with domain knowledge.

> This package should be used as supporting evidence rather than definitive proof of missing-data mechanisms.

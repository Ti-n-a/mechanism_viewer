# Contributing to ``mechanism_viewer``

Thank you for your interest in contributing to this project.

This package aims to provide visualization and analysis tools for understanding missing data mechanisms (MCAR, MAR, and MNAR).

Please read this guide before contributing.


## Development principles

Contributions should preserve the following principles:

- Reproducibility
- Readable code
- Consistent validation
- Scientific transparency
- Visualization tools easy to interpret
- Modular design


## Setting up development environment

1) Fork the repository on GitHub

2) Clone your fork locally:

    ```bash
    git clone https://github.com/<your-username>/mechanism_viewer.git
    cd mechanism_viewer
    ```

3) Create an environment:


    create a python virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

    Or create a conda environment:
    ```bash
    conda create -n mechanism-viewer python=3.11
    conda activate mechanism-viewer
    ```

4) Install dependencies:

    This installs the package together with its main dependencies listed in ``pyproject.toml``.
    ```bash
    pip install -e .
    ```

    To install documentation dependencies as well:
    ```bash
    pip install -e ".[docs]"
    ```

## What Can I Contribute?

**New Visualization Methods**

- New missingness visualization plots
- Improved interpretation tools
- Better multivariate missingness analysis methods
- More informative plot layouts or aesthetics

**New Statistical or Analytical Methods**

- Additional MCAR/MAR/MNAR detection approaches
- Alternative imputation methods
- Model-based missingness diagnostics
- Benchmarking tools

**Documentation Improvements**

- Improving explanations
- Adding tutorials
- Creating example notebooks
- Improving API documentation

**Real-World Validation**

- Testing tools on public datasets
- Providing additional use cases
- Comparing methods across datasets
- Identifying edge cases or limitations

**Bug Fixes and Performance Improvements**

- Fixing incorrect behavior
- Improving runtime performance
- Improving validation logic
- Improving reproducibility

**New Dataset Examples**

- Synthetic datasets demonstrating specific mechanisms
- Real-world missingness case studies
- Benchmark datasets for comparison


## Development Workflow

1) Create a new branch for your changes:

    ```bash
    git checkout -b your-branch-name
    ```

2) Make your changes.

3) Update exports in `__init__.py`

4) Run the package locally to check that your changes work.

5) Create or update the notebooks demonstrating the new functionality.

    This includes:
    
     - A notebook example: 
        - Placed inside the ``examples/`` folder
        - Demonstrates how the feature works
        - Uses a small, fully reproducible synthetic dataset
     
     - A real-world use case example:
       - Placed inside the ``tests_real_datasets/`` folder
       - Provides additional validation of the new functionality
       - Shows that the feature produces meaningful results
       - Includes interpretation of outputs and notes on limitations of the approach

6) Update the documentation.

    This may include:

     - Updating function docstrings
     - Updating documentation pages
     - Adding usage examples
     - Explaining assumptions, expected use cases, and limitations

7) Commit your changes:

    ```bash
    git add .
    git commit -m "Describe your change"
    ```

8) Push your branch to your fork:
    ```bash
    git push origin your-branch-name
    ```

9) Open a pull request against the main branch.


## Coding Standards

**Style**

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines
- Use descriptive variable names
- Prefer explicit code over compact code
- Avoid deeply nested logic when possible

**Type Hints**

All public functions should include type hints. Example:

```python
def example_function(
    df: pd.DataFrame,
    display_plot: bool = False
) -> tuple[plt.Figure, plt.Axes]:
```

**Validation**

All public functions should validate inputs. Use existing validation helpers from `_validation.py` whenever possible.


## Reproducibility Rules

Random operations must remain reproducible. Therefore, use `DEFAULT_RANDOM_STATE` instead of hardcoded seeds.

Do not use:

```python
np.random.seed(123)
```

Do this:

```python
rng = np.random.default_rng(DEFAULT_RANDOM_STATE)
```


## Plotting Guidelines

Visualization functions should:

- Return `(fig, ax)` objects
- Not display figures by default
- Include `display_plot=False`


## Documentation Requirements

Public functions should contain:

- Description
- Parameters
- Returns

Use NumPy-style docstrings.


## Reporting Issues

The reports should include:

- Dataset characteristics
- Expected behavior
- Actual behavior
- Package version
- Reproducible example


## Disclaimer

- Missing data mechanism detection is inherently uncertain.

- Outputs from this package should be interpreted as supporting evidence rather than definitive proof of MCAR, MAR, or MNAR mechanisms.

- Combining multiple tools is strongly recommended.


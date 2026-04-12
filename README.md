# WARNING: README NOT YET UPDATED

# mechanism_viewer

`mechanism_viewer` is a python package that can be used to help diagnose the missing data mechanism of a dataset.

It includes a variety of visualization tools, such as, plots and heatmaps, to offer multiple perspectives on the missingness, and, therefore, guide the user to a more responsible solution.

## Missing data mechanism

There are 3 types of missing data mechanism, which are MCAR, MAR, and MNAR. 

#### Missing Completely At Random (MCAR)
The missing values *depend neither on the observed variables nor on the missing values themselves*.

#### Missing At Random (MAR)
The missing values *depend on the observed variables*, but do not depend on the missing values themselves.

#### Missing Not At Random (MNAR)
The missing values *depend on the missing values themselves*, but do not depend on the observed variables.

## Steps to use the package on any computer

1) First, guarantee you have downloaded the package to the computer. You can place it in any folder, as long as you remember the path.

Note: The path to be taken is the first folder from the package, not `mechanism_viewer` folder that is inside the package next to README.

2) Open the terminal and execute the following command. The command will install the package in non-editable mode. In other words, if any changes are made to the package, you need to reinstall the package again.

```python
# Make sure to be inside the path of the package 'mechanism_viewer'
python -m pip install .
```

```python
# To execute command outside the path 
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
# For example, inside a Jupiter Notebook file on Windows
%pip install -e "<path_of_package>\mechanism_viewer"
```

Note: In this mode, it is only necessary to reload the kernel to have the most updated version of the package.

3) After installation is complete, you are ready to use the package on your code. Below, there is a quick example on how to use `plot_missing_rate()`.

## Examples

Use one of the visualization tools:
```python
import pandas as pd
from mechanism_viewer import plot_missing_rate

df = pd.read_csv("data.csv")
plot_missing_rate(df)  #plot will be printed automatically
```

To generate synthetic dataset with missing data:

```python
import pandas as pd
import mechanism_viewer

n_rows = 100
n_cols = 5
data = mechanism_viewer.generate_synthetic_dataset(n_rows, n_cols, ["continuous", "discrete" ,"discrete", "discrete_categorical", "binary"])

missing_rate = 0.2
n_complete_cols = 2

data_missing = mechanism_viewer.apply_missing_data(data, ["MAR", "MCAR", "MNAR"], [missing_rate, missing_rate, missing_rate], n_complete_cols)

display(data_missing.head(10))
```

Or using compact version:
```python
import mechanism_viewer

data_missing = mechanism_viewer.generate_dataset_with_missing_data(n_rows=100, n_cols=5, type_array=["continuous", "discrete","discrete", "discrete_categorical", "binary"], missing_mechanism_array=["MAR", "MCAR", "MNAR"], missing_rate_array=[0.2, 0.2, 0.2], n_complete_cols=2, missingness_ascending=True)

display(data_missing.head(10))
```

## Visualize every tool inside the package
An additional Jupyter Notebook file is included in the deliverables zip. The file  displays every tool of the package, using some synthetic datasets as examples of tool usage.

## Steps to reproduce the experiments

The main experiments are listed in the report, in section IV, `Analysis and Discussion`. To reproduce the experiments:

1. Make sure the package is installed and running smoothly.

2. Go to `Apendix A: Code to obtain tables in section B`

3. Copy the code of the desired experiment.

4. Run the experiment.

Note: Some small alterations might be necessary, such as, teaking the array of missing rates, or hide certain columns. 
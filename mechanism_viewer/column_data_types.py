from enum import Enum

class ColType(str, Enum):
    """
    Column data types supported by the synthetic dataset generator
    and imputation diagnostics.

    It specifies the type of synthetic column to generate and the
    imputation strategy to apply.

    Types:
    ------
    ColType.CONTINUOUS
        Numeric values with continuous variation.
    ColType.DISC_CATEGORICAL
        Discrete values representing categories.
    ColType.DISCRETE
        Integer numeric values.
    ColType.BINARY
        Binary values represented as 0 and 1.
    """
    CONTINUOUS = "continuous"
    DISC_CATEGORICAL  = "discrete_categorical"
    DISCRETE = "discrete"
    BINARY = "binary"
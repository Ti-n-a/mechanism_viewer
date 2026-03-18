from enum import Enum

class ColType(str, Enum):
    CONTINUOUS = "continuous"
    DISC_CATEGORICAL  = "discrete_categorical"
    DISCRETE = "discrete"
    BINARY = "binary"
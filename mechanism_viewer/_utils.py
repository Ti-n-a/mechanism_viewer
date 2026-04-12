"""Internal helpers for utilities used across modules."""

import pandas as pd


def get_single_class(
    y_obs: pd.Series,
    ) -> object | None:
    """
    Return the single observed class if it exists.

    Parameters
    ----------
    y_obs : pd.Series
        Observed target values.

    Returns
    -------
    object or None
        The unique class value if there is exactly one observed class, otherwise None.
    """
    target_classes = pd.Series(y_obs).dropna().unique()

    if len(target_classes) == 1:
        return target_classes[0]
    return None

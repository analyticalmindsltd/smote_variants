"""
This module implements wrappers to handle changes in the scipy interface
"""

import inspect

from scipy.stats import mode

from ._base import coalesce

__all__ = ["scipy_mode"]


def scipy_mode(array, axis=0, nan_policy="propagate", mode_func=None):
    """
    Wrapper for scipy mode

    Args:
        array (np.array): the array to apply mode to
        axis (int): the axis index
        nan_policty (str): how to handle NaNs
        version (str/None): the possible version string

    Returns:
        np.array: the mode(s) along the given axis
    """
    mode_func = coalesce(mode_func, mode)
    flag = "keepdims" in list(inspect.signature(mode_func).parameters.keys())

    if not flag:
        return mode_func(array, axis, nan_policy)

    return mode_func(array, axis, nan_policy, keepdims=True)

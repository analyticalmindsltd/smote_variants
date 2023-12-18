"""
A module for global package configurations.
"""

__all__ = [
    "suppress_external_warnings",
    "suppress_internal_warnings",
    "distance_matrix_max_memory_chunk",
    "DEFAULT_SUPPRESS_EXTERNAL_WARNINGS",
    "DEFAULT_SUPPRESS_INTERNAL_WARNINGS",
    "DISTANCE_MATRIX_MAX_MEMORY_CHUNK",
]

DEFAULT_SUPPRESS_EXTERNAL_WARNINGS = True
DEFAULT_SUPPRESS_INTERNAL_WARNINGS = True
DISTANCE_MATRIX_MAX_MEMORY_CHUNK = 50_000_000

flags = {
    "SUPPRESS_EXTERNAL_WARNINGS": DEFAULT_SUPPRESS_EXTERNAL_WARNINGS,
    "SUPPRESS_INTERNAL_WARNINGS": DEFAULT_SUPPRESS_INTERNAL_WARNINGS,
    "DISTANCE_MATRIX_MAX_MEMORY_CHUNK": DISTANCE_MATRIX_MAX_MEMORY_CHUNK,
}


def suppress_external_warnings(flag=None):
    """
    Set/get the value of SUPPRESS_EXTERNAL_WARNINGS

    Args:
        flag (bool): the new flag

    Returns:
        bool: the current value of the flag
    """

    if flag is not None:
        flags["SUPPRESS_EXTERNAL_WARNINGS"] = flag
    return flags["SUPPRESS_EXTERNAL_WARNINGS"]


def suppress_internal_warnings(flag=None):
    """
    Set/get the value of SUPPRESS_INTERNAL_WARNINGS

    Args:
        flag (bool): the new flag

    Returns:
        bool: the current value of the flag
    """

    if flag is not None:
        flags["SUPPRESS_INTERNAL_WARNINGS"] = flag
    return flags["SUPPRESS_INTERNAL_WARNINGS"]


def distance_matrix_max_memory_chunk(value=None):
    """
    Set/get the maximum memory chunk for computing distance matrices

    Args:
        value (int): the maximum size of memory to be used

    Returns:
        int: the current value
    """

    if value is not None:
        flags["DISTANCE_MATRIX_MAX_MEMORY_CHUNK"] = value
    return flags["DISTANCE_MATRIX_MAX_MEMORY_CHUNK"]

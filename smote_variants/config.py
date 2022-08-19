"""
A module for global package configurations.
"""

__all__=['suppress_external_warnings',
         'suppress_internal_warnings',
         'DEFAULT_SUPPRESS_EXTERNAL_WARNINGS',
         'DEFAULT_SUPPRESS_INTERNAL_WARNINGS']

DEFAULT_SUPPRESS_EXTERNAL_WARNINGS = True
DEFAULT_SUPPRESS_INTERNAL_WARNINGS = True

flags = {'SUPPRESS_EXTERNAL_WARNINGS': DEFAULT_SUPPRESS_EXTERNAL_WARNINGS,
         'SUPPRESS_INTERNAL_WARNINGS': DEFAULT_SUPPRESS_INTERNAL_WARNINGS}

def suppress_external_warnings(flag=None):
    """
    Set/get the value of SUPPRESS_EXTERNAL_WARNINGS

    Args:
        flag (bool): the new flag

    Returns:
        bool: the current value of the flag
    """

    if flag is not None:
        flags['SUPPRESS_EXTERNAL_WARNINGS'] = flag
    return flags['SUPPRESS_EXTERNAL_WARNINGS']

def suppress_internal_warnings(flag=None):
    """
    Set/get the value of SUPPRESS_INTERNAL_WARNINGS

    Args:
        flag (bool): the new flag

    Returns:
        bool: the current value of the flag
    """

    if flag is not None:
        flags['SUPPRESS_INTERNAL_WARNINGS'] = flag
    return flags['SUPPRESS_INTERNAL_WARNINGS']

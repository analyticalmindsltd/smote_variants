"""
Tests the configurations
"""

from smote_variants.config import (suppress_external_warnings,
                                   suppress_internal_warnings,
                                   DEFAULT_SUPPRESS_EXTERNAL_WARNINGS,
                                   DEFAULT_SUPPRESS_INTERNAL_WARNINGS)

def test_external_warnings():
    """
    Tests the external warnings settings
    """

    suppress_external_warnings(True)

    assert suppress_external_warnings()

    suppress_external_warnings(False)

    assert not suppress_external_warnings()

    suppress_external_warnings(DEFAULT_SUPPRESS_EXTERNAL_WARNINGS)

def test_internal_warnings():
    """
    Tests the internal warnings settings
    """

    suppress_internal_warnings(True)

    assert suppress_internal_warnings()

    suppress_internal_warnings(False)

    assert not suppress_internal_warnings()

    suppress_internal_warnings(DEFAULT_SUPPRESS_INTERNAL_WARNINGS)

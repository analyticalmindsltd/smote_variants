"""
This module tests the UnderSampling class.
"""

from smote_variants.base import UnderSampling

def test_undersampling():
    """
    Testing the UnderSampling class.
    """
    unders = UnderSampling()

    assert unders.sample(None, None)[0] is None

    assert len(unders.get_params()) > 0

    assert len(unders.descriptor()) > 0

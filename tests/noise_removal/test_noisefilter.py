"""
Testing the NoiseFilter class.
"""

from smote_variants.noise_removal import NoiseFilter

def test_noisefilter():
    """
    Tests the noisefilter base class.
    """
    noisef = NoiseFilter()
    noisef.set_params(**{'dummy': 1})

    assert getattr(noisef, 'dummy') == 1

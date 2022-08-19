"""
Testing the TomekLinkRemoval.
"""

from smote_variants.noise_removal import TomekLinkRemoval

from smote_variants.datasets import load_all_min_noise

def test_specific():
    """
    Oversampler specific testing
    """
    dataset = load_all_min_noise()

    obj = TomekLinkRemoval(strategy='remove_both')

    X_samp, _ = obj.remove_noise(dataset['data'], dataset['target'])

    assert len(X_samp) > 0

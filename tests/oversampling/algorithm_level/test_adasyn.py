"""
Testing the ADASYN.
"""

from smote_variants import ADASYN

from smote_variants.datasets import load_normal

def test_specific():
    """
    Oversampler specific testing
    """

    dataset = load_normal()

    obj = ADASYN(d_th=0.0)
    X_samp, _ = obj.sample(dataset['data'], dataset['target'])

    assert len(dataset['data']) == len(X_samp)

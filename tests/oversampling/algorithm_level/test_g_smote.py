"""
Testing the G_SMOTE.
"""
import pytest

from smote_variants import G_SMOTE

from smote_variants.datasets import load_normal

def test_specific():
    """
    Oversampler specific testing
    """
    obj = G_SMOTE(method="non-linear_1.0")

    dataset = load_normal()

    X_samp, _ = obj.sample(dataset['data'],
                            dataset['target'])

    assert len(X_samp) > 0

    with pytest.raises(Exception):
        G_SMOTE(method='dummy')

    with pytest.raises(Exception):
        G_SMOTE(method='non-linear_-3')

"""
Testing the ProWSyn.
"""
import numpy as np

from smote_variants import ProWSyn

from smote_variants.datasets import load_normal

def test_specific():
    """
    Oversampler specific testing
    """
    obj = ProWSyn()

    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5],
                    [5, 6], [6, 7], [8, 9]])
    y = np.array([0, 1, 0, 1, 0, 1, 0])

    assert obj.sample(X, y)[0].shape[0] > 0

    obj = ProWSyn(L=1, n_neighbors=1)

    dataset = load_normal()

    X_samp, _ = obj.sample(dataset['data'],
                            dataset['target'])

    assert len(X_samp) > 0

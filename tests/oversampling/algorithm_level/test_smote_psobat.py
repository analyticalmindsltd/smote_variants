"""
Testing the SMOTE_PSOBAT.
"""
import numpy as np

from smote_variants import SMOTE_PSOBAT

def test_specific():
    """
    Oversampler specific testing
    """
    obj = SMOTE_PSOBAT(method='pso')

    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5],
                    [5, 6], [6, 7], [8, 9]])
    y = np.array([0, 1, 0, 1, 0, 1, 0])

    assert obj.sample(X, y)[0].shape[0] > 0

    assert len(obj.parameter_combinations(raw=True)) > 0

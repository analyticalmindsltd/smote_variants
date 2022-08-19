"""
Testing the SMOTE_TomekLinks.
"""
import numpy as np

from smote_variants import SMOTE_TomekLinks

def test_specific():
    """
    Oversampler specific testing
    """
    obj = SMOTE_TomekLinks()

    X = np.array([[1, 2], [2, 2], [3, 4], [4, 4],
                    [5, 6], [6, 6], [8, 9]])
    y = np.array([0, 1, 0, 1, 0, 1, 0])

    assert obj.sample(X, y)[0].shape[0] > 0

    X_samp, _ = obj.return_results(np.zeros((0, 2)), np.array([]), X, y)

    assert X_samp.shape[0] == X.shape[0]

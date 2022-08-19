"""
Testing the Safe_Level_SMOTE.
"""
import numpy as np

from smote_variants import Safe_Level_SMOTE

def test_specific():
    """
    Oversampler specific testing
    """
    obj = Safe_Level_SMOTE()

    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5],
                    [5, 6], [6, 7], [8, 9]])
    y = np.array([0, 1, 0, 1, 0, 1, 0])

    assert obj.sample(X, y)[0].shape[0] > 0

    X, y = obj.generate_samples(np.array([]), np.array([]), np.array([]), 2)
    assert len(X) == 0

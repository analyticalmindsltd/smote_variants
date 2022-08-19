"""
Testing the EditedNN
"""

import numpy as np

from smote_variants.noise_removal import EditedNearestNeighbors

def test_specific():
    """
    Oversampler specific testing
    """

    obj = EditedNearestNeighbors()

    X_samp, _ = obj.remove_noise(np.array([[1, 1], [2, 2], [3, 3]]),
                                 np.array([0, 0, 1]))

    assert len(X_samp) == 3

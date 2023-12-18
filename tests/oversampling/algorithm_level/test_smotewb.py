"""
Testing the SMOTEWB.
"""

import numpy as np

from smote_variants import SMOTEWB


def all_noise_by_dtrees():
    """
    A data set where class membership cannot be judged based on the features.
    Different decision trees can easily provide different results.
    """
    X = np.array([[i, 80 - i] for i in range(0, 80)])
    y = np.zeros(X.shape[0], int)
    y[15] = 1
    y[10] = 1
    y[5] = 1
    return X, y


def low_kmax_good_noise():
    """Dataset with sample that is considered noise by decision trees,
    but classified as a minor by 1NN, and result low k_max value."""
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 0],
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 0],
            [2, 1],
            [2, 2],
            [2, 3],
            [2, 4],
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 3],
            [3, 4],
            [4, 0],
            [4, 1],
            [4, 2],
            [4, 3],
            [4, 4],
        ]
    )

    y = np.array(
        [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1]
    )

    return X, y


def test_specific():
    """
    Oversampler specific testing
    """

    obj = SMOTEWB(n_iters=30, max_depth=3, proportion=0.25)

    X, y = all_noise_by_dtrees()
    X_samp, _ = obj.sample(X, y)

    assert len(X_samp) >= len(X)

    obj = SMOTEWB(n_iters=100, max_depth=3, proportion=0.75)

    X, y = low_kmax_good_noise()
    X_samp, _ = obj.sample(X, y)

    assert len(X_samp) >= len(X)

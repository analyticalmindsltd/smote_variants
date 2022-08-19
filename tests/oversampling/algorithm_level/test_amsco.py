"""
Testing the AMSCO.
"""

import numpy as np

from smote_variants import AMSCO

np.random.seed(42)
X = np.random.normal(size=(20, 40))
y = np.hstack([np.repeat(1, 7), np.repeat(0, 13)])

def test_specific():
    """
    Oversampler specific testing
    """

    obj = AMSCO()
    X_samp, _ = obj.sample(X, y)

    assert len(X_samp) > 0

    particle = np.array([1, 2, 3, 4])

    for _ in range(10):
        assert len(obj.remove_elements(particle)) >= 4

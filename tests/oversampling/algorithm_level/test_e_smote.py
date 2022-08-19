"""
Testing the E_SMOTE.
"""

import numpy as np

from smote_variants import E_SMOTE

np.random.seed(42)
X = np.random.normal(size=(1000, 30))
y = np.hstack([np.repeat(1, 70), np.repeat(0, 930)])

def test_specific():
    """
    Oversampler specific testing
    """

    obj = E_SMOTE()

    X_samp, _ = obj.sample(X, y)

    assert len(X_samp) > 0

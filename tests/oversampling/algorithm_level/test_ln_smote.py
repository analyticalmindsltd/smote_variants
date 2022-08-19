"""
Testing the LN_SMOTE.
"""
import numpy as np

from smote_variants import LN_SMOTE

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5],
                    [5, 6], [6, 7], [8, 9]])
y = np.array([0, 1, 0, 1, 0, 1, 0])

def test_specific():
    """
    Oversampler specific testing
    """

    obj = LN_SMOTE()

    assert obj.random_gap(slp=1,
                            sln=0,
                            n_label=None,
                            n_neighbors=None,
                            n_dim=2).shape[0] == 2

    obj = LN_SMOTE(n_neighbors=2)

    X_samp, _ = obj.sample(X, y)

    assert len(X_samp) > 0

    obj = LN_SMOTE(n_neighbors=1)

    X_samp, _ = obj.sample(X, y)

    assert len(X_samp) > 0

    samples = obj.generate_samples(slpsln={},
                                    n_to_sample=5,
                                    X=np.array([[1, 2]]),
                                    y=None,
                                    n_neighbors=5)

    assert samples.shape[0] == 0

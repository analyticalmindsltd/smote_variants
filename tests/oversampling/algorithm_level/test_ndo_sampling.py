"""
Testing the NDO_sampling.
"""
import numpy as np

from smote_variants import NDO_sampling

from smote_variants.datasets import load_normal

def test_specific():
    """
    Oversampler specific testing
    """
    obj = NDO_sampling(T=20.0)

    dataset = load_normal()

    X_samp, _ = obj.sample(dataset['data'],
                            dataset['target'])

    assert X_samp.shape[0] > 0

    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5],
                    [5, 6], [6, 7], [8, 9]])
    y = np.array([0, 1, 0, 1, 0, 1, 0])

    assert obj.sample(X, y)[0].shape[0] > 0

    alpha = obj.determine_alpha(y = np.array([0, 0, 1, 1]),
                                ind = np.array([[2, 0],
                                                [3, 1]]),
                                dist = np.array([[0.0, 0.0],
                                                [0.0, 0.0]]))
    assert alpha == 0.0

    alpha = obj.determine_alpha(y = np.array([0, 0, 1, 1]),
                                ind = np.array([[2, 3],
                                                [3, 2]]),
                                dist = np.array([[0.0, 0.0],
                                                [0.0, 0.0]]))
    assert alpha == np.inf

    alpha = obj.determine_alpha(y = np.array([0, 0, 1, 1]),
                                ind = np.array([[2, 3],
                                                [3, 1]]),
                                dist = np.array([[0.0, 0.0],
                                                [0.0, 0.0]]))
    assert alpha == np.inf

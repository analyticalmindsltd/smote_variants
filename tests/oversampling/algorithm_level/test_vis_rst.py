"""
Testing the VIS_RST.
"""
import numpy as np

import pytest

from smote_variants import VIS_RST

def test_specific():
    """
    Oversampler specific testing
    """
    obj = VIS_RST()

    assert len(obj.parameter_combinations(raw=True)) > 0

    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5],
                    [5, 6], [6, 7], [8, 9], [9, 10], [10, 11],
                    [11, 12], [12, 13], [13, 14], [14, 15],
                    [15, 16], [16, 17]])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    assert obj.sample(X, y)[0].shape[0] > 0

    X = np.array([[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2],
                    [3, 4], [3, 4], [3, 4], [3, 4], [3, 4]])
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    assert obj.sample(X, y)[0].shape[0] > 0

    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5],
                    [5, 6], [6, 7], [8, 9], [9, 10], [10, 11],
                    [11, 12], [12, 13], [13, 14], [14, 15],
                    [15, 16], [16, 17]])
    y = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0])

    assert obj.sample(X, y)[0].shape[0] > 0

    with pytest.raises(ValueError):
        obj.check_all_boundary(np.array([1]))

    with pytest.raises(ValueError):
        obj.check_labels(np.array(['NOI']))

    with pytest.raises(ValueError):
        obj.check_all_noise(np.array([1]))

    assert obj.set_mode(None, np.array(['DAN'])) == 'no_safe'

    assert obj.set_mode(np.zeros((0, 2)), np.array(['DAN', 'SAF'])) == 'high_complexity'

    ind = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    labels = np.array(['SAF', 'DAN', 'NOI'])[ind]

    assert len(obj.generate_samples(X_min=X,
                                    nn_params={},
                                    n_to_sample=10,
                                    labels=labels,
                                    mode='high_complexity')) > 0

    assert len(obj.generate_samples(X_min=X,
                                    nn_params={},
                                    n_to_sample=10,
                                    labels=labels,
                                    mode='low_complexity')) > 0

    assert len(obj.generate_samples(X_min=X,
                                    nn_params={},
                                    n_to_sample=10,
                                    labels=labels,
                                    mode='other')) > 0

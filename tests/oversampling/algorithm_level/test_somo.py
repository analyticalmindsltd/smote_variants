"""
Testing the SOMO.
"""
import numpy as np

from smote_variants import SOMO

from smote_variants.datasets import (load_all_min_noise,
                                     load_3_min_some_maj,
                                     load_separable,
                                     load_illustration_2_class)

def test_specific():
    """
    Oversampler specific testing
    """
    obj = SOMO(n_grid=5)

    dataset = load_illustration_2_class()

    X_samp, _ = obj.sample(dataset['data'], dataset['target'])

    assert len(X_samp) > 0

    dataset = load_all_min_noise()

    X_samp, _ = obj.sample(dataset['data'],
                            dataset['target'])

    assert len(X_samp) > 0

    dataset = load_separable()

    X_samp, _ = obj.sample(dataset['data'],
                            dataset['target'])

    assert len(X_samp) > 0

    dataset = load_3_min_some_maj()

    X_samp, _ = obj.sample(dataset['data'],
                            dataset['target'])

    assert len(X_samp) > 0

    X = np.array([[1, 2], [2, 2], [3, 4], [4, 4],
                    [5, 6], [6, 6], [8, 9]])
    y = np.array([0, 1, 0, 1, 0, 1, 0])

    assert obj.sample(X, y)[0].shape[0] > 0

"""
Testing the Supervised_SMOTE.
"""

import numpy as np

from smote_variants import Supervised_SMOTE


from smote_variants.datasets import (load_all_min_noise,
                                     load_3_min_some_maj,
                                     load_illustration_2_class)

def test_specific():
    """
    Oversampler specific testing
    """
    obj = Supervised_SMOTE(th_lower=0.9, th_upper=0.95)

    dataset = load_illustration_2_class()

    X_samp, _ = obj.sample(dataset['data'], dataset['target'])

    assert len(X_samp) > 0

    dataset = load_3_min_some_maj()

    X_samp, _ = obj.sample(dataset['data'], dataset['target'])

    assert len(X_samp) > 0

    dataset = load_all_min_noise()

    X_samp, _ = obj.sample(dataset['data'], dataset['target'])

    assert len(X_samp) > 0

    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5],
                    [5, 6], [6, 7], [8, 9], [9, 10], [10, 11],
                    [11, 12], [12, 13], [13, 14], [14, 15],
                    [15, 16], [16, 17]])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    assert obj.sample(X, y)[0].shape[0] > 0

"""
Testing the CURE_SMOTE.
"""

import numpy as np

from smote_variants import CURE_SMOTE

from smote_variants.datasets import (load_all_min_noise,
                                     load_3_min_some_maj)

def test_specific():
    """
    Oversampler specific testing
    """

    obj = CURE_SMOTE(noise_th=10)

    dataset = load_all_min_noise()

    X_samp, _ = obj.sample(dataset['data'],
                            dataset['target'])

    assert len(X_samp) > 0

    obj = CURE_SMOTE(n_clusters=3)

    dataset = load_3_min_some_maj()

    X_samp, _ = obj.sample(dataset['data'],
                            dataset['target'])

    assert len(X_samp) > 0

    X_min = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    clusters = [np.array([0]), np.array([1]), np.array([2])]
    samples = obj.generate_samples_in_clusters(X_min, clusters, 10)

    assert len(samples) > 0

"""
Testing the MOT2LD.
"""
import numpy as np

import pytest

from smote_variants import MOT2LD

from smote_variants.datasets import load_normal

def test_specific():
    """
    Oversampler specific testing
    """

    with pytest.raises(Exception):
        MOT2LD(d_cut=-1)

    with pytest.raises(Exception):
        MOT2LD(d_cut='dummy')

    obj = MOT2LD()

    with pytest.raises(Exception):
        obj.check_enough_clusters(np.array([0]))

    with pytest.raises(Exception):
        obj.check_enough_peaks(np.array([]))

    with pytest.raises(Exception):
        obj.check_empty_clustering(np.array([[]]))

    obj = MOT2LD(d_cut=2)

    dataset = load_normal()

    X_samp, _ = obj.sample(dataset['data'],
                            dataset['target'])

    assert len(X_samp) > 0

    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5],
                    [5, 6], [6, 7], [8, 9]])
    y = np.array([0, 1, 0, 1, 0, 1, 0])

    assert obj.sample(X, y)[0].shape[0] > 0

"""
Testing the TRIM_SMOTE.
"""
import numpy as np

from smote_variants import TRIM_SMOTE

from smote_variants.datasets import load_illustration_2_class

def test_specific():
    """
    Oversampler specific testing
    """

    obj = TRIM_SMOTE(min_precision = 1.0)

    dataset = load_illustration_2_class()

    X_samp, _ = obj.sample(dataset['data'], dataset['target'])

    assert len(X_samp) > 0

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

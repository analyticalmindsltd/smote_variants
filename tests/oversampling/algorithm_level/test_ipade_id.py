"""
Testing the IPADE-ID.
"""
import numpy as np

import pytest

from smote_variants import IPADE_ID

def test_specific():
    """
    Oversampler specific testing
    """
    obj = IPADE_ID()

    with pytest.raises(Exception):
        obj.configuration_checks(y=np.array([0, 0]),
                                    for_validation=np.array([0, 1]),
                                    GS_y=None)
    with pytest.raises(Exception):
        obj.configuration_checks(y=np.array([0, 1]),
                                    for_validation=np.array([0, 1]),
                                    GS_y=np.array([0, 0]))

    X = np.array([[1, 1], [2, 1], [3, 1],
                    [4, 1], [5, 1], [6, 1],
                    [7, 1], [8, 1], [9, 1],
                    [10, 1], [11, 1]])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    assert IPADE_ID().sample(X, y)[0].shape[0] == X.shape[0]

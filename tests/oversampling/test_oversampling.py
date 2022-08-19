"""
This module tests the OverSampling base class
"""

import numpy as np

import pytest

from smote_variants.base import (OverSampling, OverSamplingSimplex,
                            OverSamplingBase)

def test_oversamplingbase():
    """
    Testing the OverSampling base class
    """
    overs = OverSamplingBase()

    X = np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
    y = np.array([0, 1, 0])

    X_samp, _ = overs.sampling_algorithm(X, y)

    assert X_samp.shape == X.shape

    X_samp, _ = overs.sample(X, y)

    assert X_samp.shape == X.shape

    overs.checks={'min_n_min': 0}

    X_samp, _ = overs.sample(X, y)

    assert X_samp.shape == X.shape

    n_to_sample = overs.det_n_to_sample(1.0)

    assert n_to_sample >= 0

    n_to_sample = overs.det_n_to_sample(1.0, 20, 10)

    assert n_to_sample >= 0

    with pytest.raises(Exception):
        overs.det_n_to_sample('dummy')

    X_samp, _ = overs.fit_resample(X, y)

    assert X_samp.shape == X.shape

    X_samp, _ = overs.sample_with_timing(X, y)

    assert X_samp.shape == X.shape

    X_samp, _ = overs.return_copies(X, y, "message")

    assert X_samp.shape == X.shape

    X_prep = overs.preprocessing_transform(X) # pylint: disable=invalid-name

    assert X_prep.shape == X.shape

    assert len(overs.get_params()) == 0

    assert getattr(overs.set_params(**{'dummy': 1}), 'dummy') == 1

    assert len(overs.descriptor()) > 0

    assert len(str(overs)) > 0

def test_oversampling():
    """
    Testing the OverSampling class
    """
    overs = OverSampling(random_state=5)

    assert overs.get_params()['random_state'] == 5

    assert overs.sample_between_points(np.array([1]),
                                       np.array([2])).shape[0] == 1

    x_samp = overs.sample_between_points_componentwise(np.array([1]),
                                                       np.array([2]))

    assert x_samp.shape[0] == 1

    x_samp = overs.sample_between_points_componentwise(np.array([1]),
                                                       np.array([2]),
                                                       np.array([1]))

    assert x_samp.shape[0] == 1

    x_samp = overs.sample_by_jittering(np.array([1]), 1)

    assert x_samp.shape[0] == 1

    x_samp = overs.sample_by_jittering_componentwise(np.array([1]), 1)

    assert x_samp.shape[0] == 1

    x_samp = overs.sample_by_gaussian_jittering(np.array([1]), 1)

    assert x_samp.shape[0] == 1

def test_oversamplingsimplex():
    """
    Testing the OverSamplingSimplex class
    """
    with pytest.warns(Warning):
        overs = OverSamplingSimplex(random_state=5,
                                    checks={'simplex_dim': 3})

    assert overs.get_params()['random_state'] == 5

"""
Testing oversampling and noise removal with high dimensionality data.
"""

import numpy as np
import pytest

import smote_variants as sv

from smote_variants.datasets import load_high_dim
from .additional_objs import additional_objs

dataset = load_high_dim()

@pytest.mark.parametrize("smote_class", sv.get_all_oversamplers())
def test_high_dim(smote_class):
    """
    Testing an oversampler with high dimensionality data.

    Args:
        smote_class (class): an oversampler class
    """
    smote_obj = smote_class()
    X, y = smote_obj.sample(dataset['data'], dataset['target'])
    assert len(X) > 0
    assert X.shape[1] == smote_obj\
                                .preprocessing_transform(dataset['data']).shape[1]
    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", additional_objs)
def test_high_dim_plus(smote_obj):
    """
    Testing some specific parameterizations with high dimensionality data.

    Args:
        smote_obj (obj): an oversampling object
    """
    X, y = smote_obj.sample(dataset['data'], dataset['target'])
    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("nf_class", sv.get_all_noisefilters())
def test_normal(nf_class):
    """
    Testing a noise filter with high dimensionality data

    Args:
        nf_class (class): a noise filter class
    """
    X, y = nf_class().remove_noise(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

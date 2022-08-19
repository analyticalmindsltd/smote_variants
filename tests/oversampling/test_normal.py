"""
Testing the oversamplers and noise removals under normal conditions.
"""

import pytest
import numpy as np

import smote_variants as sv

from smote_variants.datasets import load_normal
from .additional_objs import additional_objs

dataset = load_normal()

@pytest.mark.parametrize("smote_class", sv.get_all_oversamplers())
def test_normal(smote_class):
    """
    Tests an oversmampler

    Args:
        smote_class (class): an oversampler class
    """
    X, y = smote_class().sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", additional_objs)
def test_normal_additional(smote_obj):
    """
    Tests some oversamplers with additional parameterizations.

    Args:
        smote_obj (obj): an oversampling object
    """
    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("nf_class", sv.get_all_noisefilters())
def test_normal_noisefilters(nf_class):
    """
    Tests a noise remove technique.

    Args:
        nf_class (class): a noise removal class
    """
    X, y = nf_class().remove_noise(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

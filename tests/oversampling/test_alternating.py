"""
Testing oversampling with an alternating pattern.
"""

import pytest
import numpy as np

import smote_variants as sv

from smote_variants.datasets import load_alternating
from .additional_objs import additional_objs

dataset = load_alternating()

@pytest.mark.parametrize("smote_class", sv.get_all_oversamplers())
def test_alternating(smote_class):
    """
    Testing oversamplers with alternating minority samples.

    Args:
        smote_class (class): an oversampler class.
    """
    X, y = smote_class().sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", additional_objs)
def test_alternating_additional(smote_obj):
    """
    Testing oversamplers with alternating minority samples.

    Args:
        smote_obj (obj): an oversampler obj
    """
    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

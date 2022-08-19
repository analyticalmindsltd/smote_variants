"""
Testing oversampling with illustration data.
"""

import pytest
import numpy as np

import smote_variants as sv

from smote_variants.datasets import load_illustration_2_class
from .additional_objs import additional_objs

dataset = load_illustration_2_class()

@pytest.mark.parametrize("smote_class", sv.get_all_oversamplers())
def test_illustration(smote_class):
    """
    Testing oversamplers with illustration data.

    Args:
        smote_class (class): an oversampler class.
    """
    X, y = smote_class().sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", additional_objs)
def test_illustration_additional(smote_obj):
    """
    Testing oversamplers with illustration data.

    Args:
        smote_obj (obj): an oversampler obj
    """
    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

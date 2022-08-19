"""
Testing oversampling with linearly dependent data.
"""

import pytest
import numpy as np

import smote_variants as sv

from smote_variants.datasets import load_linearly_dependent
from .additional_objs import additional_objs

dataset = load_linearly_dependent()

@pytest.mark.parametrize("smote_class", sv.get_all_oversamplers())
def test_linearly_dependent(smote_class):
    """
    Testing oversamplers with 2 minority samples.

    Args:
        smote_class (class): an oversampler class.
    """
    X, y = smote_class().sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", additional_objs)
def test_linearly_dependent_additional(smote_obj):
    """
    Testing oversamplers with 2 minority samples.

    Args:
        smote_obj (obj): an oversampler obj
    """
    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

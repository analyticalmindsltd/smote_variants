"""
Tests oversamplers with repeated samples
"""

import pytest
import numpy as np

import smote_variants as sv

from smote_variants.datasets import load_repeated

dataset = load_repeated()

@pytest.mark.parametrize("smote_class", sv.get_all_oversamplers())
def test_repeated(smote_class):
    """
    Tests oversamplers with repeated samples

    Args:
        smote_class (class): an oversampling class
    """
    X, y = smote_class().sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

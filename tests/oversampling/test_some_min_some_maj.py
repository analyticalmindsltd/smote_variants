"""
Tests oversamplers with a few samples.
"""

import pytest
import numpy as np

import smote_variants as sv

from smote_variants.datasets import load_some_min_some_maj

dataset = load_some_min_some_maj()

@pytest.mark.parametrize("smote_class", sv.get_all_oversamplers())
def test_some_min_some_maj(smote_class):
    """
    Tests an oversampler with only a few samples.

    Args:
        smote_class (class): the oversampler class
    """
    X, y = smote_class().sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

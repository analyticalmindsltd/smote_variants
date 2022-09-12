"""
Tests oversamplers with equalized data.
"""

import pytest
import numpy as np

import smote_variants as sv

from smote_variants.datasets import load_same_num

# disabling smote-variants logging
import logging
logger = logging.getLogger('smote_variants')
logger.setLevel(logging.CRITICAL)

dataset = load_same_num()

@pytest.mark.parametrize("smote_class", sv.get_all_oversamplers())
def test_same_num(smote_class):
    """
    Tests oversamplers with equalized data.

    Args:
        smote_class (class): an oversampling class
    """
    X, y = smote_class().sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

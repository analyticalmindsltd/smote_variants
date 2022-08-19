"""
Testing oversampling with metric learning.
"""

import pytest
import numpy as np

import smote_variants as sv

from smote_variants.datasets import load_illustration_2_class

dataset = load_illustration_2_class()

@pytest.mark.parametrize("smote_class", sv.get_metric_learning_oversamplers())
def test_metric_learning(smote_class):
    """
    Testing oversamplers with metric learning.

    Args:
        smote_class (class): an oversampler class.
    """
    nn_params = {'metric': 'precomputed',
                 'metric_learning_method': 'ITML'}
    X, y = smote_class(nn_params=nn_params).sample(dataset['data'],
                                                   dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

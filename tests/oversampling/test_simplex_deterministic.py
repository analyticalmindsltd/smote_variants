"""
Testing oversampling with deterministic simplex sampling.
"""

import pytest
import numpy as np

import smote_variants as sv

from smote_variants.datasets import load_illustration_2_class

dataset = load_illustration_2_class()

@pytest.mark.parametrize("smote_class",
    sv.get_simplex_sampling_oversamplers(within_simplex_sampling='random',
                                            n_dim_range=2))
def test_simplex_deterministic(smote_class):
    """
    Testing oversamplers with determinist csimplex sampling.

    Args:
        smote_class (class): an oversampler class.
    """
    ss_params = {'within_simplex_sampling': 'deterministic'}
    X, y = smote_class(ss_params=ss_params).sample(dataset['data'],
                                                   dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

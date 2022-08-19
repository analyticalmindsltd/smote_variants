"""
Tests the multiclass oversampling functionalities.
"""

import logging

import numpy as np
from sklearn import datasets
import pytest

import smote_variants as sv

logger = logging.getLogger("smote_variants")
logger.setLevel(logging.ERROR)

def test_multiclass():
    """
    Test the multiclass oversampling.
    """
    dataset = datasets.load_wine()

    oversampler = sv.MulticlassOversampling('SMOTE')

    X_samp, y_samp = oversampler.sample(dataset['data'], dataset['target'])

    assert len(X_samp) > 0

    _, counts = np.unique(y_samp, return_counts=True)

    assert np.all(counts == counts[0])

    oversampler = sv.MulticlassOversampling('SMOTE',
                                            strategy='equalize_1_vs_many')

    X_samp, y_samp = oversampler.sample(dataset['data'], dataset['target'])

    assert len(X_samp) > 0

def test_non_compatible():
    """
    Test the use of incompatible oversampler.
    """
    with pytest.raises(ValueError):
        sv.MulticlassOversampling('SUNDO')

def test_nonsense():
    """
    Test the use of meaningless strategy.
    """
    with pytest.raises(ValueError):
        sv.MulticlassOversampling('SMOTE', strategy='nonsense')

def test_get_params():
    """
    Test the get_params function.
    """
    params = sv.MulticlassOversampling('SMOTE').get_params()
    assert len(params) > 0

"""
Testing the reproducibility of oversampling and noise filtering.
"""

import numpy as np
import pytest

import smote_variants as sv

from smote_variants.datasets import load_normal
from .additional_objs import additional_objs

dataset = load_normal()

X_normal = dataset['data']
y_normal = dataset['target']

@pytest.mark.parametrize("smote_class", sv.get_all_oversamplers())
def test_reproducibility(smote_class):
    """
    Tests the reproducibility of oversampling.

    Args:
        smote_class (class): an oversampling class
    """
    X_orig = X_normal.copy()
    y_orig = y_normal.copy()

    X_a, y_a = smote_class(random_state=5).sample(X_normal, y_normal)
    oversampler = smote_class(random_state=5)
    X_b, y_b = oversampler.sample(X_normal, y_normal)
    X_c, y_c = smote_class(**oversampler.get_params()).sample(X_normal, y_normal)

    assert np.array_equal(X_a, X_b)
    assert np.array_equal(X_b, X_c)
    assert np.array_equal(X_orig, X_normal)

    assert np.array_equal(y_a, y_b)
    assert np.array_equal(y_b, y_c)
    assert np.array_equal(y_orig, y_normal)

@pytest.mark.parametrize("smote_obj", additional_objs)
def test_reproducibility_additional(smote_obj):
    """
    Tests the reproducibility of specific parameterizations.

    Args:
        smote_obj (obj): an oversampling object
    """
    smote_obj = smote_obj.__class__(**smote_obj.get_params())

    X_orig = X_normal.copy()
    y_orig = y_normal.copy()

    X_a, y_a = smote_obj.sample(X_normal, y_normal)
    X_c, y_c = smote_obj.__class__(**smote_obj.get_params()).sample(X_normal, y_normal)

    assert np.array_equal(X_a, X_c)
    assert np.array_equal(y_a, y_c)
    assert np.array_equal(X_normal, X_orig)
    assert np.array_equal(y_normal, y_orig)

@pytest.mark.parametrize("nf_class", sv.get_all_noisefilters())
def test_reproducibility_noisefilters(nf_class):
    """
    Tests the reproducibility of noise filtering.

    Args:
        nf_class (class): a noise filtering class
    """
    X_orig = X_normal.copy()
    y_orig = y_normal.copy()

    X_a, y_a = nf_class().remove_noise(X_normal, y_normal)
    noisefilter = nf_class()
    X_b, y_b = noisefilter.remove_noise(X_normal, y_normal)
    X_c, y_c = nf_class(**noisefilter.get_params()).remove_noise(X_normal, y_normal)

    assert np.array_equal(X_a, X_b)
    assert np.array_equal(X_b, X_c)
    assert np.array_equal(X_normal, X_orig)

    assert np.array_equal(y_a, y_b)
    assert np.array_equal(y_b, y_c)
    assert np.array_equal(y_normal, y_orig)

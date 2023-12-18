"""
This module tests the undersampling functionalities
"""

import pytest

import numpy as np

from smote_variants.datasets import load_normal
from smote_variants.undersampling import (
    RandomUndersampling,
    OversamplingDrivenUndersampling,
)
from smote_variants.oversampling import SMOTE

undersamplings = [RandomUndersampling, OversamplingDrivenUndersampling]


@pytest.mark.parametrize("undersampling", undersamplings)
def test_undersampling_normal(undersampling):
    """
    Testing the undersampling

    Args:
        undersampling (cls): the undersampling class
    """

    dataset = load_normal()

    X = dataset["data"]
    y = dataset["target"]

    params = undersampling.parameter_combinations()

    undersampling_obj = undersampling(**params[0])

    _, y_samp = undersampling_obj.sample(X, y)

    assert np.sum(y_samp) == len(y_samp) - np.sum(y_samp)

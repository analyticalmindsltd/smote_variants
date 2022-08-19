"""
Testing the parameterization.
"""

import numpy as np
import pytest

import smote_variants as sv

@pytest.mark.parametrize("smote_class", sv.get_all_oversamplers())
def test_parameters(smote_class):
    """
    Test the parameterization.

    Args:
        class_name (class): an oversampling class
    """
    random_state = np.random.RandomState(5)

    par_comb = smote_class.parameter_combinations()

    original_parameters = random_state.choice(par_comb)
    oversampler = smote_class(**original_parameters)
    parameters = oversampler.get_params()

    assert all(v == parameters[k] for k, v in original_parameters.items())

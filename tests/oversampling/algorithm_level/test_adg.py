"""
Testing the ADG.
"""
import numpy as np

import pytest

from smote_variants import ADG

from smote_variants.datasets import load_normal

def test_specific():
    """
    Oversampler specific testing
    """
    with pytest.raises(Exception):
        ADG(kernel='dummy')

    with pytest.raises(Exception):
        ADG(kernel='rbf_-6')

    dataset = load_normal()

    X_samp, _ = ADG(kernel='rbf_1').sample(dataset['data'],
                                           dataset['target'])

    assert len(X_samp) > 0

    X_samp, _ = ADG().sample(dataset['data'],
                             dataset['target'])

    assert len(X_samp) > 0

    with pytest.raises(Exception):
        ADG().check_early_stopping(0.0)

    with pytest.raises(Exception):
        ADG().step_9_16(partial_results={'Z_hat': np.array([[]])},
                        kernel_function='inner',
                        y=None)

"""
Testing the DSRBF.
"""

from smote_variants import DSRBF

from smote_variants.datasets import (load_all_min_noise)

def test_specific():
    """
    Oversampler specific testing
    """

    obj = DSRBF(hidden_range=(30, 40))

    dataset = load_all_min_noise()

    X_samp, _ = obj.sample(dataset['data'],
                            dataset['target'])

    assert len(X_samp) > 0

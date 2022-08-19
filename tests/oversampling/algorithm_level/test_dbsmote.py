"""
Testing the DBSMOTE.
"""

from smote_variants import DBSMOTE

from smote_variants.datasets import (load_all_min_noise,
                                     load_3_min_some_maj,
                                     load_separable)

def test_specific():
    """
    Oversampler specific testing
    """

    obj = DBSMOTE(db_iter_limit=1)

    dataset = load_3_min_some_maj()

    X_samp, _ = obj.sample(dataset['data'],
                            dataset['target'])

    assert len(X_samp) > 0

    obj = DBSMOTE(db_iter_limit=1)

    dataset = load_all_min_noise()

    X_samp, _ = obj.sample(dataset['data'],
                            dataset['target'])

    assert len(X_samp) > 0

    obj = DBSMOTE(db_iter_limit=1)

    dataset = load_separable()

    X_samp, _ = obj.sample(dataset['data'],
                            dataset['target'])

    assert len(X_samp) > 0

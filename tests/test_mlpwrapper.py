"""
Tests the operation of the MLPWrapper
"""

import logging

from sklearn import datasets

import smote_variants as sv

logger = logging.getLogger("smote_variants")
logger.setLevel(logging.ERROR)

def test_mlp_wrapper():
    """
    Testing the MLP wrapper.
    """
    dataset = datasets.load_wine()
    classifier = sv.classifiers.MLPClassifierWrapper()
    classifier.fit(dataset['data'], dataset['target'])

    assert classifier is not None

    pred = classifier.predict(dataset['data'])

    assert len(pred) == len(dataset['data'])

    pred_proba = classifier.predict_proba(dataset['data'])

    assert len(pred_proba) == len(dataset['data'])

    params = classifier.get_params()

    assert len(params) > 0

    clone = classifier.copy()

    assert sv.base.equal_dicts(params, clone.get_params())

"""
Testing the oversampling classifier.
"""

from sklearn import datasets

from smote_variants.base import equal_dicts
from smote_variants.classifiers import OversamplingClassifier, ErrorWarningClassifier

dataset = datasets.load_breast_cancer()

def test_oversamplingclassifier():
    """
    Testing the oversampling classifier
    """
    osc = OversamplingClassifier(oversampler=('smote_variants', 'SMOTE', {}),
                                    classifier=('sklearn.neighbors', 'KNeighborsClassifier', {}))

    osc.fit(dataset['data'], dataset['target'])

    pred = osc.predict(dataset['data'])

    assert len(pred) == len(dataset['data'])

    prob = osc.predict_proba(dataset['data'])

    assert len(prob) == len(dataset['data'])

    params = osc.get_params()

    assert len(params) > 0

    osc.set_params(**params)

    assert equal_dicts(params, osc.get_params())

    assert len(ErrorWarningClassifier().get_params()) > 0

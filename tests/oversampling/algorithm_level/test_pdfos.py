"""
Testing the PDFOS.
"""
import numpy as np

from smote_variants import PDFOS, FullRankTransformer

def test_specific():
    """
    Oversampler specific testing
    """
    obj = PDFOS()

    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5],
                    [5, 6], [6, 7], [8, 9]])
    y = np.array([0, 1, 0, 1, 0, 1, 0])

    assert obj.sample(X, y)[0].shape[0] > 0

    assert FullRankTransformer().transform(None) is None

"""
Testing the Assembled_SMOTE.
"""

import numpy as np

from smote_variants import Assembled_SMOTE

def test_specific():
    """
    Oversampler specific testing
    """

    obj = Assembled_SMOTE()

    vectors = [np.array([[1.0, 2.0]]), np.array([[2.0, 3.0]])]

    samples = obj.generate_samples_in_clusters(vectors=vectors,
                                                n_to_sample=4,
                                                nn_params={})
    assert len(samples) == 4

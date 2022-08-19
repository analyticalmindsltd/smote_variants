"""
Testing the SOI_CJ.
"""

from smote_variants import SOI_CJ

from smote_variants.datasets import load_illustration_2_class

def test_specific():
    """
    Oversampler specific testing
    """
    dataset = load_illustration_2_class()

    obj = SOI_CJ(method='interpolation')

    X_samp, _ = obj.sample(dataset['data'], dataset['target'])

    assert len(X_samp) > 0

    obj = SOI_CJ(method='jittering')

    X_samp, _ = obj.sample(dataset['data'], dataset['target'])

    assert len(X_samp) > 0

    cluster_idx = set([1, 2, 3])
    cluster_jdx = set([3, 4, 5, 6, 7])
    intersection = set([3])

    cluster_idx, cluster_jdx = obj.rearrange_clusters(cluster_idx,
                                                      cluster_jdx,
                                                      intersection)

    assert len(cluster_jdx) > 0

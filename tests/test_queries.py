"""
Testing the queries.
"""

import smote_variants as sv

def test_queries():
    """
    Testing the queries.
    """
    assert len(sv.get_all_oversamplers()) > 0
    assert len(sv.get_all_oversamplers(n_quickest=5)) == 5
    assert len(sv.get_all_noisefilters()) > 0
    assert len(sv.get_multiclass_oversamplers()) > 0
    assert len(sv.get_multiclass_oversamplers(n_quickest=5)) == 5
    assert len(sv.get_metric_learning_oversamplers()) > 0
    assert len(sv.get_metric_learning_oversamplers(n_quickest=5)) == 5
    assert len(sv.get_simplex_sampling_oversamplers()) > 0
    assert len(sv.get_simplex_sampling_oversamplers(n_quickest=5)) == 5
    assert len(sv.get_simplex_sampling_oversamplers(
                            within_simplex_sampling='random')) > 0
    assert len(sv.get_simplex_sampling_oversamplers(
                            within_simplex_sampling=['random'])) > 0
    assert len(sv.get_simplex_sampling_oversamplers(
                        exclude_within_simplex_sampling='deterministic')) > 0
    assert len(sv.get_simplex_sampling_oversamplers(
                        exclude_within_simplex_sampling=['deterministic'])) > 0
    assert len(sv.get_simplex_sampling_oversamplers(
                                            with_gaussian_component=True)) > 0
    assert len(sv.get_simplex_sampling_oversamplers(
                                            with_gaussian_component=False)) > 0
    assert len(sv.get_simplex_sampling_oversamplers(n_dim_range=2)) > 0
    assert len(sv.get_simplex_sampling_oversamplers(n_dim_range=[1, 4])) > 0

    oversamplers = sv.get_all_oversamplers()[:5]
    assert len(sv.generate_parameter_combinations(oversamplers,
                                                result_format='dict')) > 0

    assert len(sv.generate_parameter_combinations(oversamplers,
                                        result_format='smote_variants')) > 0

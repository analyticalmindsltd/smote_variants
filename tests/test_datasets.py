"""
Test the datasets module.
"""

from smote_variants.datasets import (load_1_dim,
                                     load_illustration_2_class,
                                     load_illustration_3_class,
                                     load_illustration_4_class,
                                     load_normal,
                                     load_same_num,
                                     load_some_min_some_maj,
                                     load_1_min_some_maj,
                                     load_2_min_some_maj,
                                     load_3_min_some_maj,
                                     load_4_min_some_maj,
                                     load_5_min_some_maj,
                                     load_1_min_1_maj,
                                     load_repeated,
                                     load_all_min_noise,
                                     load_separable,
                                     load_linearly_dependent,
                                     load_alternating,
                                     load_high_dim)

def test_dataset_loaders():
    """
    Testing dataset loaders
    """

    for loader in [load_1_dim,
                    load_illustration_2_class,
                    load_illustration_3_class,
                    load_illustration_4_class,
                    load_normal,
                    load_same_num,
                    load_some_min_some_maj,
                    load_1_min_some_maj,
                    load_2_min_some_maj,
                    load_3_min_some_maj,
                    load_4_min_some_maj,
                    load_5_min_some_maj,
                    load_1_min_1_maj,
                    load_repeated,
                    load_all_min_noise,
                    load_separable,
                    load_linearly_dependent,
                    load_alternating,
                    load_high_dim]:
        dataset = loader()
        assert len(dataset) == 3

"""
This module tests the functionalities related to the processing of raw data.
"""
import os
import shutil

from sklearn import datasets

from smote_variants.evaluation import (evaluate_oversamplers,
                                        datasets_in_cache,
                                        load_dataset_data,
                                        create_summary,
                                        is_deterministic)

dataset_0 = datasets.load_breast_cancer()
dataset_0['name'] = 'breast_cancer'
dataset_1 = datasets.load_iris()
dataset_1['name'] = 'iris'
dataset_1['target'][dataset_1['target'] == 2] = 1

datasets = [dataset_0, dataset_1]

oversamplers = [('smote_variants', 'SMOTE', {'n_jobs': 1, 'random_state': 5}),
            ('smote_variants', 'distance_SMOTE', {'n_jobs': 1,
                                                    'n_neighbors': 9,
                                                    'random_state': 5})]

classifiers = [('sklearn.tree', 'DecisionTreeClassifier', {'random_state': 5}),
                ('sklearn.neighbors', 'KNeighborsClassifier', {'n_jobs': 1})]

cache_path = os.path.join('.', 'test_path')

def test_all_functions():
    """
    Testing all processing functions
    """
    shutil.rmtree(cache_path, ignore_errors=True)

    evaluate_oversamplers(datasets=[dataset_0, dataset_1],
                            oversamplers=oversamplers,
                            classifiers=classifiers,
                            cache_path=cache_path,
                            validator_params={'n_repeats': 2, 'n_splits': 2, 'random_state': 5},
                            scaler=('sklearn.preprocessing', 'StandardScaler', {}),
                            n_jobs=2,
                            parse_results=False)

    read_datasets = datasets_in_cache(cache_path)

    assert len(read_datasets) == len(datasets)
    assert 'breast_cancer' in read_datasets
    assert 'iris' in read_datasets

    assert len(load_dataset_data(read_datasets['iris'])) == 16
    assert len(load_dataset_data(read_datasets['breast_cancer'])) == 16

    raw_data = load_dataset_data(read_datasets['iris'])

    assert len(create_summary(raw_data)) == 4

    shutil.rmtree(cache_path, ignore_errors=True)

def test_is_deterministic():
    """
    Testing the is_deterministic function
    """
    assert is_deterministic({}) is None
    assert is_deterministic({'ss_params': {'within_simplex_sampling': 'deterministic'}}) \
                == 'deterministic'

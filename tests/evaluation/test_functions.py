"""
Testing the evaluation functions.
"""

import os
import shutil
import glob

from sklearn import datasets

from smote_variants.base import equal_dicts

from smote_variants.evaluation import (evaluate_oversamplers, model_selection,
                                        execute_1_job, execute_1_os, execute_1_eval,
                                        do_clean_up, do_parse_results, Folding)

dataset_0 = datasets.load_breast_cancer()
dataset_0['name'] = 'breast_cancer'
dataset_1 = datasets.load_iris()
dataset_1['name'] = 'iris'
dataset_1['target'][dataset_1['target'] == 2] = 1

oversamplers = [('smote_variants', 'SMOTE', {'n_jobs': 1, 'random_state': 5}),
            ('smote_variants', 'SMOTE', {'n_jobs': 1, 'n_neighbors': 9, 'random_state': 5})]

classifiers = [('sklearn.tree', 'DecisionTreeClassifier', {'random_state': 5}),
                ('sklearn.neighbors', 'KNeighborsClassifier', {'n_jobs': 1})]

cache_path = os.path.join('.', 'test_path')

folding = Folding(dataset_0)

def test_execute_1_job():
    """
    Testing the execution of 1 evaluation job.
    """
    fold = next(folding.fold())
    result = execute_1_job(fold,
                            oversamplers,
                            ('sklearn.preprocessing', 'StandardScaler', {}),
                            classifiers,
                            serialization='json')
    assert len(result) > 0

def test_execute_1_subjob():
    """
    Testing the execution of 1 oversampling and 1 evaluation subjob.
    """
    fold = next(folding.fold())
    result = execute_1_os(fold,
                            oversamplers,
                            ('sklearn.preprocessing', 'StandardScaler', {}),
                            serialization='json',
                            cache_path=None)
    assert len(result) > 0

    result = execute_1_eval(result,
                            classifiers,
                            serialization='json',
                            cache_path=None)
    assert len(result) > 0

def test_do_clean_up():
    """
    Testing the clean-up functionalities.
    """
    os.makedirs(cache_path, exist_ok=True)

    with open(os.path.join(cache_path, 'fold_dummy.txt'),
                'wt', encoding='utf-8') as file:
        file.write('dummy')

    with open(os.path.join(cache_path, 'oversampling_dummy.txt'),
                'wt', encoding='utf-8') as file:
        file.write('dummy')

    with open(os.path.join(cache_path, 'evaluation_dummy.txt'),
                'wt', encoding='utf-8') as file:
        file.write('dummy')

    do_clean_up(cache_path, 'oversamplings')

    assert len(glob.glob(os.path.join(cache_path, 'oversampling*'))) == 0
    assert len(glob.glob(os.path.join(cache_path, 'fold*'))) > 0
    assert len(glob.glob(os.path.join(cache_path, 'evaluation*'))) > 0

    do_clean_up(cache_path, 'all')

    assert len(glob.glob(os.path.join(cache_path, 'oversampling*'))) == 0
    assert len(glob.glob(os.path.join(cache_path, 'fold*'))) == 0
    assert len(glob.glob(os.path.join(cache_path, 'evaluation*'))) > 0

    shutil.rmtree(cache_path)

def test_evalute_oversamplers():
    """
    Testing the evaluation of oversamplers.
    """
    results_0 = evaluate_oversamplers(datasets=[dataset_0, dataset_1],
            oversamplers=oversamplers,
            classifiers=classifiers,
            cache_path=None,
            validator_params={'n_repeats': 2, 'n_splits': 5, 'random_state': 5},
            scaler=('sklearn.preprocessing', 'StandardScaler', {}),
            n_jobs=2,
            parse_results=True)

    shutil.rmtree(cache_path, ignore_errors=True)

    results_1 = evaluate_oversamplers(datasets=[dataset_0, dataset_1],
            oversamplers=oversamplers,
            classifiers=classifiers,
            cache_path=cache_path,
            validator_params={'n_repeats': 2, 'n_splits': 5, 'random_state': 5},
            scaler=('sklearn.preprocessing', 'StandardScaler', {}),
            n_jobs=2,
            parse_results=False)

    results_1 = do_parse_results(results_1, False, serialization='json')
    results_1 = do_parse_results(results_1, True, serialization='json')

    shutil.rmtree(cache_path, ignore_errors=True)

    assert results_0['acc_mean'].max() == results_1['acc_mean'].max()

def test_model_selection():
    """
    Testing the model selection.
    """
    samp_0, clas_0 = model_selection(dataset=dataset_0,
            oversamplers=oversamplers,
            classifiers=classifiers,
            cache_path=None,
            validator_params={'n_repeats': 2, 'n_splits': 5, 'random_state': 5},
            scaler=('sklearn.preprocessing', 'StandardScaler', {}),
            n_jobs=2,
            score='auc')

    shutil.rmtree(cache_path, ignore_errors=True)

    samp_1, clas_1 = model_selection(dataset=dataset_0,
            oversamplers=oversamplers,
            classifiers=classifiers,
            cache_path=cache_path,
            validator_params=None,
            scaler=('sklearn.preprocessing', 'StandardScaler', {}),
            n_jobs=2,
            score='auc')

    shutil.rmtree(cache_path, ignore_errors=True)

    assert equal_dicts(samp_0.get_params(), samp_1.get_params())
    assert equal_dicts(clas_0.get_params(), clas_1.get_params())

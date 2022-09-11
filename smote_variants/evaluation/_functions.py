"""
This module implements the model selection functionalities.
"""

import os
import glob
from ast import literal_eval

import datetime
import logging
import gc

from joblib import Parallel, delayed

import numpy as np
import pandas as pd

from ..base import load_dict, instantiate_obj
from ..base import all_scores
from ._folding import Folding
from ._sampling import SamplingJob
from ._evaluation import EvaluationJob
from .._logger import logger
from ._parallelization import ThreadTimeoutProcessPool

__all__ = ['evaluate_oversamplers',
            'model_selection',
            'cross_validation',
            'execute_1_job',
            'execute_1_os',
            'execute_1_eval',
            'do_clean_up',
            'do_parse_results',
            '_verbose_logging']

def execute_1_job(fold,
                    oversamplers,
                    scaler,
                    classifiers,
                    serialization):
    """
    Executes one oversampling and evaluation job

    Args:
        fold (dict/str): a fold
        oversamplers (list): list of oversamplers
        scaler (tuple): specification of a scaler
        classifier (list): list of classifiers
        serialization (str): serialization type ('json'/'pickle')

    Returns:
        list: the results
    """
    results = []
    for oversampler in oversamplers:
        job = SamplingJob(fold,
                            oversampler[1],
                            oversampler[2],
                            scaler=scaler,
                            cache_path=None,
                            serialization=serialization)
        result = job.do_oversampling()

        job = EvaluationJob(result,
                            classifiers,
                            cache_path=None,
                            serialization=serialization)
        results.extend(job.do_evaluation())
    return results

def execute_1_os(fold,
                    oversamplers,
                    scaler,
                    cache_path,
                    serialization):
    """
    Executes one oversampling job

    Args:
        fold (dict/str): a fold
        oversamplers (list): list of oversamplers
        scaler (tuple): specification of a scaler
        cache_path (str): the cache path
        serialization (str): serialization type ('json'/'pickle')

    Returns:
        list: the results
    """
    #import os
    #os.environ['OPENBLAS_NUM_THREADS'] = '1'
    #os.environ['MKL_NUM_THREADS'] = '1'
    #os.environ['BLIS_NUM_THREADS'] = '1'
    #os.environ['OMP_NUM_THREADS'] = '1'
    #os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    #os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

    logger.setLevel(logging.CRITICAL)

    results = []
    for oversampler in oversamplers:
        job = SamplingJob(fold,
                            oversampler[1],
                            oversampler[2],
                            scaler=scaler,
                            cache_path=cache_path,
                            serialization=serialization)

        result = job.do_oversampling()

        results.append(result)


    return results

def execute_1_eval(oversampling,
                    classifiers,
                    cache_path,
                    serialization):
    """
    Executes one evaluation job

    Args:
        oversampling (dict/str): an oversampling
        classifiers (list): list of classifiers
        cache_path (str): the cache path
        serialization (str): serialization type ('json'/'pickle')

    Returns:
        list: the results
    """
    #import os
    #os.environ['OPENBLAS_NUM_THREADS'] = '1'
    #os.environ['MKL_NUM_THREADS'] = '1'
    #os.environ['BLIS_NUM_THREADS'] = '1'
    #os.environ['OMP_NUM_THREADS'] = '1'

    #logger.setLevel(logging.CRITICAL)

    job = EvaluationJob(oversampling,
                            classifiers,
                            cache_path=cache_path,
                            serialization=serialization)
    result = job.do_evaluation()
    return result

def do_clean_up(cache_path, clean_up):
    """
    Remove unnecessary files

    Args:
        cache_path (str): the cache path
        clean_up (str): 'oversamplings'/'all'
    """

    if clean_up in ['oversamplings', 'all']:
        files = glob.glob(os.path.join(cache_path, 'oversampling*'))
        for file in files:
            os.remove(file)
    if clean_up == 'all':
        files = glob.glob(os.path.join(cache_path, 'fold*'))
        for file in files:
            os.remove(file)

def _pivot_best_scores(pdf):
    """
    Pivot the best parameters for each oversampler and classifier

    Args:
        pdf (pd.DataFrame): the results averaged over folds

    Returns:
        pd.DataFrame: the pivoted results
    """
    results_dict_classifier = {}
    results_dict_oversampler = {}
    score_means = {}
    score_stds = {}
    for score in all_scores:
        tmp = pdf[pdf[score + '_mean'] == pdf[score + '_mean'].max()]

        results_dict_classifier[f'{score}_classifier_params'] = [None]
        results_dict_classifier[f'{score}_oversampler_params'] = [None]
        score_means[f'{score}_mean'] = [None]
        score_stds[f'{score}_std'] = [None]

        if len(tmp) > 0:
            results_dict_classifier[f'{score}_classifier_params'] =\
                                     [tmp.iloc[0]['classifier_params']]
            results_dict_classifier[f'{score}_oversampler_params'] =\
                                     [tmp.iloc[0]['oversampler_params']]
            score_means[f'{score}_mean'] = [tmp.iloc[0][f'{score}_mean']]
            score_stds[f'{score}_std'] = [tmp.iloc[0][f'{score}_std']]

    return pd.DataFrame({**results_dict_classifier,
                         **results_dict_oversampler,
                         **score_means,
                         **score_stds})

def _explode_entries(pdf):
    pdf = pd.concat([pdf, pdf['scores'].apply(pd.Series)], axis=1)
    pdf['dataset'] = pdf['fold_descriptor'].apply(lambda x: x['name'])
    pdf['fold_idx'] = pdf['fold_descriptor'].apply(lambda x: x['fold_idx'])
    pdf['repeat_idx'] = pdf['fold_descriptor'].apply(lambda x: x['repeat_idx'])
    pdf['oversampler_params_key'] = pdf['oversampler_params'].apply(str)
    pdf['classifier_params_key'] = pdf['classifier_params'].apply(str)
    return pdf


def do_parse_results(results, parse_results, serialization):
    """
    Parse the results if needed.

    Args:
        results (list): list of results
        parse_results (bool): flag, if True, files are read
        serialization (str): serialization type 'json'/'pickle'

    Returns:
        list/pd.DataFrame: the unparsed or parsed results
    """
    if parse_results is True:
        if isinstance(results[0], str):
            loaded = []
            for res in results:
                loaded.append(load_dict(res, serialization=serialization))
            results = loaded

        pdf = pd.DataFrame(results)
        pdf_prep = _explode_entries(pdf)

        aggregations = {**{score + '_mean': (score, 'mean') for score in all_scores},
                        **{score + '_std': (score, 'std') for score in all_scores}}

        pdf_avg = pdf_prep.groupby(['dataset', 'oversampler_module', 'oversampler',
                                'classifier', 'oversampler_params_key',
                                'classifier_module',
                                'classifier_params_key'])\
                            .agg(**aggregations)\
                            .reset_index(drop=False)
        pdf_avg['oversampler_params'] = pdf_avg['oversampler_params_key']
        pdf_avg['classifier_params'] = pdf_avg['classifier_params_key']
        pdf_avg = pdf_avg.drop(['oversampler_params_key',
                                'classifier_params_key'], axis='columns')
        return pdf_avg
    return results

def execute_parallel_oversampling(*, folding,
                                    oversamplers,
                                    scaler,
                                    cache_path_tmp,
                                    serialization,
                                    n_jobs,
                                    timeout):
    """
    Execute the oversampling in parallel

    Args:
        folding (Folding): a folding object
        oversamplers (list): the list of oversamplers
        scaler (tuple): the scaler specification
        cache_path_tmp (str): the cache path
        serialization (str): the serialization type
        n_jobs (int): the number of jobs
        timeout (float): the timeout time (seconds)

    Returns:
        list: the results
    """
    all_folds = folding.folding_files()
    jobs = []

    for fold in all_folds:
        for oversampler in oversamplers:
            jobs.append(SamplingJob(fold,
                            oversampler[1],
                            oversampler[2],
                            scaler=scaler,
                            cache_path=cache_path_tmp,
                            serialization=serialization))

    np.random.shuffle(jobs)

    ttpp = ThreadTimeoutProcessPool(n_jobs, timeout)

    results_os = ttpp.execute(jobs=jobs)

    return results_os

def cached_evaluation(*, folding,
                        oversamplers,
                        classifiers,
                        scaler,
                        cache_path,
                        serialization,
                        n_jobs,
                        timeout,
                        clean_up):
    """
    Executes cached evaluation

    Args:
        folding (obj): a folding object
        oversamplers (list): list of oversampling classes/objects
        classifiers (list): list of classifier objects
        scaler (tuple): scaler object
        cache_path (str): path to a cache directory
        serialization (str): serialization method 'json'/'pickle'
        n_jobs (int): number of parallel jobs
        timeout (float): the timeout in seconds (-1: no timeout)
        clean_up (str): 'oversamplings'/'all'

    Returns:
        list: the results
    """
    folding.cache_foldings()

    cache_path_tmp = os.path.join(cache_path, folding.properties['name'])

    if n_jobs > 1:
        results_os = execute_parallel_oversampling(folding=folding,
                                                    oversamplers=oversamplers,
                                                    scaler=scaler,
                                                    cache_path_tmp=cache_path_tmp,
                                                    serialization=serialization,
                                                    n_jobs=n_jobs,
                                                    timeout=timeout)
    else:
        results_os = []
        for fold in folding.fold():
            results = execute_1_os(fold,
                                    oversamplers,
                                    scaler,
                                    cache_path_tmp,
                                    serialization)
            results_os.extend(results)

    if n_jobs > 1:
        results = Parallel(n_jobs=n_jobs,
                            batch_size=10)(delayed(execute_1_eval)(overs,
                                                                classifiers,
                                                                cache_path_tmp,
                                                                serialization)
                                        for overs in results_os)
    else:
        results = [execute_1_eval(overs,
                                    classifiers,
                                    cache_path_tmp,
                                    serialization) for overs in results_os]

    do_clean_up(cache_path_tmp, clean_up)

    return results

def _verbose_logging(message, verbosity):
    if verbosity == 0:
        logger.info(message)
    else:
        print(f"{str(datetime.datetime.now())}: {message}")

def _unify_lists(list_of_lists):
    result = []
    for tmp in list_of_lists:
        result.extend(tmp)

    return result

def evaluate_oversampler_on_dataset(dataset,
                                    oversamplers,
                                    classifiers,
                                    *,
                                    cache_path=None,
                                    validator_params=None,
                                    scaler=('sklearn.preprocessing',
                                            'StandardScaler',
                                            {}),
                                    serialization='json',
                                    clean_up='oversamplings',
                                    n_jobs=1,
                                    timeout=-1,
                                    explicit_gc=False):
    """
    Evaluates oversampling techniques using various classifiers on one dataset

    Args:
        datasets (list): list of datasets and/or dataset loaders - a dataset
                            is a dict with 'data', 'target' and 'name' keys
        oversamplers (list): list of oversampling classes/objects
        classifiers (list): list of classifier objects
        cache_path (str): path to a cache directory
        validator_params (dict/None): parameters of the validator
        scaler (tuple): scaler object
        serialization (str): serialization method 'json'/'pickle'
        clean_up (str): 'oversamplings'/'all'
        n_jobs (int): number of parallel jobs
        timeout (float/None): the timeout for the oversampling jobs, negative
                                or None means no timeout
        explicit_gc (bool): whether to run garbage collection expliticlty

    Returns:
        pd.DataFrame: the evaluation results
    """

    all_results = []
    folding = Folding(dataset,
                        cache_path=cache_path,
                        validator_params=validator_params,
                        serialization=serialization)

    if cache_path is None:
        all_results.extend(Parallel(n_jobs=n_jobs,
                                    batch_size=1)(delayed(execute_1_job)(fold,
                                                                    oversamplers,
                                                                    scaler,
                                                                    classifiers,
                                                                    serialization)
                                                    for fold in folding.fold()))

    else:
        all_results.extend(cached_evaluation(folding=folding,
                                            oversamplers=oversamplers,
                                            classifiers=classifiers,
                                            scaler=scaler,
                                            cache_path=cache_path,
                                            serialization=serialization,
                                            n_jobs=n_jobs,
                                            timeout=timeout,
                                            clean_up=clean_up))

    # optionally explicitly execute garbage collection after each dataset
    if explicit_gc:
        gc.collect()

    return all_results

def evaluate_oversamplers(datasets,
                            oversamplers,
                            classifiers,
                            *,
                            cache_path=None,
                            validator_params=None,
                            scaler=('sklearn.preprocessing',
                                    'StandardScaler',
                                    {}),
                            serialization='json',
                            clean_up='oversamplings',
                            n_jobs=1,
                            timeout=-1,
                            parse_results=True,
                            explicit_gc=False,
                            verbosity=1):
    """
    Evaluates oversampling techniques using various classifiers on
    various datasets

    Args:
        datasets (list): list of datasets and/or dataset loaders - a dataset
                            is a dict with 'data', 'target' and 'name' keys
        oversamplers (list): list of oversampling classes/objects
        classifiers (list): list of classifier objects
        cache_path (str): path to a cache directory
        validator_params (dict/None): parameters of the validator
        scaler (tuple): scaler object
        serialization (str): serialization method 'json'/'pickle'
        clean_up (str): 'oversamplings'/'all'
        n_jobs (int): number of parallel jobs
        timeout (float/None): the timeout for the oversampling jobs, negative
                                or None means no timeout
        parse_results (bool): whether to parse the results when caching happens
        explicit_gc (bool): whether to run garbage collection expliticlty
        verbosity (int): verbosity level

    Returns:
        pd.DataFrame: the evaluation results

    Example:
        import smote_variants as sv
        import imbalanced_datasets as imbd

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier

        datasets= [imbd.load_glass2(), imbd.load_ecoli4()]

        oversamplers= [('smote_variants', 'SMOTE_ENN', {}),
                       ('smote_variants', 'NEATER', {}),
                       ('smote_variants', 'Lee', {})]

        classifiers= [('sklearn.neighbors', 'KNeighborsClassifier',
                      {'n_neighbors': 3}),
                      ('sklearn.neighbors', 'KNeighborsClassifier',
                      {'n_neighbors': 5}),
                      ('sklearn.tree', 'DecisionTreeClassifier', {})]

        results= evaluate_oversamplers(datasets,
                                    oversamplers,
                                    classifiers)

    """

    if validator_params is None:
        validator_params = {'n_repeats': 2, 'n_splits': 5, 'random_state': 5}

    all_results = []

    for dataset in datasets:
        _verbose_logging(f"processing dataset: {dataset['name']}", verbosity)

        all_results.extend(evaluate_oversampler_on_dataset(dataset=dataset,
                                                            oversamplers=oversamplers,
                                                            classifiers=classifiers,
                                                            cache_path=cache_path,
                                                            validator_params=validator_params,
                                                            scaler=scaler,
                                                            serialization=serialization,
                                                            clean_up=clean_up,
                                                            n_jobs=n_jobs,
                                                            timeout=timeout,
                                                            explicit_gc=explicit_gc))

    all_results = _unify_lists(all_results)
    return do_parse_results(all_results, parse_results, serialization)

def model_selection(dataset,
                    oversamplers,
                    classifiers,
                    *,
                    score='auc',
                    cache_path=None,
                    validator_params=None,
                    scaler=('sklearn.preprocessing',
                            'StandardScaler',
                            {}),
                    serialization='json',
                    clean_up='oversamplings',
                    n_jobs=1,
                    timeout=-1):
    """
    Executes model selection using various classifiers on
    various datasets

    Args:
        datasets (list): list of datasets and/or dataset loaders - a dataset
                            is a dict with 'data', 'target' and 'name' keys
        oversamplers (list): list of oversampling classes/objects
        classifiers (list): list of classifier objects
        score (str): the score to drive the model selection
        cache_path (str): path to a cache directory
        validator_params (dict/None): parameters of the validator
        scaler (tuple): scaler object
        serialization (str): serialization method 'json'/'pickle'
        clean_up (str): 'oversamplings'/'all'
        n_jobs (int): number of parallel jobs
        timeout (float/None): the timeout for the oversampling jobs, negative
                                or None means no timeout

    Returns:
        pd.DataFrame: the evaluation results

    Example:
        import smote_variants as sv
        import imbalanced_datasets as imbd

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier

        datasets= [imbd.load_glass2(), imbd.load_ecoli4()]

        oversamplers= [('smote_variants', 'SMOTE_ENN', {}),
                        ('smote_variants', 'NEATER', {}),
                        ('smote_variants', 'Lee', {})]

        classifiers= [('sklearn.neighbors', 'KNeighborsClassifier',
                      {'n_neighbors': 3}),
                      ('sklearn.neighbors', 'KNeighborsClassifier',
                      {'n_neighbors': 5}),
                      ('sklearn.tree', 'DecisionTreeClassifier', {})]

        results= evaluate_oversamplers(datasets,
                                       oversamplers,
                                       classifiers)
    """
    pdf = evaluate_oversamplers(datasets=[dataset],
                                    oversamplers=oversamplers,
                                    classifiers=classifiers,
                                    cache_path=cache_path,
                                    validator_params=validator_params,
                                    scaler=scaler,
                                    serialization=serialization,
                                    clean_up=clean_up,
                                    n_jobs=n_jobs,
                                    parse_results=True,
                                    timeout=timeout)

    pdf = pdf.groupby(['dataset', 'oversampler', 'oversampler_module',
                        'classifier', 'classifier_module'])\
                        .apply(_pivot_best_scores)\
                        .reset_index(drop=False)\
                        .drop('level_5', axis='columns')

    best_row = pdf[pdf[f'{score}_mean'] == pdf[f'{score}_mean'].max()].iloc[0]

    oversampler = (best_row['oversampler_module'],
                    best_row['oversampler'],
                    literal_eval(best_row[f'{score}_oversampler_params']))
    classifier = (best_row['classifier_module'],
                    best_row['classifier'],
                    literal_eval(best_row[f'{score}_classifier_params']))

    return instantiate_obj(oversampler), instantiate_obj(classifier)

def cross_validation():
    """
    dont really know yet
    """

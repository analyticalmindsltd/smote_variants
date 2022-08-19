"""
This module implements the model selection functionalities.
"""

import os
import glob
from ast import literal_eval

from joblib import Parallel, delayed

import pandas as pd

from ..base import load_dict, instantiate_obj
from ..base import all_scores
from ._folding import Folding
from ._sampling import SamplingJob
from ._evaluation import EvaluationJob

__all__ = ['evaluate_oversamplers',
            'model_selection',
            'cross_validation',
            'execute_1_job',
            'execute_1_os',
            'execute_1_eval',
            'do_clean_up',
            'do_parse_results']

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
    for oversampler in oversamplers:
        job = SamplingJob(fold,
                            oversampler[1],
                            oversampler[2],
                            scaler=scaler,
                            cache_path=cache_path,
                            serialization=serialization)
        result = job.do_oversampling()
    return result

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
    pdf['database'] = pdf['fold_descriptor'].apply(lambda x: x['name'])
    pdf['fold_idx'] = pdf['fold_descriptor'].apply(lambda x: x['fold_idx'])
    pdf['repeat_idx'] = pdf['fold_descriptor'].apply(lambda x: x['repeat_idx'])
    pdf['oversampler_params'] = pdf['oversampler']
    pdf['oversampler'] = pdf['oversampler_params'].apply(lambda x: x['class_name'])
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

        pdf_avg = pdf_prep.groupby(['database', 'oversampler',
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
                            parse_results=True):
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
        parse_results (bool): whether to parse the results when caching happens
        asf (bool): asdf

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
        folding = Folding(dataset,
                    cache_path=cache_path,
                    validator_params=validator_params,
                    serialization=serialization)

        if cache_path is None:
            results = Parallel(n_jobs=n_jobs,
                                batch_size=1)(delayed(execute_1_job)(fold,
                                                                oversamplers,
                                                                scaler,
                                                                classifiers,
                                                                serialization)
                                                for fold in folding.fold())

            for result in results:
                all_results.extend(result)
        else:
            folding.cache_foldings()

            cache_path = os.path.join(cache_path, folding.properties['name'])

            results = Parallel(n_jobs=n_jobs,
                                batch_size=1)(delayed(execute_1_os)(fold,
                                                            oversamplers,
                                                            scaler,
                                                            cache_path,
                                                            serialization)
                                        for fold in folding.folding_files())

            results = Parallel(n_jobs=n_jobs,
                                batch_size=1)(delayed(execute_1_eval)(overs,
                                                                    classifiers,
                                                                    cache_path,
                                                                    serialization)
                                            for overs in results)

            for result in results:
                all_results.extend(result)

            do_clean_up(cache_path, clean_up)

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
                    n_jobs=1):
    """
    Evaluates model selection using various classifiers on
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
        parse_results (bool): whether to parse the results when caching happens

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
                                    parse_results=True)

    pdf = pdf.groupby(['database', 'oversampler',
                                    'classifier', 'classifier_module'])\
                        .apply(_pivot_best_scores)\
                        .reset_index(drop=False)\
                        .drop('level_4', axis='columns')

    best_mask = pdf[f'{score}_mean'] == pdf[f'{score}_mean'].max()
    best_row = pdf[best_mask].iloc[0]

    oversampler = ('smote_variants',
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

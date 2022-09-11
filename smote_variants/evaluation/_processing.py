"""
This module contains the code used to process the evaluation data.
"""

import os
import glob

import pandas as pd

from ..base import load_dict

__all__ = ['datasets_in_cache',
            'load_dataset_data',
            'create_summary',
            'is_deterministic']

def datasets_in_cache(cache_path):
    """
    Determine all datasets available in the cache

    Args:
        cache_path (str): the cache path

    Returns:
        dict: {dataset_name: path} dictionary
    """
    pattern = os.path.join(cache_path, '*')
    dataset_paths = glob.glob(pattern)
    dataset_dict = {}
    for dataset_path in dataset_paths:
        tmp = dataset_path.split(os.path.sep)
        dataset_dict[tmp[-1]] = dataset_path

    return dataset_dict

def load_dataset_data(path):
    """
    Load the results for a given dataset.

    Args:
        path (str): the path of the dataset results

    Returns:
        pd.DataFrame: all results of the dataset
    """
    files = glob.glob(os.path.join(path, 'evaluation*'))
    data = [load_dict(file) for file in files]
    return pd.DataFrame(data)

def is_deterministic(params):
    """
    Determines if an oversampling technique parameterization is deterministic.

    Args:
        params (dict): the parameterization

    Returns:
        str/None: the within_simplex_sampling parameter
    """
    if 'ss_params' in params:
        return params['ss_params']['within_simplex_sampling']
    return None

def _extract_field(pdf, field, subfield):
    pdf[subfield] = pdf[field].apply(lambda x: x[subfield])
    return pdf

def _extract_attributes(pdf, scores):
    """
    Extract some scores and attributes from the inner dictionaries.

    Args:
        pdf (pd.DataFrame): the evaluation results for a dataset

    Returns:
        pd.DataFrame: the results with some attributes added
    """

    pdf['db_name'] = pdf['fold_descriptor'].apply(lambda x: x['name'])
    pdf['split_idx'] = pdf['fold_descriptor'].apply(lambda x: x['split_idx'])
    pdf['repeat_idx'] = pdf['fold_descriptor'].apply(lambda x: x['repeat_idx'])
    pdf['fold_idx'] = pdf['fold_descriptor'].apply(lambda x: x['fold_idx'])

    pdf['deterministic'] = pdf['oversampler_params'].apply(is_deterministic)

    for score in scores:
        pdf = _extract_field(pdf, "scores", score)

    return pdf

def _serialize_data(pdf,
                    columns=None,
                    keep=None):
    """
    Group and serialize some attributes

    Args:
        pdf (pd.DataFrame): the dataframe to process
        columns (list): the columns to serialize
        keep (list): the columns to keep

    Returns:
        pd.DataFrame: the grouped dataframe with some attributes serialized and kept
    """
    if keep is None:
        keep = ['oversampler_params', 'classifier_params']
    return pd.Series({**{col: pdf.sort_values('fold_idx')[col].values.tolist() for col in columns},
                     **pdf.reset_index(drop=True).iloc[0][keep].to_dict()})

def create_summary(data,
                    scores=None,
                    columns_to_serialize=None):
    """
    Create a summary of the results by grouping and serializing the results for various folds

    Args:
        data (pd.DataFrame): the raw evaluation results to summarize
        scores (list): the scores to extract
        columns_to_serialize (list): the columns to serialize

    Returns:
        pd.DataFrame: the summary dataframe
    """
    if scores is None:
        scores = ['auc']

    if columns_to_serialize is None:
        columns_to_serialize = ['auc', 'y_pred', 'y_test',
                                'oversampling_error',
                                'oversampling_warning',
                                'oversampling_runtime']
    pdf = _extract_attributes(data, scores)

    pdf['oversampler_params_str'] = pdf['oversampler_params'].apply(str)
    pdf['classifier_params_str'] = pdf['classifier_params'].apply(str)

    pdf = pdf.groupby(['db_name',
                'oversampler_params_str',
                'classifier_params_str',
                'oversampler_module',
                'oversampler',
                'classifier_module',
                'classifier',
                'deterministic'])\
    .apply(lambda x: _serialize_data(x, columns_to_serialize))\
    .reset_index(drop=False)\
    .drop(['oversampler_params_str', 'classifier_params_str'], axis='columns')

    return pdf

"""
Test the sampling.
"""

import os
import shutil

import numpy as np

from sklearn import datasets

from smote_variants.evaluation import Folding, SamplingJob
import smote_variants

DATASET = datasets.load_breast_cancer()
DATASET['name'] = 'breast_cancer'

DATASET_WARNING = {'data': np.array([[1, 2], [2, 3],
                                     [1, 2], [2, 3],
                                     [1, 2], [2, 3],
                                     [1, 2], [2, 3],
                                     [1, 2], [2, 3],
                                     [1, 2], [2, 3],
                                     [1, 2], [2, 3],
                                     [1, 2], [2, 3]]),
                   'target': np.array([0, 1,
                                      0, 1,
                                      0, 1,
                                      0, 1,
                                      0, 1,
                                      0, 1,
                                      0, 1,
                                      0, 1]),
                   'name': 'warning dataset'}

cache_path = os.path.join('.', 'test_path')

def test_sampling():
    """
    Testing the sampling.
    """
    shutil.rmtree(cache_path, ignore_errors=True)

    folding = Folding(DATASET,
                    cache_path=cache_path,
                    reset=True)
    oversampler = 'SMOTE'
    oversampler_params = {}

    for fold in folding.fold():
        sjob = SamplingJob(fold,
                        oversampler,
                        oversampler_params,
                        cache_path=cache_path)

        result = sjob.do_oversampling()
        assert os.path.exists(result)

    for fold in folding.fold():
        sjob = SamplingJob(fold,
                        oversampler,
                        oversampler_params,
                        cache_path=cache_path)

        result = sjob.do_oversampling()
        assert os.path.exists(result)

    shutil.rmtree(cache_path)

def test_sampling_warning():
    """
    Testing the sampling with warning.
    """
    smote_variants.config.suppress_internal_warnings(False)

    folding = Folding(DATASET_WARNING,
                    cache_path=cache_path,
                    validator_params={'n_splits': 2, 'n_repeats': 1},
                    reset=True)
    oversampler = 'SMOTE'
    oversampler_params = {}

    for fold in folding.fold():
        sjob = SamplingJob(fold,
                        oversampler,
                        oversampler_params)

        result = sjob.do_oversampling()

        assert len(result['warning']) > 3

    smote_variants.config.suppress_internal_warnings(True)

def test_sampling_error():
    """
    Testing the sampling with error.
    """
    folding = Folding(DATASET,
                    cache_path=cache_path,
                    reset=True)

    for fold in folding.fold():
        oversampler = 'NoSMOTE'
        oversampler_params = {'raise_value_error': True}

        sjob = SamplingJob(fold,
                        oversampler,
                        oversampler_params)

        result = sjob.do_oversampling()

        assert len(result['error']) > 3

        oversampler = 'NoSMOTE'
        oversampler_params = {'raise_runtime_error': True}

        sjob = SamplingJob(fold,
                        oversampler,
                        oversampler_params)

        result = sjob.do_oversampling()

        assert len(result['error']) > 3

def test_sampling_cached_folding():
    """
    Testing the sampling with cached folding.
    """
    shutil.rmtree(cache_path, ignore_errors=True)

    folding = Folding(DATASET,
                    cache_path=cache_path,
                    reset=True)
    oversampler = 'SMOTE'
    oversampler_params = {}

    for fold in folding.folding_files():
        sjob = SamplingJob(fold,
                        oversampler,
                        oversampler_params,
                        cache_path=cache_path)

        result = sjob.do_oversampling()
        assert os.path.exists(result)

    for fold in folding.folding_files():
        sjob = SamplingJob(fold,
                        oversampler,
                        oversampler_params,
                        cache_path=cache_path)

        result = sjob.do_oversampling()
        assert os.path.exists(result)

    shutil.rmtree(cache_path)

def test_sampling_no_caching():
    """
    Testing the sampling without caching.
    """
    folding = Folding(DATASET,
                    reset=True)
    oversampler = 'SMOTE'
    oversampler_params = {}

    for fold in folding.fold():
        sjob = SamplingJob(fold,
                        oversampler,
                        oversampler_params)

        result = sjob.do_oversampling()
        assert isinstance(result, dict)

def test_get_params():
    """
    Testing the get_params function.
    """
    sampling = SamplingJob({},
                    cache_path=None,
                    oversampler = 'SMOTE',
                    oversampler_params = {})
    assert len(sampling.get_params()) > 0

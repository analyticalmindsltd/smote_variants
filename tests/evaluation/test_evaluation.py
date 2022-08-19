"""
Testing the evaluation.
"""

import os
import shutil

from sklearn import datasets

from smote_variants.evaluation import Folding, SamplingJob, EvaluationJob
import smote_variants

DATASET = datasets.load_breast_cancer()
DATASET['name'] = 'breast_cancer'

OVERSAMPLER = 'SMOTE'
OVERSAMPLER_PARAMS = {}

CLASSIFIERS = [('sklearn.tree', 'DecisionTreeClassifier', {})]

CACHE_PATH = os.path.join('.', 'test_path')

def test_evaluation():
    """
    Testing the evaluation.
    """
    shutil.rmtree(CACHE_PATH, ignore_errors=True)

    folding = Folding(DATASET,
                    cache_path=CACHE_PATH,
                    reset=True)

    for fold in folding.fold():
        sjob = SamplingJob(fold,
                        OVERSAMPLER,
                        OVERSAMPLER_PARAMS,
                        cache_path=CACHE_PATH)

        result = sjob.do_oversampling()

        ejob = EvaluationJob(result,
                            CLASSIFIERS,
                            cache_path=CACHE_PATH)

        result = ejob.do_evaluation()

        assert os.path.exists(result[0])

    for fold in folding.fold():
        sjob = SamplingJob(fold,
                        OVERSAMPLER,
                        OVERSAMPLER_PARAMS,
                        cache_path=CACHE_PATH)

        result = sjob.do_oversampling()

        ejob = EvaluationJob(result,
                            CLASSIFIERS,
                            cache_path=CACHE_PATH)

        result = ejob.do_evaluation()

        assert os.path.exists(result[0])

    shutil.rmtree(CACHE_PATH)

def test_evaluation_warning():
    """
    Testing the evaluation warning.
    """
    folding = Folding(DATASET,
                    reset=True)

    smote_variants.config.suppress_internal_warnings(False)

    for fold in folding.fold():
        sjob = SamplingJob(fold,
                        OVERSAMPLER,
                        OVERSAMPLER_PARAMS)

        result = sjob.do_oversampling()

        ejob = EvaluationJob(result,
                            [('smote_variants.classifiers',
                             'ErrorWarningClassifier',
                             {'raise_warning': True})])

        result = ejob.do_evaluation()

        assert len(result[0]['warning']) > 3

        smote_variants.config.suppress_internal_warnings(True)

def test_evaluation_error():
    """
    Testing the evaluation error.
    """
    folding = Folding(DATASET,
                    reset=True)

    for fold in folding.fold():
        sjob = SamplingJob(fold,
                        OVERSAMPLER,
                        OVERSAMPLER_PARAMS)

        result = sjob.do_oversampling()

        ejob = EvaluationJob(result,
                            [('smote_variants.classifiers',
                             'ErrorWarningClassifier',
                             {'raise_value_error': True})])

        result_eval = ejob.do_evaluation()

        assert len(result_eval[0]['error']) > 3

        ejob = EvaluationJob(result,
                            [('smote_variants.classifiers',
                             'ErrorWarningClassifier',
                             {'raise_runtime_error': True})])

        result_eval = ejob.do_evaluation()

        assert len(result_eval[0]['error']) > 3

def test_sampling_cached_folding():
    """
    Testing the sampling of a cached folding.
    """
    shutil.rmtree(CACHE_PATH, ignore_errors=True)

    folding = Folding(DATASET,
                    cache_path=CACHE_PATH,
                    reset=True)
    oversampler = 'SMOTE'
    oversampler_params = {}

    for fold in folding.folding_files():
        sjob = SamplingJob(fold,
                        oversampler,
                        oversampler_params,
                        cache_path=CACHE_PATH)

        result = sjob.do_oversampling()

        ejob = EvaluationJob(result,
                            CLASSIFIERS,
                            cache_path=CACHE_PATH)

        result = ejob.do_evaluation()

        assert os.path.exists(result[0])

    for fold in folding.folding_files():
        sjob = SamplingJob(fold,
                        oversampler,
                        oversampler_params,
                        cache_path=CACHE_PATH)

        result = sjob.do_oversampling()

        ejob = EvaluationJob(result,
                            CLASSIFIERS,
                            cache_path=CACHE_PATH)

        result = ejob.do_evaluation()

        assert os.path.exists(result[0])

    shutil.rmtree(CACHE_PATH)

def test_sampling_no_caching():
    """
    Testing sampling with no caching.
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

        ejob = EvaluationJob(result,
                            CLASSIFIERS)

        result = ejob.do_evaluation()

        assert isinstance(result[0], dict)

def test_get_params():
    """
    Test the get_params function.
    """
    ejob = EvaluationJob({},
                        CLASSIFIERS,
                        cache_path=CACHE_PATH)

    assert len(ejob.get_params()) > 0

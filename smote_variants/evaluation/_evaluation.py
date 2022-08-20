"""
This module implements the parallelizable Evaluation Job
"""

import os
import hashlib
import time
import warnings

import numpy as np

from ..base import instantiate_obj, load_dict, dump_dict, check_if_damaged
from ..base import calculate_all_scores

__all__ = ['EvaluationJob']

class EvaluationJob:
    """
    An evaluation job to be executed in parallel
    """

    def __init__(self,
                oversampling,
                classifiers,
                *,
                cache_path=None,
                serialization="json",
                reset=False):
        """
        Constructor of the object

        oversampling (str/dict): an oversampling or a filename containing an
                                    oversampling
        classifiers (list(tuple)): classifier specification
        cache_path (str): the cache path
        serialization (str): 'json'/'pickle'
        reset (str): whether to recompute the oversampling
        """
        self.oversampling = oversampling
        self.classifiers = classifiers
        self.cache_path = cache_path
        self.serialization = serialization
        self.reset = reset

        if self.cache_path is not None:
            os.makedirs(self.cache_path, exist_ok=True)

    def get_params(self, deep=False):
        """
        Returns the parameters of the object.

        Args:
            deep (bool): deep parameters

        Returns:
            dict: the dictionary of parameters
        """
        _ = deep # pylint workaround
        return {'oversampling': self.oversampling,
                'classifiers': self.classifiers,
                'cache_path': self.cache_path,
                'serialization': self.serialization,
                'reset': self.reset}

    def target_filename_damaged(self, classifier, oversampling):
        """
        Determine the target filename and whether it is damaged

        Args:
            classifier (tuple): a classifier specification
            oversampling (dict): an oversampling

        Returns:
            str, bool: the target filename and whether it is damaged
        """
        target_filename = None

        label0 = str(classifier).encode('utf-8')
        label1 = str((oversampling['oversampler'])).encode('utf-8')

        hashcode = str(hashlib.md5(label0 + label1).hexdigest())

        damaged = False
        if self.cache_path is not None:
            repeat_idx = oversampling['fold_descriptor']['repeat_idx']
            filename = f"evaluation_{repeat_idx:04}"\
                        f"_{oversampling['fold_descriptor']['fold_idx']:04}"\
                        f"_{hashcode}.{self.serialization}"
            target_filename = os.path.join(self.cache_path, filename)

            damaged = check_if_damaged(target_filename, self.serialization)

        return target_filename, damaged

    def do_evaluation(self):
        """
        do the evaluation

        Returns:
            dict/str: an evaluation dictionary or a filename
        """
        oversampling = self.oversampling

        all_results = []

        if isinstance(oversampling, str):
            oversampling = load_dict(oversampling, oversampling.split('.')[-1],
                                ['X_train', 'X_test', 'y_train', 'y_test'])

        for classifier in self.classifiers:
            target_filename, damaged = self.target_filename_damaged(classifier,
                                                                    oversampling)

            if self.cache_path is not None and not damaged and not self.reset:
                all_results.append(target_filename)
                continue

            classifier_obj = instantiate_obj(classifier)

            error = None
            warning_list = None

            begin_timestamp = time.time()

            with warnings.catch_warnings(record=True) as warning:
                try:
                    y_pred = classifier_obj.fit(oversampling['X_train'],
                                                oversampling['y_train'])\
                                            .predict_proba(oversampling['X_test'])
                except ValueError as value_error:
                    error = str(value_error)
                    y_pred = np.zeros(shape=(oversampling['y_test'].shape[0], 2))
                except RuntimeError as runtime_error:
                    error = str(runtime_error)
                    y_pred = np.zeros(shape=(oversampling['y_test'].shape[0], 2))

                warning_list = [(str(warn.category), warn.message) for warn in warning]

            evaluation = {'fold_descriptor': oversampling['fold_descriptor'],
                    'oversampler': oversampling['oversampler'],
                    'runtime': time.time() - begin_timestamp,
                    'y_pred': y_pred[:, 1],
                    'y_test': oversampling['y_test'],
                    'scores': calculate_all_scores(oversampling['y_test'],
                                                y_pred[:, 1]),
                    'classifier': classifier[1],
                    'classifier_params': classifier[2],
                    'classifier_module': classifier[0],
                    'error': error,
                    'warning': str(warning_list)}

            if self.cache_path is not None:
                dump_dict(evaluation, target_filename, self.serialization,
                        ['y_pred', 'y_test'])
                all_results.append(target_filename)
                continue

            all_results.append(evaluation)

        return all_results

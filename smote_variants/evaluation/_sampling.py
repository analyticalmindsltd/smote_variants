"""
This module implements the parallelizable Sampling Job
"""

import os
import hashlib
import time
import warnings

from ..base import instantiate_obj, load_dict, dump_dict, check_if_damaged

__all__= ['SamplingJob']

class SamplingJob:
    """
    An oversampling job to be executed in parallel
    """
    def __init__(self,
                 folding,
                 oversampler,
                 oversampler_params,
                 *,
                 scaler=('sklearn.preprocessing', 'StandardScaler', {}),
                 cache_path=None,
                 serialization="json",
                 reset=False):
        """
        Constructor of the object

        folding (str/dict): a folding or a filename containing a folding
        oversampler (str): name of the oversampler class
        oversampler_params (dict): the dictionary of oversampler params
        scaler (obj): a scaler object
        cache_path (str): the cache path
        serialization (str): 'json'/'pickle'
        reset (str): whether to recompute the oversampling
        """
        self.folding = folding
        self.oversampler = oversampler
        self.oversampler_params = oversampler_params
        self.scaler = scaler
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
        return {'folding': self.folding,
                'oversampler': self.oversampler,
                'oversampler_params': self.oversampler_params,
                'cache_path': self.cache_path,
                'serialization': self.serialization,
                'reset': self.reset}

    def target_filename_damaged(self, folding):
        """
        Determine the target filename and whether it is damaged.

        Args:
            folding (obj): a folding

        Returns:
            str, bool: the target filename and the damaged flag
        """
        target_filename = None

        label = str((self.oversampler, self.oversampler_params)).encode('utf-8')
        hashcode = str(hashlib.md5(label).hexdigest())

        damaged = False
        if self.cache_path is not None:
            filename = f"oversampling_{folding['fold_descriptor']['repeat_idx']:04}"\
                        f"_{folding['fold_descriptor']['fold_idx']:04}"\
                        f"_{hashcode}.{self.serialization}"
            target_filename = os.path.join(self.cache_path, filename)

            damaged = check_if_damaged(target_filename, self.serialization)

        return target_filename, damaged

    def do_oversampling(self):
        """
        do the oversampling

        Returns:
            dict/str: an oversampling dictionary or a filename
        """
        folding = self.folding

        if isinstance(folding, str):
            folding = load_dict(folding, folding.split('.')[-1],
                                ['X_train', 'y_train', 'X_test', 'y_test'])

        target_filename, damaged = self.target_filename_damaged(folding)

        if self.cache_path is not None and not damaged and not self.reset:
            return target_filename

        oversampler_obj = instantiate_obj(('smote_variants',
                                            self.oversampler,
                                            self.oversampler_params))

        X_train = folding['X_train']
        X_test = folding['X_test']

        if self.scaler is not None:
            scaler = instantiate_obj(self.scaler)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        oversampling = {'fold_descriptor': folding['fold_descriptor'],
                        'y_test': folding['y_test'],
                        'oversampler': oversampler_obj.get_params()}

        begin_timestamp = time.time()
        with warnings.catch_warnings(record=True) as warning:
            try:
                X_samp, y_samp = oversampler_obj.sample(folding['X_train'],
                                                        folding['y_train'])
                X_test = oversampler_obj.preprocessing_transform(folding['X_test'])
                oversampling = {**oversampling,
                                'X_train': X_samp,
                                'y_train': y_samp,
                                'X_test': X_test,
                                'runtime': time.time() - begin_timestamp,
                                'error': None,
                                'warning': None}
            except ValueError as value_error:
                oversampling = {**oversampling,
                                'X_train': folding['X_train'],
                                'y_train': folding['y_train'],
                                'X_test': folding['X_test'],
                                'runtime': time.time() - begin_timestamp,
                                'error': str(value_error)}
            except RuntimeError as runtime_error:
                oversampling = {**oversampling,
                                'X_train': folding['X_train'],
                                'y_train': folding['y_train'],
                                'X_test': folding['X_test'],
                                'runtime': time.time() - begin_timestamp,
                                'error': str(runtime_error)}

            oversampling = {**oversampling,
                            'warning': str([(str(warn.category), warn.message)
                                                            for warn in warning])}

        if self.cache_path is not None:
            dump_dict(oversampling, target_filename, self.serialization,
                        ['X_train', 'y_train', 'X_test', 'y_test'])
            return target_filename

        return oversampling

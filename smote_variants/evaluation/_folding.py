"""
This module implements the folding abstraction.
"""

import os
import glob
import shutil

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from ..base import dump_dict

__all__= ['Folding']

class Folding():
    """
    Cache-able folding of dataset for cross-validation
    """

    def __init__(self,
                dataset,
                *,
                serialization='json',
                cache_path=None,
                validator_params=None,
                reset=False):
        """
        Constructor of Folding object

        Args:
            dataset (dict): dataset dictionary with keys 'data', 'target'
                            and 'DESCR'
            validator (obj): cross-validator object
            serialization (str): 'json'/'pickle'
            cache_path (str): path to cache directory
            validator_params (str): validator parameters
        """
        self.dataset = dataset
        if validator_params is None:
            self.validator_params = {'n_repeats': 2,
                                        'n_splits': 5,
                                        'random_state': 5}
        else:
            self.validator_params = validator_params
        self.serialization = serialization
        self.cache_path = cache_path
        self.reset = reset

        labels, counts = np.unique(self.dataset['target'],
                                    return_counts=True)

        min_label = int(np.max(labels[counts == np.min(counts)]))
        maj_label = int(labels[labels != min_label][0])

        n_min = int(counts[min_label])
        n_maj = int(counts[maj_label])
        imb_ratio = float(counts[maj_label]/counts[min_label])

        self.properties = {'db_n': len(dataset['data']),
                            'db_n_attr': len(dataset['data'][0]),
                            'imbalance_ratio': imb_ratio,
                            'name': dataset['name'],
                            'label_stats': {'min_label': n_min,
                                            'maj_label': n_maj}}

        if self.cache_path is not None:
            self.folding_dir_path = os.path.join(self.cache_path, dataset['name'])

            # creating the folding directory
            os.makedirs(self.folding_dir_path, exist_ok=True)

            if reset:
                self._do_reset()

    def _do_reset(self):
        """
        Remove all foldings
        """
        shutil.rmtree(self.folding_dir_path, ignore_errors=True)
        os.makedirs(self.folding_dir_path, exist_ok=True)

    def folding_files(self):
        """
        Get the list of filenames

        Returns:
            list: the list of fold filenames
        """
        if self.serialization == 'json':
            path = os.path.join(self.folding_dir_path, 'fold_*.json')
        elif self.serialization == 'pickle':
            path = os.path.join(self.folding_dir_path, 'fold_*.pickle')

        if len(glob.glob(path)) == 0:
            self.cache_foldings()

        return sorted(glob.glob(path))

    def fold(self):
        """
        Does the folding or reads it from file if already available

        Returns:
            list(tuple): list of tuples of X_train, y_train, X_test, y_test
                            objects
        """
        validator = RepeatedStratifiedKFold(**self.validator_params)

        data = self.dataset['data']
        target = self.dataset['target']

        properties = self.properties
        n_splits = self.validator_params['n_splits']
        properties['n_repeats'] = self.validator_params['n_repeats']
        properties['n_splits'] = n_splits

        for idx, (train, test) in enumerate(validator.split(data,
                                                            target,
                                                            target)):
            properties['split_idx'] = idx
            properties['repeat_idx'] = idx // n_splits
            properties['fold_idx'] = idx - (idx // n_splits)*n_splits

            folding = {'X_train': data[train],
                        'y_train': target[train],
                        'X_test': data[test],
                        'y_test': target[test],
                        'fold_descriptor': properties}

            yield folding

    def cache_foldings(self):
        """
        Cache the foldings
        """
        for folding in self.fold():
            repeat_idx = folding['fold_descriptor']['repeat_idx']
            fold_idx = folding['fold_descriptor']['fold_idx']
            filename = f"fold_{repeat_idx:04}_{fold_idx:04}"\
                        f".{self.serialization}"
            full_name = os.path.join(self.folding_dir_path, filename)

            dump_dict(folding, full_name, self.serialization,
                        ['X_train', 'X_test', 'y_train', 'y_test'])

    def get_params(self, deep=False):
        """
        Return the parameters of the object

        Args:
            deep (bool): deep parameters

        Returns:
            dict: the parameters of the object
        """
        _ = deep # pylint fix

        return {'dataset': self.dataset,
                'serialization': self.serialization,
                'cache_path': self.cache_path,
                'validator_params': self.validator_params,
                'reset': self.reset}

    def descriptor(self):
        """
        Returns the descriptor of the object

        Returns:
            str: the descriptor of the object
        """
        return str(self.get_params())

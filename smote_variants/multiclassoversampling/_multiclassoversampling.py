"""
This moduel imports the multiclass oversampling capabilities.
"""
import inspect

import numpy as np

from ..base import StatisticsMixin, instantiate_obj
from .._logger import logger

_logger = logger

# exported names
__all__ = ['MulticlassOversampling']

def proportion_1_vs_many(stats,
                        labels,
                        maj_label,
                        class_idx):
    """
    Determines the proportion to sample for the 1_vs_many strategy.

    Args:
        stats (dict): class statistics
        labels (list): class labels
        maj_label (int): the majority label
        class_idx (int): the class index to determine the proportion for

    Returns:
        float: the proportion
    """
    to_gen = stats[maj_label] - stats[labels[class_idx]]
    to_gen_to_all = stats[maj_label] - stats[labels[class_idx]]

    return to_gen/to_gen_to_all

def proportion_1_vs_many_succ(stats,
                                labels,
                                maj_label,
                                class_idx):
    """
    Determines the proportion to sample for the 1_vs_many_successive strategy.

    Args:
        stats (dict): class statistics
        labels (list): class labels
        maj_label (int): the majority label
        class_idx (int): the class index to determine the proportion for

    Returns:
        float: the proportion
    """
    n_majority = stats[maj_label]
    n_class_i = stats[labels[class_idx]]
    num_to_generate = n_majority - n_class_i

    num_to_gen_to_all = class_idx * n_majority - n_class_i

    return num_to_generate/num_to_gen_to_all

def has_proportion_parameter(oversampler):
    """
    Checks if an oversampler object has proportion parameters

    Args:
        oversampler: oversampler object

    Returns:
        bool: True if the object has 'proportion' parameter
    """
    return 'proportion' in \
            list(inspect.signature(oversampler.__class__).parameters.keys())

class MulticlassOversampling(StatisticsMixin):
    """
    Carries out multiclass oversampling

    Example::

        import smote_variants as sv
        import sklearn.datasets as datasets

        dataset= datasets.load_wine()

        oversampler= sv.MulticlassOversampling(sv.distance_SMOTE())

        X_samp, y_samp= oversampler.sample(dataset['data'], dataset['target'])
    """

    def __init__(self,
                 oversampler='SMOTE',
                 oversampler_params=None,
                 strategy="eq_1_vs_many_successive"):
        """
        Constructor of the multiclass oversampling object

        Args:
            oversampler (obj): an oversampling object
            strategy (str/obj): a multiclass oversampling strategy, currently
                                'eq_1_vs_many_successive' or
                                'equalize_1_vs_many'
        """
        self.oversampler = oversampler
        if oversampler_params is None:
            oversampler_params = {}
        self.oversampler_params = oversampler_params
        self.strategy = strategy

        if not has_proportion_parameter(instantiate_obj(('smote_variants',
                                                    self.oversampler,
                                                    self.oversampler_params))):
            raise ValueError((f"Multiclass oversampling strategy {self.strategy}"
                       " cannot be used with oversampling techniques without"
                       " proportion parameter"))

        if self.strategy not in ['eq_1_vs_many_successive',
                                        'equalize_1_vs_many']:
            message = "Multiclass oversampling startegy %s not implemented."
            message = message % self.strategy
            raise ValueError(message)

    def sample_equalize_1_vs_many(self, X, y):
        """
        Does the sample generation by oversampling each minority class to the
        cardinality of the majority class using all original samples in each
        run.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        _logger.info("%s: %s", self.__class__.__name__,
                f"Running multiclass oversampling with strategy {self.strategy}")

        # extract class label statistics
        self.class_label_statistics(y)

        # sort labels by number of samples
        labels = self.class_stats.keys()
        labels = sorted(labels, key=lambda x: -self.class_stats[x])

        # dict to store the results
        results = {}
        results[labels[0]] = X[y == labels[0]]

        # running oversampling for all minority classes against all oversampled
        # classes
        for class_idx in range(1, len(labels)):
            _logger.info("%s: %s", self.__class__.__name__,
                    f"Sampling minority class with label: {labels[class_idx]}")

            # prepare data to pass to oversampling
            X_train = np.vstack([X[y != labels[class_idx]],
                                    X[y == labels[class_idx]]])
            y_train = np.hstack([np.repeat(0, np.sum(y != labels[class_idx])),
                                 np.repeat(1, np.sum(y == labels[class_idx]))])

            # prepare parameters by properly setting the proportion value
            params = self.oversampler_params.copy()
            params['proportion'] = proportion_1_vs_many(self.class_stats,
                                                        labels,
                                                        labels[0],
                                                        class_idx)

            # instantiating new oversampling object with the proper proportion
            # parameter
            oversampler = instantiate_obj(('smote_variants',
                                            self.oversampler,
                                            params))

            # executing the sampling
            X_samp, y_samp = oversampler.sample(X_train, y_train)

            # registering the newly oversampled minority class in the output
            # set
            results[labels[class_idx]] = X_samp[len(X_train):][y_samp[len(X_train):] == 1]

        # constructing the output set
        X_final = results[labels[1]]
        y_final = np.repeat(labels[1], len(results[labels[1]]))

        for class_idx in range(2, len(labels)):
            X_final = np.vstack([X_final, results[labels[class_idx]]])
            y_final = np.hstack([y_final, np.repeat(labels[class_idx],
                                            len(results[labels[class_idx]]))])

        return np.vstack([X, X_final]), np.hstack([y, y_final])

    def sample_equalize_1_vs_many_successive(self, X, y):
        """
        Does the sample generation by oversampling each minority class
        successively to the cardinality of the majority class,
        incorporating the results of previous oversamplings.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        _logger.info("%s: %s", self.__class__.__name__,
            f"Running multiclass oversampling with strategy {self.strategy}")

        # extract class label statistics
        self.class_label_statistics(y)

        # sort labels by number of samples
        labels = self.class_stats.keys()
        labels = sorted(labels, key=lambda x: -self.class_stats[x])

        # determining the majority class data
        X_maj = X[y == labels[0]]

        # dict to store the results
        results = {labels[0]: X_maj}

        # running oversampling for all minority classes against all
        # oversampled classes
        for class_idx in range(1, len(labels)):
            _logger.info("%s: %s", self.__class__.__name__,
                f"Sampling minority class with label: {labels[class_idx]}")

            # prepare data to pass to oversampling
            X_train = np.vstack([X_maj, X[y == labels[class_idx]]])
            y_train = np.hstack([np.repeat(0, len(X_maj)),
                                 np.repeat(1, np.sum(y == labels[class_idx]))])

            # prepare parameters by properly setting the proportion value
            params = self.oversampler_params.copy()
            params['proportion'] = proportion_1_vs_many_succ(self.class_stats,
                                                                labels,
                                                                labels[0],
                                                                class_idx)

            # executing the sampling
            X_samp, y_samp = instantiate_obj(('smote_variants',
                                                self.oversampler,
                                                params)).sample(X_train, y_train)

            # adding the newly oversampled minority class to the majority data
            X_maj = np.vstack([X_maj, X_samp[y_samp == 1]])

            # registaring the newly oversampled minority class in the output
            # set
            result_mask = y_samp[len(X_train):] == 1
            results[labels[class_idx]] = X_samp[len(X_train):][result_mask]

        # constructing the output set
        X_final = results[labels[1]]
        y_final = np.repeat(labels[1], len(results[labels[1]]))

        for class_idx in range(2, len(labels)):
            X_final = np.vstack([X_final, results[labels[class_idx]]])
            y_final = np.hstack([y_final, np.repeat(labels[class_idx],
                                            len(results[labels[class_idx]]))])

        return np.vstack([X, X_final]), np.hstack([y, y_final])

    def sample(self, X, y):
        """
        Does the sample generation according to the oversampling strategy.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        if self.strategy == "eq_1_vs_many_successive":
            return self.sample_equalize_1_vs_many_successive(X, y)

        return self.sample_equalize_1_vs_many(X, y)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the multiclass oversampling object
        """
        _ = deep
        return {'oversampler': self.oversampler,
                'oversampler_params': self.oversampler_params,
                'strategy': self.strategy}

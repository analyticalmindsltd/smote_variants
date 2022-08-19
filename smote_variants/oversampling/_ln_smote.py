"""
This module implements the LN_SMOTE method.
"""

import numpy as np

from ..base import coalesce
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSampling
from .._logger import logger
_logger= logger

__all__= ['LN_SMOTE']

class LN_SMOTE(OverSampling): # pylint: disable=invalid-name
    """
    References:
        * BibTex::

            @INPROCEEDINGS{ln_smote,
                            author={Maciejewski, T. and Stefanowski, J.},
                            booktitle={2011 IEEE Symposium on Computational
                                        Intelligence and Data Mining (CIDM)},
                            title={Local neighbourhood extension of SMOTE for
                                        mining imbalanced data},
                            year={2011},
                            volume={},
                            number={},
                            pages={104-111},
                            keywords={Bayes methods;data mining;pattern
                                        classification;local neighbourhood
                                        extension;imbalanced data mining;
                                        focused resampling technique;SMOTE
                                        over-sampling method;naive Bayes
                                        classifiers;Noise measurement;Noise;
                                        Decision trees;Breast cancer;
                                        Sensitivity;Data mining;Training},
                            doi={10.1109/CIDM.2011.5949434},
                            ISSN={},
                            month={April}}
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_componentwise,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): parameter of the NearestNeighbors component
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0.0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
        self.n_jobs = n_jobs

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def safe_level(self, p_idx, *, y, indices, n_idx=None):
        """
        computing the safe level of samples

        Args:
            p_idx (int): index of positive sample
            y (np.array): the target labels
            indices (np.array): the neighborhood structure
            n_idx (int): index of other sample

        Returns:
            int: safe level
        """
        if n_idx is None:
            # implementation for 1 sample only
            return np.sum(y[indices[p_idx][1:-1]] == self.min_label)

        # implementation for 2 samples
        if ((not y[n_idx] != self.maj_label)
                and p_idx in indices[n_idx][1:-1]):
            # -1 because p_idx will be replaced
            n_positives = np.sum(y[indices[n_idx][1:-1]] == self.min_label) - 1
            if y[indices[n_idx][-1]] == self.min_label:
                # this is the effect of replacing p_idx by the next
                # (k+1)th neighbor
                n_positives = n_positives + 1
            return n_positives
        return np.sum(y[indices[n_idx][1:-1]] == self.min_label)

    def random_gap(self, *, slp, sln, n_label, n_neighbors, n_dim):
        """
        determining random gap

        Args:
            slp (int): safe level of p
            sln (int): safe level of n
            n_label (int): label of n
            n_neighbors (int): number of neighbors
            n_dim (int): number of dimensions

        Returns:
            float: gap
        """
        delta = 0.0

        if sln == 0 and slp > 0:
            return np.repeat(delta, n_dim)

        sl_ratio = slp/sln

        if sl_ratio == 1:
            delta = self.random_state.random_sample(size=n_dim)
        elif sl_ratio > 1:
            delta = self.random_state.random_sample(size=n_dim)/sl_ratio
        else:
            delta = 1.0 - self.random_state.random_sample(size=n_dim)*sl_ratio

        if not n_label == self.min_label:
            delta = delta * sln / n_neighbors

        return delta

    def calculate_slp_sln(self, minority_indices, indices, y):
        """
        Calculate the safe levels

        Args:
            minority_indices (np.array): the minority indices
            indices (np.array): the neighborhood structure
            y (np.array): the target labels

        Returns:
            dict: safe levels of positives and their
                                neighbors
        """
        slpsln = {}

        for p_idx in minority_indices:
            slp = self.safe_level(p_idx, y=y, indices=indices)
            for n_idx in indices[p_idx][1:-1]:
                sln = self.safe_level(p_idx, n_idx=n_idx, y=y,
                                            indices=indices)
                slpsln[(p_idx, n_idx)] = (slp, sln)

        return slpsln

    def generate_samples(self, *, slpsln, n_to_sample, X, y, n_neighbors):
        """
        Generate samples

        Args:
            slpsln (dict): the safe level values
            n_to_sample (int): number of samples to generate
            X (np.array): all feature vectors
            y (np.array): all target labels
            n_neighbors (int): the number of neighbors to use

        Returns:
            np.array: the generated samples
        """

        if len(slpsln) == 0:
            return np.zeros(shape=(0, X.shape[1]))

        samples = []
        # generating samples
        while len(samples) < n_to_sample:
            idx = self.random_state.choice(np.arange(len(slpsln)))
            p_idx, n_idx = list(slpsln.keys())[idx]

            slp, sln = slpsln[(p_idx, n_idx)]

            delta = self.random_gap(slp=slp,
                                    sln=sln,
                                    n_label=y[n_idx],
                                    n_neighbors=n_neighbors,
                                    n_dim=X.shape[1])

            samples.append(X[p_idx] + (X[n_idx] - X[p_idx]) * delta)

        return np.vstack(samples)

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        # number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        n_neighbors = np.min([len(X) - 2, self.n_neighbors])

        if n_neighbors < 2:
            return self.return_copies(X, y, f"Too few samples {len(X)}")

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # nearest neighbors of each instance to each instance in the dataset
        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors + 2,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X)
        indices = nnmt.kneighbors(X, return_distance=False)

        minority_indices = np.where(y == self.min_label)[0]

        # calculate safe levels
        slpsln = self.calculate_slp_sln(minority_indices, indices, y)

        slpsln = {key: item for key, item in slpsln.items()\
                                            if item[0] > 0 and item[1] > 0}

        samples = self.generate_samples(slpsln=slpsln,
                                        n_to_sample=n_to_sample,
                                        X=X,
                                        y=y,
                                        n_neighbors=n_neighbors)

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}

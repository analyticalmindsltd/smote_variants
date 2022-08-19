"""
This module implements the Selected_SMOTE method.
"""

import numpy as np

from ..base import coalesce
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSampling
from .._logger import logger
_logger= logger

__all__= ['Selected_SMOTE']

class Selected_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @article{smote_out_smote_cosine_selected_smote,
                    title={SMOTE-Out, SMOTE-Cosine, and Selected-SMOTE: An
                                enhancement strategy to handle imbalance in
                                data level},
                    author={Fajri Koto},
                    journal={2014 International Conference on Advanced
                                Computer Science and Information System},
                    year={2014},
                    pages={280-284}
                    }

    Notes:
        * Significant attribute selection was not described in the paper,
            therefore we have implemented something meaningful.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_componentwise,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 perc_sign_attr=0.5,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            strategy (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): parameter of the NearestNeighbors component
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            perc_sign_attr (float): [0,1] - percentage of significant
                                            attributes
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state)
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_in_range(perc_sign_attr, 'perc_sign_attr', [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
        self.perc_sign_attr = perc_sign_attr
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
                                  'n_neighbors': [3, 5, 7],
                                  'perc_sign_attr': [0.3, 0.5, 0.8]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def significant_attributes(self, X_min, X_maj):
        """
        Significant attribute selection.

        Args:
            X_min (np.array): the minority samples
            X_maj (np.array): the majority samples

        Returns:
            np.array: the significant attribute mask
        """
        # significant attribute selection was not described in the paper
        # I have implemented significant attribute selection by checking
        # the overlap between ranges of minority and majority class attributes
        # the attributes with bigger overlap respecting their ranges
        # are considered more significant
        min_ranges_a = np.min(X_min, axis=0)
        min_ranges_b = np.max(X_min, axis=0)
        maj_ranges_a = np.min(X_maj, axis=0)
        maj_ranges_b = np.max(X_maj, axis=0)

        # end points of overlaps
        max_a = np.max(np.vstack([min_ranges_a, maj_ranges_a]), axis=0)
        min_b = np.min(np.vstack([min_ranges_b, maj_ranges_b]), axis=0)

        # size of overlap
        overlap = min_b - max_a

        # replacing negative values (no overlap) by zero
        overlap = np.where(overlap < 0, 0, overlap)
        # percentage of overlap compared to the ranges of attributes in the
        # minority set
        percentages = min_ranges_b - min_ranges_a
        percentages[percentages != 0] = overlap[percentages != 0] \
                                            / percentages[percentages != 0]

        # number of significant attributes to determine
        tmp = int(np.rint(self.perc_sign_attr*len(percentages)))
        num_sign_attr = np.min([1, tmp])

        sign_attr = (percentages >= sorted(percentages)[-num_sign_attr])
        sign_attr = sign_attr.astype(int)

        return sign_attr

    def generate_samples(self,
                            X_min,
                            base_indices,
                            neigh_indices,
                            significant_attr):
        """
        Generate the samples.

        Args:
            X_min (np.array): minority samples
            base_indices (np.array): indices of base vectors
            neigh_indices (np.array): indices of neighbor vectors
            significant_attr (np.array): significant attribute mask

        Returns:
            np.array: the generated samples
        """
        base_vectors = X_min[base_indices]
        neigh_vectors = X_min[neigh_indices]
        random = self.random_state.random_sample(size=base_vectors.shape)
        samples = base_vectors + \
                    (neigh_vectors - base_vectors) * random * significant_attr
        return samples

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        n_to_sample = self.det_n_to_sample(self.proportion)
        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        X_min = X[y == self.min_label]

        n_neighbors = np.min([len(X_min), self.n_neighbors + 1])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= \
            self.metric_tensor_from_nn_params(nn_params, X, y)

        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                        n_jobs=self.n_jobs,
                                                        **(nn_params))
        nnmt.fit(X_min)

        indices = nnmt.kneighbors(X_min, return_distance=False)

        significant_attr = self.significant_attributes(X_min,
                                                        X[y == self.maj_label])

        base_indices = self.random_state.choice(np.arange(X_min.shape[0]),
                                                n_to_sample)
        neigh_indices = self.random_state.choice(np.arange(1, n_neighbors),
                                                    n_to_sample)

        samples = self.generate_samples(X_min=X_min,
                                        base_indices=base_indices,
                                        neigh_indices=indices[base_indices,
                                                            neigh_indices],
                                        significant_attr=significant_attr)

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
                'perc_sign_attr': self.perc_sign_attr,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}

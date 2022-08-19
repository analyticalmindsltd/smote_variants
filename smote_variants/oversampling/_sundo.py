"""
This module implements the SUNDO method.
"""

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from ..base import coalesce
from ..base import (NearestNeighborsWithMetricTensor,
                                pairwise_distances_mahalanobis)
from ..base import OverSampling

from .._logger import logger
_logger= logger

__all__= ['SUNDO']

class SUNDO(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{sundo,
                            author={Cateni, S. and Colla, V. and Vannucci, M.},
                            booktitle={2011 11th International Conference on
                                        Intelligent Systems Design and
                                        Applications},
                            title={Novel resampling method for the
                                    classification of imbalanced datasets for
                                    industrial and other real-world problems},
                            year={2011},
                            volume={},
                            number={},
                            pages={402-407},
                            keywords={decision trees;pattern classification;
                                        sampling methods;support vector
                                        machines;resampling method;imbalanced
                                        dataset classification;industrial
                                        problem;real world problem;
                                        oversampling technique;undersampling
                                        technique;support vector machine;
                                        decision tree;binary classification;
                                        synthetic dataset;public dataset;
                                        industrial dataset;Support vector
                                        machines;Training;Accuracy;Databases;
                                        Intelligent systems;Breast cancer;
                                        Decision trees;oversampling;
                                        undersampling;imbalanced dataset},
                            doi={10.1109/ISDA.2011.6121689},
                            ISSN={2164-7151},
                            month={Nov}}
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_application,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 *,
                 nn_params=None,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
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

        self.check_n_jobs(n_jobs, 'n_jobs')

        self.nn_params = coalesce(nn_params, {})
        self.n_jobs = n_jobs

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        _ = raw # pylint hack
        return [{}]

    def generate_samples(self, X_min, X_maj, nn_params, N): # pylint: disable=invalid-name
        """
        Generate samples.

        Args:
            X_min (np.array): the minority vectors
            X_maj (np.array): the majority vectors
            nn_params (dict): the nearest neighbor parameters
            N (int): the number of samples to generate

        Returns:
            np.array: the generated samples
        """
        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=1,
                                                n_jobs=self.n_jobs,
                                                **nn_params)
        nnmt.fit(X_maj)

        stds = np.std(X_min, axis=0)
        # At one point the algorithm says to keep those points which are
        # the most distant from majority samples, and not leaving any minority
        # sample isolated. This can be implemented by generating multiple
        # samples for each point and keep the one most distant from the
        # majority samples.
        base_indices = self.random_state.choice(np.arange(X_min.shape[0]), N)
        base_vectors = X_min[base_indices]

        n_trials = 3
        offsets = self.random_state.normal(size=(base_vectors.shape[0],
                                                 n_trials,
                                                 base_vectors.shape[1]))
        samples = base_vectors[:, None, :] + offsets * stds[None, None, :]
        sample_stack = samples.reshape(base_vectors.shape[0] * n_trials, base_vectors.shape[1])
        dist, _ = nnmt.kneighbors(sample_stack)
        dist = dist[:, 0].reshape(base_vectors.shape[0], n_trials)
        samples = samples[np.arange(samples.shape[0]), np.argmax(dist, axis=1)]

        return samples

    def determine_removal(self, X_maj, nn_params, N): # pylint: disable=invalid-name
        """
        Determine which majority samples need to be removed.

        Args:
            X_maj (np.array): the majority vectors
            nn_params (dict): the nearest neighbor parameters
            N (int): the number of samples to generate

        Returns:
            np.array: the indices to remove
        """
        # normalize
        mms = MinMaxScaler()
        X_maj_normalized = mms.fit_transform(X_maj) # pylint: disable=invalid-name

        # computing the distance matrix
        distm = pairwise_distances_mahalanobis(X_maj_normalized,
                                             tensor=nn_params.get('metric_tensor', None))

        # len(X_maj) offsets for the diagonal 0 elements, 2N because
        # every distances appears twice
        threshold = sorted(distm.flatten())[np.min([X_maj.shape[0] + 2 * N,
                                                 distm.shape[0]**2 - 1])]
        np.fill_diagonal(distm, np.inf)

        # extracting the coordinates of pairs closer than threshold
        pairs_to_break = np.where(distm < threshold)
        pairs_to_break = np.vstack(pairs_to_break)

        # sorting the pairs, otherwise both points would be removed
        pairs_to_break.sort(axis=1)

        # uniqueing the coordinates - the final number might be less than N
        to_remove = np.unique(pairs_to_break[0])

        return to_remove

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        n_1 = len(X_min)
        n_0 = len(X) - n_1
        N = int(np.rint(0.5*n_0 - 0.5*n_1 + 0.5)) # pylint: disable=invalid-name

        if N == 0:
            return self.return_copies(X, y, "N is 0")

        # generating minority samples
        nn_params = {**self.nn_params}
        nn_params['metric_tensor'] = \
                    self.metric_tensor_from_nn_params(nn_params, X, y)

        samples = self.generate_samples(X_min, X_maj, nn_params, N)

        # Extending the minority dataset with the new samples
        X_min_extended = np.vstack([X_min, samples]) # pylint: disable=invalid-name

        # Removing N elements from the majority dataset
        to_remove = self.determine_removal(X_maj, nn_params, N)

        # removing the selected elements
        X_maj_cleaned = np.delete(X_maj, to_remove, axis=0) # pylint: disable=invalid-name

        return (np.vstack([X_min_extended, X_maj_cleaned]),
                np.hstack([np.repeat(self.min_label, len(X_min_extended)),
                           np.repeat(self.maj_label, len(X_maj_cleaned))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}

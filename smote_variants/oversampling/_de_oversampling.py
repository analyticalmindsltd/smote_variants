"""
This module implements the DE_oversampling method.
"""
import warnings

import numpy as np

from sklearn.cluster import KMeans
from ..config import suppress_external_warnings
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSampling

from .._logger import logger
_logger= logger

__all__= ['DE_oversampling']

class DE_oversampling(OverSampling): # pylint: disable=invalid-name
    """
    References:
        * BibTex::

            @INPROCEEDINGS{de_oversampling,
                            author={Chen, L. and Cai, Z. and Chen, L. and
                                    Gu, Q.},
                            booktitle={2010 Third International Conference
                                       on Knowledge Discovery and Data Mining},
                            title={A Novel Differential Evolution-Clustering
                                    Hybrid Resampling Algorithm on Imbalanced
                                    Datasets},
                            year={2010},
                            volume={},
                            number={},
                            pages={81-85},
                            keywords={pattern clustering;sampling methods;
                                        support vector machines;differential
                                        evolution;clustering algorithm;hybrid
                                        resampling algorithm;imbalanced
                                        datasets;support vector machine;
                                        minority class;mutation operators;
                                        crossover operators;data cleaning
                                        method;F-measure criterion;ROC area
                                        criterion;Support vector machines;
                                        Intrusion detection;Support vector
                                        machine classification;Cleaning;
                                        Electronic mail;Clustering algorithms;
                                        Signal to noise ratio;Learning
                                        systems;Data mining;Geology;imbalanced
                                        datasets;hybrid resampling;clustering;
                                        differential evolution;support vector
                                        machine},
                            doi={10.1109/WKDD.2010.48},
                            ISSN={},
                            month={Jan},}
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 crossover_rate=0.5,
                 similarity_threshold=0.5,
                 n_clusters=30,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor
                                component
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            crossover_rate (float): cross over rate of evoluation
            similarity_threshold (float): similarity threshold parameter
            n_clusters (int): number of clusters for cleansing
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state)
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 2)
        self.check_in_range(crossover_rate, 'crossover_rate', [0, 1])
        self.check_in_range(similarity_threshold,
                            'similarity_threshold', [0, 1])
        self.check_greater_or_equal(n_clusters, 'n_clusters', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.crossover_rate = crossover_rate
        self.similarity_threshold = similarity_threshold
        self.n_clusters = n_clusters
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
                                  'crossover_rate': [0.1, 0.5, 0.9],
                                  'similarity_threshold': [0.5, 0.9],
                                  'n_clusters': [10, 20, 50]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def generate_samples(self, n_to_sample, X_min, indices):
        """
        Generate samples

        Args:
            n_to_sample (int): number of samples to generate
            X_min (np.array): minority samples
            indices (np.array): the neighborhood graph

        Returns:
            np.array: the generated samples
        """
        n_dim = X_min.shape[1]
        samples = []
        for _ in range(n_to_sample):
            # mutation according to the description in the paper
            random_index = self.random_state.randint(len(X_min))
            random_point = X_min[random_index]
            random_neighbor_indices = self.random_state.choice(
                indices[random_index][1:], 2, replace=False)
            random_neighbor_1 = X_min[random_neighbor_indices[0]]
            random_neighbor_2 = X_min[random_neighbor_indices[1]]

            mutated = random_point + \
                (random_neighbor_1 - random_neighbor_2) * \
                self.random_state.random_sample()

            # crossover - updates the vector 'mutated'
            rand_s = self.random_state.randint(n_dim)
            for idx in range(n_dim):
                random_value = self.random_state.random_sample()
                if random_value >= self.crossover_rate and not idx == rand_s:
                    mutated[idx] = random_point[idx]
                elif random_value < self.crossover_rate or idx == rand_s:
                    pass

            samples.append(mutated)
        return samples

    def cleansing(self, X, y):
        """
        Cleansing

        Args:
            X (np.array): the extended features
            y (np.array): the extended target labels

        Returns:
            np.array: indices to remove
        """
        # cleansing based on clustering
        kmeans = KMeans(n_clusters=min([len(X), self.n_clusters]),
                        random_state=self._random_state_init)

        with warnings.catch_warnings():
            if suppress_external_warnings():
                warnings.simplefilter("ignore")
            kmeans.fit(X)

        def cluster_filter(label):
            return len(np.unique(y[np.where(kmeans.labels_ == label)[0]])) == 1

        one_label_clusters = [label \
                    for label in np.unique(kmeans.labels_) if cluster_filter(label)]
        to_remove = []

        # going through the clusters having one label only
        for label in one_label_clusters:
            cluster_indices = np.where(kmeans.labels_ == label)[0]
            mean_of_cluster = kmeans.cluster_centers_[label]

            # finding center-like sample
            tmp = np.linalg.norm(X[cluster_indices] - mean_of_cluster, axis=1)
            center_like_index = np.argmin(tmp)
            center_like_index = cluster_indices[center_like_index]

            # removing the samples similar to the center-like sample
            for idx in cluster_indices:
                if idx != center_like_index:
                    denom = (np.linalg.norm(X[idx]) *
                                        np.linalg.norm(X[center_like_index]))
                    num = np.inner(X[idx], X[center_like_index])

                    if denom == 0.0 or (num / denom) > self.similarity_threshold:
                        to_remove.append(idx)

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
        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        X_min = X[y == self.min_label]

        n_neighbors = min([len(X_min), self.n_neighbors+1])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X_min)
        indices = nnmt.kneighbors(X_min, return_distance=False)

        # generating samples
        samples = self.generate_samples(n_to_sample, X_min, indices)

        # assembling all data for clearning
        X, y = np.vstack([X, np.vstack(samples)]), np.hstack(
            [y, np.repeat(self.min_label, len(samples))])

        to_remove = self.cleansing(X, y)

        return np.delete(X, to_remove, axis=0), np.delete(y, to_remove)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'crossover_rate': self.crossover_rate,
                'similarity_threshold': self.similarity_threshold,
                'n_clusters': self.n_clusters,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}

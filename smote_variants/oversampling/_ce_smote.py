"""
This module implements the CE_SMOTE method.
"""
import warnings

import numpy as np

from sklearn.cluster import KMeans
import scipy.optimize as soptimize

from ..base import coalesce_dict, coalesce
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['CE_SMOTE']

class CE_SMOTE(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{ce_smote,
                                author={Chen, S. and Guo, G. and Chen, L.},
                                booktitle={2010 IEEE 24th International
                                            Conference on Advanced Information
                                            Networking and Applications
                                            Workshops},
                                title={A New Over-Sampling Method Based on
                                        Cluster Ensembles},
                                year={2010},
                                volume={},
                                number={},
                                pages={599-604},
                                keywords={data mining;Internet;pattern
                                            classification;pattern clustering;
                                            over sampling method;cluster
                                            ensembles;classification method;
                                            imbalanced data handling;CE-SMOTE;
                                            clustering consistency index;
                                            cluster boundary minority samples;
                                            imbalanced public data set;
                                            Mathematics;Computer science;
                                            Electronic mail;Accuracy;Nearest
                                            neighbor searches;Application
                                            software;Data mining;Conferences;
                                            Web sites;Information retrieval;
                                            classification;imbalanced data
                                            sets;cluster ensembles;
                                            over-sampling},
                                doi={10.1109/WAINA.2010.40},
                                ISSN={},
                                month={April}}
    """

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_borderline,
                  OverSamplingSimplex.cat_uses_clustering,
                  OverSamplingSimplex.cat_sample_ordinary,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 *,
                 h=10,
                 k=5,
                 nn_params=None,
                 ss_params=None,
                 alpha=0.5,
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
            h (int): size of ensemble
            k (int): number of clusters/neighbors
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            alpha (float): [0,1] threshold to select boundary samples
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(h, "h", 1)
        self.check_greater_or_equal(k, "k", 1)
        self.check_in_range(alpha, "alpha", [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.h = h # pylint: disable=invalid-name
        self.k = k # pylint: disable=invalid-name
        self.nn_params = coalesce(nn_params, {})
        self.alpha = alpha
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
                                  'h': [5, 10, 15],
                                  'k': [3, 5, 7],
                                  'alpha': [0.2, 0.5, 0.8]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def do_the_clustering(self, X):
        """
        Do the clustering.

        Args:
            X (np.array): all features

        Returns:
            np.array: the labels
        """
        n_dim = X.shape[1]
        labels = []
        for _ in range(self.h):
            n_features = self.random_state.randint(np.max([int(n_dim/2), 1]), n_dim + 1)
            features = self.random_state.choice(n_dim, n_features, replace=False)
            n_clusters = min([len(X), self.k])
            kmeans = KMeans(n_clusters=n_clusters,
                            random_state=self._random_state_init)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                kmeans.fit(X[:, features])
            labels.append(kmeans.labels_)

        return labels

    def cluster_matching(self, labels):
        """
        Do the cluster matching.

        Args:
            labels (list): labelings

        Returns:
            np.array: the cluster matched labeling
        """
        base_label = 0
        for idx, labeling in enumerate(labels):
            if not idx == base_label:
                cost_matrix = np.zeros(shape=(self.k, self.k))

                for jdx in range(self.k):
                    mask_j = labels[base_label] == jdx
                    for kdx in range(self.k):
                        mask_k = labeling == kdx
                        mask_jk = np.logical_and(mask_j, mask_k)
                        cost_matrix[jdx, kdx] = np.sum(mask_jk)

                # solving the assignment problem
                row_ind, _ = soptimize.linear_sum_assignment(-cost_matrix)

                # doing the relabeling
                relabeling = labeling.copy()

                for jdx in range(len(row_ind)):
                    relabeling[labeling == kdx] = jdx

                labels[idx] = relabeling

        return np.vstack(labels)

    def cluster_consistency_index(self, labels):
        """
        Computing the cluster consistency index

        Args:
            np.array: labeling

        Returns:
            np.array: the cluster consistency indices
        """
        cci = np.apply_along_axis(lambda x: max(set(x.tolist()),
                                    key=x.tolist().count), 0, labels)
        cci = np.sum(labels == cci, axis=0)
        cci = cci / self.h

        return cci

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
            self.return_copies(X, y, "Sampling is not needed")

        # do the clustering and labelling
        labels = self.do_the_clustering(X)

        # do the cluster matching, clustering 0 will be considered the one to
        # match the others to the problem of finding cluster matching is
        # basically the "assignment problem"
        labels = self.cluster_matching(labels)

        # compute clustering consistency index
        cci = self.cluster_consistency_index(labels)

        # determining minority boundary samples
        P_boundary = X[(y == self.min_label) & (cci <= self.alpha)] # pylint: disable=invalid-name

        # there might be no boundary samples
        if len(P_boundary) <= self.n_dim - 1:
            return self.return_copies(X, y, "empty boundary")

        # finding nearest neighbors of boundary samples
        n_neighbors = min([len(P_boundary), self.k])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= \
                        self.metric_tensor_from_nn_params(nn_params, X, y)

        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(P_boundary)
        ind = nnmt.kneighbors(P_boundary, return_distance=False)

        samples = self.sample_simplex(X=P_boundary,
                                        indices=ind,
                                        n_to_sample=n_to_sample)

        # do the sampling
        #samples = []
        #for _ in range(n_to_sample):
        #    idx = self.random_state.randint(len(ind))
        #    point_a = P_boundary[idx]
        #    point_b = P_boundary[self.random_state.choice(ind[idx][1:])]
        #    samples.append(self.sample_between_points(point_a, point_b))

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'h': self.h,
                'k': self.k,
                'nn_params': self.nn_params,
                'alpha': self.alpha,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}

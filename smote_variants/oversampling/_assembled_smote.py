"""
This module implements the Assembled_SMOTE method.
"""
import warnings

import numpy as np

from sklearn.decomposition import PCA

from ..config import suppress_external_warnings
from ..base import coalesce_dict, coalesce
from ..base import (NearestNeighborsWithMetricTensor,
                                pairwise_distances_mahalanobis)
from ..base import OverSamplingSimplex

from .._logger import logger
_logger= logger

__all__= ['Assembled_SMOTE']

class Assembled_SMOTE(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{assembled_smote,
                            author={Zhou, B. and Yang, C. and Guo, H. and
                                        Hu, J.},
                            booktitle={The 2013 International Joint Conference
                                        on Neural Networks (IJCNN)},
                            title={A quasi-linear SVM combined with assembled
                                    SMOTE for imbalanced data classification},
                            year={2013},
                            volume={},
                            number={},
                            pages={1-7},
                            keywords={approximation theory;interpolation;
                                        pattern classification;sampling
                                        methods;support vector machines;trees
                                        (mathematics);quasilinear SVM;
                                        assembled SMOTE;imbalanced dataset
                                        classification problem;oversampling
                                        method;quasilinear kernel function;
                                        approximate nonlinear separation
                                        boundary;mulitlocal linear boundaries;
                                        interpolation;data distribution
                                        information;minimal spanning tree;
                                        local linear partitioning method;
                                        linear separation boundary;synthetic
                                        minority class samples;oversampled
                                        dataset classification;standard SVM;
                                        composite quasilinear kernel function;
                                        artificial data datasets;benchmark
                                        datasets;classification performance
                                        improvement;synthetic minority
                                        over-sampling technique;Support vector
                                        machines;Kernel;Merging;Standards;
                                        Sociology;Statistics;Interpolation},
                            doi={10.1109/IJCNN.2013.6707035},
                            ISSN={2161-4407},
                            month={Aug}}

    Notes:
        * Absolute value of the angles extracted should be taken.
            (implemented this way)
        * It is not specified how many samples are generated in the various
            clusters.
    """

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_uses_clustering,
                  OverSamplingSimplex.cat_borderline,
                  OverSamplingSimplex.cat_sample_ordinary,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 pop=2,
                 thres=0.3,
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
            n_neighbors (int): number of neighbors in nearest neighbors
                                component
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
            pop (int): lower threshold on cluster sizes
            thres (float): threshold on angles
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
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(pop, "pop", 1)
        self.check_in_range(thres, "thres", [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
        self.pop = pop
        self.thres = thres
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
                                  'pop': [2, 4, 5],
                                  'thres': [0.1, 0.3, 0.5]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def determine_border_non_border(self, X, y, X_min, nn_params):
        """
        Determine the border and non-border samples

        Args:
            X (np.array): all samples
            y (np.array): the target labels
            X_min (np.array): the minority samples
            nn_params (dict): the parameters of nearest neighbors

        Returns:
            np.array, np.array: the border and non-border samples
        """
        # fitting nearest neighbors model
        n_neighbors = min([len(X), self.n_neighbors+1])
        nearestn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nearestn.fit(X)
        ind = nearestn.kneighbors(X_min, return_distance=False)

        # finding the set of border and non-border minority elements
        n_min_neighbors = [np.sum(y[ind[i]] == self.min_label)
                           for i in range(len(ind))]
        border_mask = np.logical_not(np.array(n_min_neighbors) == n_neighbors)
        X_border = X_min[border_mask] # pylint: disable=invalid-name
        X_non_border = X_min[np.logical_not(border_mask)] # pylint: disable=invalid-name

        return X_border, X_non_border

    def do_the_clustering(self,
                            X_border,  # pylint: disable=invalid-name
                            nn_params):
        """
        Carry out the clustering of the border samples.

        Args:
            X_border (np.array): the border samples
            nn_params (dict): the nearest neighbor parameters

        Returns:
            list(np.array): the clusters
        """

        # initializing clustering
        clusters = [np.array([i]) for i in range(len(X_border))]

        distm = pairwise_distances_mahalanobis(X_border,
                                            tensor=nn_params.get('metric_tensor', None))
        for idx in range(len(distm)):
            distm[idx, idx] = np.inf

        # do the clustering
        while len(distm) > 1 and np.min(distm) < np.inf:
            # extracting coordinates of clusters with the minimum distance
            min_coord = np.where(distm == np.min(distm))
            merge_a = min_coord[0][0]
            merge_b = min_coord[1][0]

            # checking the size of clusters to see if they should be merged
            if (len(clusters[merge_a]) < self.pop
                    or len(clusters[merge_b]) < self.pop):
                # if both clusters are small, do the merge
                clusters[merge_a] = np.hstack([clusters[merge_a],
                                               clusters[merge_b]])
                del clusters[merge_b]
                # update the distance matrix accordingly
                distm[merge_a] = np.min(np.vstack([distm[merge_a], distm[merge_b]]),
                                     axis=0)
                distm[:, merge_a] = distm[merge_a]
                # remove columns
                distm = np.delete(distm, merge_b, axis=0)
                distm = np.delete(distm, merge_b, axis=1)
                # fix the diagonal entries
                for idx in range(len(distm)):
                    distm[idx, idx] = np.inf
            else:
                # otherwise find principal directions
                with warnings.catch_warnings():
                    if suppress_external_warnings():
                        warnings.simplefilter('ignore')
                    pca_a = PCA(n_components=1).fit(X_border[clusters[merge_a]])
                    pca_b = PCA(n_components=1).fit(X_border[clusters[merge_b]])
                # extract the angle of principal directions
                numerator = np.dot(pca_a.components_[0], pca_b.components_[0])
                denominator = np.linalg.norm(pca_a.components_[0])
                denominator *= np.linalg.norm(pca_b.components_[0])
                angle = abs(numerator/denominator)
                # check if angle if angle is above a specific threshold
                if angle > self.thres:
                    # do the merge
                    clusters[merge_a] = np.hstack([clusters[merge_a],
                                                   clusters[merge_b]])
                    del clusters[merge_b]
                    # update the distance matrix acoordingly
                    distm[merge_a] = np.min(np.vstack([distm[merge_a], distm[merge_b]]),
                                         axis=0)
                    distm[:, merge_a] = distm[merge_a]
                    # remove columns
                    distm = np.delete(distm, merge_b, axis=0)
                    distm = np.delete(distm, merge_b, axis=1)
                    # fixing the digaonal entries
                    for idx in range(len(distm)):
                        distm[idx, idx] = np.inf
                else:
                    # changing the distance of clusters to fininte
                    distm[merge_a, merge_b] = np.inf
                    distm[merge_b, merge_a] = np.inf

        return clusters

    def determine_indices(self, vectors, nn_params):
        """
        Determines the neighborhood structure within the cluster

        Args:
            vectors (np.array): vectors within the cluster
            nn_params (dict): the nearest neighbor parameters

        Returns:
            np.array: the neighborhood structure
        """
        n_neighbors = np.min([self.n_neighbors + 1, len(vectors)])
        nearestn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))

        indices = nearestn.fit(vectors).kneighbors(vectors,
                                                    return_distance=False)

        return indices

    def generate_samples_in_clusters(self, vectors, n_to_sample, nn_params):
        """
        Generate samples within the clusters.

        Args:
            vectors (list(np.array)): the vectors of the clusters
            n_to_sample (int): the overall number of samples to generate
            nn_params (dict): the nearest neighbors parameters

        Returns:
            np.array: the generated samples
        """
        # extract cluster sizes and calculating point distribution in clusters
        # the last element of the clusters is the set of non-border xamples
        cluster_sizes = np.array([len(vect) for vect in vectors])
        cluster_density = cluster_sizes/np.sum(cluster_sizes)

        cluster_indices = self.random_state.choice(len(cluster_sizes),
                                                    n_to_sample,
                                                    p=cluster_density)

        cluster_unique, cluster_count = np.unique(cluster_indices,
                                                    return_counts=True)

        samples = []
        for idx, cluster in enumerate(cluster_unique):
            indices = self.determine_indices(vectors[cluster], nn_params)

            #if len(vectors[cluster]) >= self.n_dim:
            samples.append(self.sample_simplex(X=vectors[cluster],
                                            indices=indices,
                                            n_to_sample=cluster_count[idx]))
            #else:
            #    sample_indices = self.random_state.choice(np.arange(len(vectors[cluster])),
            #                                                cluster_count[idx])
            #    samples.append(vectors[cluster][sample_indices])

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

        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed.")

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= \
                        self.metric_tensor_from_nn_params(nn_params, X, y)

        X_min = X[y == self.min_label]

        X_border, X_non_border = self.determine_border_non_border(X, y, # pylint: disable=invalid-name
                                                                    X_min,
                                                                    nn_params)

        if len(X_border) == 0:
            return self.return_copies(X, y, "X_border is empty")

        clusters = self.do_the_clustering(X_border, nn_params)

        # extract vectors belonging to the various clusters
        vectors = [X_border[c] for c in clusters if len(c) > 0]
        # adding non-border samples
        if len(X_non_border) > 0:
            vectors.append(X_non_border)

        samples = self.generate_samples_in_clusters(vectors, n_to_sample, nn_params)

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
                'pop': self.pop,
                'thres': self.thres,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}

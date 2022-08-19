"""
This module implements the MWMOTE method.
"""
import warnings

import numpy as np

from sklearn.cluster import KMeans
from scipy.linalg import circulant

from ..base import coalesce, coalesce_dict
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['MWMOTE']

class MWMOTE(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @ARTICLE{mwmote,
                        author={Barua, S. and Islam, M. M. and Yao, X. and
                                Murase, K.},
                        journal={IEEE Transactions on Knowledge and Data
                                Engineering},
                        title={MWMOTE--Majority Weighted Minority Oversampling
                                Technique for Imbalanced Data Set Learning},
                        year={2014},
                        volume={26},
                        number={2},
                        pages={405-425},
                        keywords={learning (artificial intelligence);pattern
                                    clustering;sampling methods;AUC;area under
                                    curve;ROC;receiver operating curve;G-mean;
                                    geometric mean;minority class cluster;
                                    clustering approach;weighted informative
                                    minority class samples;Euclidean distance;
                                    hard-to-learn informative minority class
                                    samples;majority class;synthetic minority
                                    class samples;synthetic oversampling
                                    methods;imbalanced learning problems;
                                    imbalanced data set learning;
                                    MWMOTE-majority weighted minority
                                    oversampling technique;Sampling methods;
                                    Noise measurement;Boosting;Simulation;
                                    Complexity theory;Interpolation;Abstracts;
                                    Imbalanced learning;undersampling;
                                    oversampling;synthetic sample generation;
                                    clustering},
                        doi={10.1109/TKDE.2012.232},
                        ISSN={1041-4347},
                        month={Feb}}

    Notes:
        * The original method was not prepared for the case of having clusters
            of 1 elements.
        * It is not clear if within the cluster of the informative minority item
            the informative minority item should be choosen again
    """

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_uses_clustering,
                  OverSamplingSimplex.cat_borderline,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 *,
                 k1=5,
                 k2=5,
                 k3=5,
                 M=10,
                 cf_th=5.0,
                 cmax=10.0,
                 nn_params=None,
                 ss_params=None,
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
            k1 (int): parameter of the NearestNeighbors component
            k2 (int): parameter of the NearestNeighbors component
            k3 (int): parameter of the NearestNeighbors component
            M (int): number of clusters
            cf_th (float): cutoff threshold
            cmax (float): maximum closeness value
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        nn_params = coalesce(nn_params, {})

        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(k1, 'k1', 1)
        self.check_greater_or_equal(k2, 'k2', 1)
        self.check_greater_or_equal(k3, 'k3', 1)
        self.check_greater_or_equal(M, 'M', 1)
        self.check_greater_or_equal(cf_th, 'cf_th', 0)
        self.check_greater_or_equal(cmax, 'cmax', 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.params = {'k1': k1,
                        'k2': k2,
                        'k3': k3,
                        'M': M,
                        'cf_th': cf_th,
                        'cmax': cmax}
        self.nn_params = nn_params
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
                                  'k1': [5, 9],
                                  'k2': [5, 9],
                                  'k3': [5, 9],
                                  'M': [4, 10],
                                  'cf_th': [5.0],
                                  'cmax': [10.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def filter_minorities(self, X, y, nn_params, minority):
        """
        Filter minority samples.

        Args:
            X (np.array): all training vectors
            y (np.array): all target labels
            nn_params (dict): the nearest neighbors parameters
            minority (np.array): the minority indices

        Returns:
            np.array: the filtered minority indices
        """
        # Step 1
        n_neighbors = min([len(X), self.params['k1'] + 1])
        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X)
        ind1 = nnmt.kneighbors(X, return_distance=False)

        # Step 2
        arr = [i for i in minority if np.sum(y[ind1[i][1:]] == self.min_label)]
        filtered_minority = np.array(arr)

        return filtered_minority

    def determine_border_majority(self, X, X_maj,
                                    filtered_minority,
                                    nn_params):
        """
        Determine border majority samples.

        Args:
            X (np.array): all training samples
            X_maj (np.array): all majority samples
            filtered_minority (np.array): filtered minority samples
            nn_params (dict): nearest neighbors parameters

        Returns:
            np.array: the bordering majority samples
        """
        # Step 3 - ind2 needs to be indexed by indices of the lengh of X_maj
        n_neighbors = min([len(X_maj), self.params['k2'] + 1])
        nn_maj= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                    n_jobs=self.n_jobs,
                                                    **(nn_params))
        nn_maj.fit(X_maj)
        ind2 = nn_maj.kneighbors(X[filtered_minority], return_distance=False)

        # Step 4
        border_majority = np.unique(ind2.flatten())

        return border_majority

    def determine_informative_minority(self, X_min, X_maj,
                                        border_majority, nn_params):
        """
        Determine the informative minority samples.

        Args:
            X_min (np.array): minority samples
            X_maj (np.array): majority samples
            border_majority (np.array): border majority samples
            nn_params (dict): the nearest neighbors parameters

        Returns:
            np.array: indices of informative minority samples
        """

        # Step 5 - ind3 needs to be indexed by indices of the length of X_min
        n_neighbors = min([self.params['k3'], len(X_min)])
        nn_min = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                    n_jobs=self.n_jobs,
                                                    **(nn_params))
        nn_min.fit(X_min)
        ind3 = nn_min.kneighbors(X_maj[border_majority], return_distance=False)

        # Step 6 - informative minority indexes X_min
        informative_minority = np.unique(ind3.flatten())

        return informative_minority

    def closeness_factor(self, y_vec, x_vec):
        """
        Closeness factor according to the Eq (6)

        Args:
            y_vec (np.array): training instance (border_majority)
            x_vec (np.array): training instance (informative_minority)
            cf_th (float): cutoff threshold
            cmax (float): maximum values

        Returns:
            float: closeness factor
        """
        dist = np.linalg.norm(y_vec - x_vec) / y_vec.shape[0]

        #if dist == 0:
        #    dist = 0.1

        dist = np.max([0.1, dist])

        f_val = np.min([1.0 / dist, self.params['cf_th']])

        return f_val / self.params['cf_th'] * self.params['cmax']

    def determine_information_weights(self,
                                    X_maj,
                                    X_min,
                                    border_majority,
                                    informative_minority):
        """
        Determine information weights.

        Args:
            X_maj (np.array): majority samples
            X_min (np.array): minority samples
            border_majority (np.array): border majority indices
            informative minority (np.array): informative minority indices

        Returns:
            np.array: information weights
        """
        closeness_factors = np.zeros(shape=(border_majority.shape[0],
                                            informative_minority.shape[0]))

        for idx, bm_i in enumerate(border_majority):
            for jdx, im_j in enumerate(informative_minority):
                closeness_factors[idx, jdx] = self.closeness_factor(X_maj[bm_i],
                                                                    X_min[im_j])

        _logger.info("%s: computing information weights", self.__class__.__name__)
        information_weights = np.zeros(shape=(border_majority.shape[0],
                                              informative_minority.shape[0]))
        for idx in range(border_majority.shape[0]):
            norm_factor = np.sum(closeness_factors[idx, :])
            for jdx in range(informative_minority.shape[0]):
                cf_ij = closeness_factors[idx, jdx]
                information_weights[idx, jdx] = cf_ij**2 / norm_factor

        return information_weights

    def determine_selection_probabilities(self, *, X, y,
                                            X_min,
                                            filtered_minority,
                                            nn_params):
        """
        Determine selection probabilities.

        Args:
            X (np.array): all training vectors
            y (np.array): all target labels
            X_min (np.array): minority samples
            filtered_minority (np.array): indices of the filtered
                                            minority samples
            minority (np.array): the minority indices
            nn_params (dict): the nearest neighbors parameters

        Returns:
            np.array, np.array: indices of informative minority and
                                selection probabilities
        """
        X_maj = X[y == self.maj_label]

        border_majority = self.determine_border_majority(X, X_maj,
                                                            filtered_minority,
                                                            nn_params)

        informative_minority = self.determine_informative_minority(X_min, X_maj,
                                                    border_majority, nn_params)

        # Steps 7 - 9
        _logger.info("%s: computing closeness factors", self.__class__.__name__)

        information_weights = self.determine_information_weights(X_maj,
                                                        X_min,
                                                        border_majority,
                                                        informative_minority)

        selection_weights = np.sum(information_weights, axis=0)
        selection_probabilities = selection_weights/np.sum(selection_weights)

        return informative_minority, selection_probabilities

    def determine_cluster_probs(self, informative_minority,
                                    selection_probabilities,
                                    cluster_labels):
        """
        Determine cluster probabilities.

        Args:
            informative_minority (np.array): indices of informative minority
                                                sample indices
            selection_probabilities (np.array): selection probabilities
            cluster_labels (np.array): cluster labels

        Returns:
            np.array: the cluster probabilities
        """
        cluster_weights = np.zeros(shape=(len(np.unique(cluster_labels)),))
        for idx, p_idx in enumerate(informative_minority):
            cluster_weights[cluster_labels[p_idx]] += \
                                    selection_probabilities[idx]

        return cluster_weights / np.sum(cluster_weights)

    def determine_within_prob(self,
                                cluster,
                                informative_minority,
                                selection_probabilities):
        """
        Determine the within cluster probabilities.

        Args:
            cluster (np.array): the cluster indices
            informative_minority (np.array): the informative minority
                                                indices
            selection_probabilities (np.array): the informative minority
                                                probabilities

        Returns:
            np.array: the within cluster probabilities
        """
        within_prob = np.zeros(shape=(len(cluster),))

        for idx, c_idx in enumerate(cluster):
            where = np.where(informative_minority == c_idx)[0]
            if len(where) > 0:
                within_prob[idx] += selection_probabilities[where[0]]

        within_prob = within_prob / np.sum(within_prob)

        return within_prob

    def sample_clusters(self,
                        informative_minority,
                        selection_probabilities,
                        cluster_labels,
                        n_to_sample):
        """
        Sample the clusters.

        Args:
            informative_minority (np.array): indices of the informative
                                                minority
            selection_probabilities (np.array): the selection probabilities
            cluster_labels (np.array): the cluster labels
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array, np.array: the selected cluster labels and counts
        """
        cluster_probs = self.determine_cluster_probs(informative_minority,
                                                    selection_probabilities,
                                                    cluster_labels)

        clusters_selected = self.random_state.choice(len(cluster_probs),
                                                        n_to_sample,
                                                        p=cluster_probs)

        cluster_unique, cluster_count = np.unique(clusters_selected,
                                                    return_counts=True)

        return cluster_unique, cluster_count

    def generate_samples_in_clusters(self, *, X_min, clusters,
                                    cluster_labels,
                                    informative_minority,
                                    selection_probabilities,
                                    n_to_sample):
        """
        Generate samples in clusters.

        Args:
            X_min (np.array): the minority vectors
            clusters (np.array): the clusters
            cluster_labels (np.array): the cluster labeling
            informative_minority (np.array): indices of the informative
                                                minority
            selection_probabilities (np.array): the selection probabilities
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array: the generated samples
        """
        # Step 11-12
        cluster_unique, cluster_count = \
                self.sample_clusters(informative_minority,
                                    selection_probabilities,
                                    cluster_labels,
                                    n_to_sample)

        #n_dim_original = self.n_dim
        samples = []
        for idx, cluster in enumerate(cluster_unique):
            cluster_vectors = X_min[clusters[cluster]]
            within_prob = self.determine_within_prob(clusters[cluster],
                                                    informative_minority,
                                                    selection_probabilities)

            #self.n_dim = np.min([self.n_dim, cluster_vectors.shape[0]])
            samples.append(self.sample_simplex(X=cluster_vectors,
                                    indices=circulant(np.arange(cluster_vectors.shape[0])),
                                    n_to_sample=cluster_count[idx],
                                    base_weights=within_prob))
            #self.n_dim = n_dim_original
        #"""


        #for cluster in clusters:
        #    if len(cluster) == 1:
        #        print(X_min[cluster])
        #
        #samples=[]
        ## Step 12
        #
        #for _ in range(n_to_sample):
        #    random_index = self.random_state.choice(informative_minority,
        #                                            p=selection_probabilities)
        #    cluster_label = cluster_labels[random_index]
        #    cluster = clusters[cluster_label]
        #    random_index_in_cluster = self.random_state.choice(cluster)
        #    print(cluster_label, random_index, random_index_in_cluster)
        #    X_random = X_min[random_index]
        #    X_random_cluster = X_min[random_index_in_cluster]
        #    #samples.append(self.sample_between_points(X_random,
        #    #                                          X_random_cluster))
        #    samples.append(X_random + (X_random_cluster - X_random)\
        # *self.random_state.random_sample())

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
            return self.return_copies(X, y, "Sampling is not needed")

        X_min = X[y == self.min_label]

        nn_params= {**self.nn_params}
        nn_params['metric_tensor'] = \
                self.metric_tensor_from_nn_params(nn_params, X, y)

        filtered_minority = self.filter_minorities(X, y, nn_params,
                                            np.where(y == self.min_label)[0])

        if len(filtered_minority) == 0:
            return self.return_copies(X, y,
                        "No minority samples remaining after filtering")

        informative_minority, selection_probabilities = \
                self.determine_selection_probabilities(X=X, y=y,
                                                        X_min=X_min,
                                                        filtered_minority=filtered_minority,
                                                        nn_params=nn_params)

        # Step 10
        _logger.info("%s: do clustering", self.__class__.__name__)
        kmeans = KMeans(n_clusters=np.min([len(np.unique(X_min, axis=1)),
                                            self.params['M']]),
                        random_state=self._random_state_init)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            kmeans.fit(X_min)
        #imin_labels = kmeans.labels_[informative_minority]

        clusters = [np.where(kmeans.labels_ == i)[0]
                    for i in range(np.max(kmeans.labels_)+1)]

        samples = self.generate_samples_in_clusters(X_min=X_min,
                            clusters=clusters,
                            cluster_labels=kmeans.labels_,
                            informative_minority=informative_minority,
                            selection_probabilities=selection_probabilities,
                            n_to_sample=n_to_sample)

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'k1': self.params['k1'],
                'k2': self.params['k2'],
                'k3': self.params['k3'],
                'M': self.params['M'],
                'cf_th': self.params['cf_th'],
                'cmax': self.params['cmax'],
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}

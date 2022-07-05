import numpy as np

from sklearn.cluster import KMeans
import scipy.optimize as soptimize

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['CE_SMOTE']

class CE_SMOTE(OverSampling):
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

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_borderline,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 *,
                 h=10,
                 k=5,
                 nn_params={},
                 alpha=0.5,
                 n_jobs=1,
                 random_state=None):
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
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(h, "h", 1)
        self.check_greater_or_equal(k, "k", 1)
        self.check_in_range(alpha, "alpha", [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.h = h
        self.k = k
        self.nn_params = nn_params
        self.alpha = alpha
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

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

    def sample(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # do the clustering and labelling
        d = len(X[0])
        labels = []
        for _ in range(self.h):
            f = self.random_state.randint(int(d/2), d)
            features = self.random_state.choice(np.arange(d), f)
            n_clusters = min([len(X), self.k])
            kmeans = KMeans(n_clusters=n_clusters,
                            random_state=self._random_state_init)
            kmeans.fit(X[:, features])
            labels.append(kmeans.labels_)

        # do the cluster matching, clustering 0 will be considered the one to
        # match the others to the problem of finding cluster matching is
        # basically the "assignment problem"
        base_label = 0
        for i in range(len(labels)):
            if not i == base_label:
                cost_matrix = np.zeros(shape=(self.k, self.k))
                for j in range(self.k):
                    mask_j = labels[base_label] == j
                    for k in range(self.k):
                        mask_k = labels[i] == k
                        mask_jk = np.logical_and(mask_j, mask_k)
                        cost_matrix[j, k] = np.sum(mask_jk)
                # solving the assignment problem
                row_ind, _ = soptimize.linear_sum_assignment(-cost_matrix)
                # doing the relabeling
                relabeling = labels[i].copy()
                for j in range(len(row_ind)):
                    relabeling[labels[i] == k] = j
                labels[i] = relabeling

        # compute clustering consistency index
        labels = np.vstack(labels)
        cci = np.apply_along_axis(lambda x: max(
            set(x.tolist()), key=x.tolist().count), 0, labels)
        cci = np.sum(labels == cci, axis=0)
        cci = cci/self.h

        # determining minority boundary samples
        P_boundary = X[np.logical_and(
            y == self.min_label, cci < self.alpha)]

        # there might be no boundary samples
        if len(P_boundary) <= 1:
            _logger.warning(self.__class__.__name__ + ": " + "empty boundary")
            return X.copy(), y.copy()

        # finding nearest neighbors of boundary samples
        n_neighbors = min([len(P_boundary), self.k])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(P_boundary)
        ind = nn.kneighbors(P_boundary, return_distance=False)

        # do the sampling
        samples = []
        for _ in range(n_to_sample):
            idx = self.random_state.randint(len(ind))
            point_a = P_boundary[idx]
            point_b = P_boundary[self.random_state.choice(ind[idx][1:])]
            samples.append(self.sample_between_points(point_a, point_b))

        return (np.vstack([X, np.vstack(samples)]),
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
                'random_state': self._random_state_init}

import numpy as np

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['SMOTE_D']

class SMOTE_D(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{smote_d,
                            author="Torres, Fredy Rodr{\'i}guez
                            and Carrasco-Ochoa, Jes{\'u}s A.
                            and Mart{\'i}nez-Trinidad, Jos{\'e} Fco.",
                            editor="Mart{\'i}nez-Trinidad, Jos{\'e} Francisco
                            and Carrasco-Ochoa, Jes{\'u}s Ariel
                            and Ayala Ramirez, Victor
                            and Olvera-L{\'o}pez, Jos{\'e} Arturo
                            and Jiang, Xiaoyi",
                            title="SMOTE-D a Deterministic Version of SMOTE",
                            booktitle="Pattern Recognition",
                            year="2016",
                            publisher="Springer International Publishing",
                            address="Cham",
                            pages="177--188",
                            isbn="978-3-319-39393-3"
                            }

    Notes:
        * Copying happens if two points are the neighbors of each other.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_metric_learning]

    def __init__(self, 
                 proportion=1.0, 
                 k=3, 
                 *,
                 nn_params={},
                 n_jobs=1, 
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            k (int): number of neighbors in nearest neighbors component
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(k, "k", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.k = k
        self.nn_params = nn_params
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
                                  'k': [3, 5, 7]}
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

        X_min = X[y == self.min_label]

        # fitting nearest neighbors model
        n_neighbors = min([len(X_min), self.k+1])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        # extracting standard deviations of distances
        stds = np.std(dist[:, 1:], axis=1)

        # estimating sampling density
        if np.sum(stds) > 0:
            p_i = stds/np.sum(stds)
        else:
            _logger.warning(self.__class__.__name__ +
                            ": " + "zero distribution")
            return X.copy(), y.copy()

        # the other component of sampling density
        p_ij = dist[:, 1:]/np.sum(dist[:, 1:], axis=1)[:, None]

        # number of samples to generate between minority points
        counts_ij = n_to_sample*p_i[:, None]*p_ij

        # do the sampling
        samples = []
        for i in range(len(p_i)):
            for j in range(min([len(X_min)-1, self.k])):
                while counts_ij[i][j] > 0:
                    if self.random_state.random_sample() < counts_ij[i][j]:
                        translation = X_min[ind[i][j+1]] - X_min[i]
                        weight = counts_ij[i][j] + 1
                        #samples.append(
                        #    X_min[i] + translation/counts_ij[i][j]+1)
                        samples.append(
                            X_min[i] + translation/weight)
                    counts_ij[i][j] = counts_ij[i][j] - 1

        if len(samples) > 0:
            return (np.vstack([X, np.vstack(samples)]),
                    np.hstack([y, np.repeat(self.min_label, len(samples))]))
        else:
            return X.copy(), y.copy()

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'k': self.k,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


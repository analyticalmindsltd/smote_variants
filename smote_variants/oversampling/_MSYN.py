import numpy as np

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['MSYN']

class MSYN(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{msyn,
                            author="Fan, Xiannian
                            and Tang, Ke
                            and Weise, Thomas",
                            editor="Huang, Joshua Zhexue
                            and Cao, Longbing
                            and Srivastava, Jaideep",
                            title="Margin-Based Over-Sampling Method for
                                    Learning from Imbalanced Datasets",
                            booktitle="Advances in Knowledge Discovery and
                                        Data Mining",
                            year="2011",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="309--320",
                            abstract="Learning from imbalanced datasets has
                                        drawn more and more attentions from
                                        both theoretical and practical aspects.
                                        Over- sampling is a popular and simple
                                        method for imbalanced learning. In this
                                        paper, we show that there is an
                                        inherently potential risk associated
                                        with the over-sampling algorithms in
                                        terms of the large margin principle.
                                        Then we propose a new synthetic over
                                        sampling method, named Margin-guided
                                        Synthetic Over-sampling (MSYN), to
                                        reduce this risk. The MSYN improves
                                        learning with respect to the data
                                        distributions guided by the
                                        margin-based rule. Empirical study
                                        verities the efficacy of MSYN.",
                            isbn="978-3-642-20847-8"
                            }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 pressure=1.5,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 n_jobs=1,
                 random_state=None,
                 proportion=None):
        """
        Constructor of the sampling object

        Args:
            pressure (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): number of neighbors in the SMOTE sampling
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
        self.check_greater_or_equal(pressure, 'pressure', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.pressure = proportion or pressure
        self.proportion = self.pressure
        self.n_neighbors = n_neighbors
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
        parameter_combinations = {'proportion': [2.5, 2.0, 1.5, 1.0],
                                  'n_neighbors': [3, 5, 7]}
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

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        min_indices = np.where(y == self.min_label)[0]
        maj_indices = np.where(y == self.maj_label)[0]

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # generating samples
        smote = SMOTE(proportion=self.pressure,
                      n_neighbors=self.n_neighbors,
                      nn_params=nn_params,
                      n_jobs=self.n_jobs,
                      random_state=self._random_state_init)

        X_res, y_res = smote.sample(X, y)
        X_new, _ = X_res[len(X):], y_res[len(X):]

        if len(X_new) == 0:
            m = "Sampling is not needed"
            _logger.warning(self.__class__.__name__ + ": " + m)
            return X.copy(), y.copy()

        # Compute nearest hit and miss for both classes
        nn= NearestNeighborsWithMetricTensor(n_neighbors=len(X), 
                                                n_jobs=self.n_jobs, 
                                                **nn_params)
        nn.fit(X)
        dist, ind = nn.kneighbors(X)

        # computing nearest hit and miss distances, these will be used to
        # compute thetas
        nearest_hit_dist = np.array([dist[i][next(j for j in range(
            1, len(X)) if y[i] == y[ind[i][j]])] for i in range(len(X))])
        nearest_miss_dist = np.array([dist[i][next(j for j in range(
            1, len(X)) if y[i] != y[ind[i][j]])] for i in range(len(X))])

        # computing the thetas without new samples being involved
        theta_A_sub_alpha = 0.5*(nearest_miss_dist - nearest_hit_dist)
        theta_min = theta_A_sub_alpha[min_indices]
        theta_maj = theta_A_sub_alpha[maj_indices]

        metric_tensor = nn_params['metric_tensor'] if nn_params.get('metric_tensor', None) is not None\
                                                    else np.eye(len(X[0]))

        # computing the f_3 score for all new samples
        f_3 = []
        for x in X_new:
            # determining the distances of the new sample from the training set
            #distances = np.linalg.norm(X - x, axis=1)
            distances= np.sqrt(np.einsum('ij,ij -> i', (X - x), np.dot(X - x, metric_tensor)))

            # computing nearest hit and miss distances involving the new
            # elements
            mask = nearest_hit_dist[min_indices] < distances[min_indices]
            nearest_hit_dist_min = np.where(mask,
                                            nearest_hit_dist[min_indices],
                                            distances[min_indices])
            nearest_miss_dist_min = nearest_miss_dist[min_indices]
            nearest_hit_dist_maj = nearest_hit_dist[maj_indices]
            mask = nearest_miss_dist[maj_indices] < distances[maj_indices]
            nearest_miss_dist_maj = np.where(mask,
                                             nearest_miss_dist[maj_indices],
                                             distances[maj_indices])

            # computing the thetas incorporating the new elements
            theta_x_min = 0.5*(nearest_miss_dist_min - nearest_hit_dist_min)
            theta_x_maj = 0.5*(nearest_miss_dist_maj - nearest_hit_dist_maj)

            # determining the delta scores and computing f_3
            Delta_P = np.sum(theta_x_min - theta_min)
            Delta_N = np.sum(theta_x_maj - theta_maj)

            f_3.append(-Delta_N/(Delta_P + 0.01))

        f_3 = np.array(f_3)

        # determining the elements with the minimum f_3 scores to add
        _, new_ind = zip(
            *sorted(zip(f_3, np.arange(len(f_3))), key=lambda x: x[0]))
        new_ind = list(new_ind[:(len(X_maj) - len(X_min))])

        return (np.vstack([X, X_new[new_ind]]),
                np.hstack([y, np.repeat(self.min_label, len(new_ind))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'pressure': self.pressure,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init,
                'proportion': self.proportion}

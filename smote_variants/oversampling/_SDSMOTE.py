import numpy as np

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['SDSMOTE']

class SDSMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{sdsmote,
                            author={Li, K. and Zhang, W. and Lu, Q. and
                                        Fang, X.},
                            booktitle={2014 International Conference on
                                        Identification, Information and
                                        Knowledge in the Internet of
                                        Things},
                            title={An Improved SMOTE Imbalanced Data
                                    Classification Method Based on Support
                                    Degree},
                            year={2014},
                            volume={},
                            number={},
                            pages={34-38},
                            keywords={data mining;pattern classification;
                                        sampling methods;improved SMOTE
                                        imbalanced data classification
                                        method;support degree;data mining;
                                        class distribution;imbalanced
                                        data-set classification;over sampling
                                        method;minority class sample
                                        generation;minority class sample
                                        selection;minority class boundary
                                        sample identification;Classification
                                        algorithms;Training;Bagging;Computers;
                                        Testing;Algorithm design and analysis;
                                        Data mining;Imbalanced data-sets;
                                        Classification;Boundary sample;Support
                                        degree;SMOTE},
                            doi={10.1109/IIKI.2014.14},
                            ISSN={},
                            month={Oct}}
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_borderline,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
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
            n_neighbors (int): number of neighbors in nearest neighbors
                                component
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
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
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
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
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

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        # fitting nearest neighbors model to find closest majority points to
        # minority samples
        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)
        
        nn = NearestNeighborsWithMetricTensor(n_neighbors=len(X_maj), 
                                                n_jobs=self.n_jobs, 
                                                **nn_params)
        nn.fit(X_maj)
        dist, ind = nn.kneighbors(X_min)

        # calculating the sum according to S3 in the paper
        S_i = np.sum(dist, axis=1)
        # calculating average distance according to S5
        S = np.sum(S_i)
        S_ave = S/(len(X_min)*len(X_maj))

        # calculate support degree
        def support_degree(x):
            return len(nn.radius_neighbors(x.reshape(1, -1),
                                           S_ave,
                                           return_distance=False))

        k = np.array([support_degree(X_min[i]) for i in range(len(X_min))])
        density = k/np.sum(k)

        # fitting nearest neighbors model to minority samples to run
        # SMOTE-like sampling
        n_neighbors = min([len(X_min), self.n_neighbors+1])
        nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **nn_params)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            idx = self.random_state.choice(np.arange(len(density)), p=density)
            random_neighbor_idx = self.random_state.choice(ind[idx][1:])
            X_a = X_min[idx]
            X_b = X_min[random_neighbor_idx]
            samples.append(self.sample_between_points(X_a, X_b))

        return (np.vstack([X, np.vstack(samples)]),
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
                'random_state': self._random_state_init}


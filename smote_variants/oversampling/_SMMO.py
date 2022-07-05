import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['SMMO']

class SMMO(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{smmo,
                            author = {de la Calleja, Jorge and Fuentes, Olac
                                        and González, Jesús},
                            booktitle= {Proceedings of the Twenty-First
                                        International Florida Artificial
                                        Intelligence Research Society
                                        Conference},
                            year = {2008},
                            month = {01},
                            pages = {276-281},
                            title = {Selecting Minority Examples from
                                    Misclassified Data for Over-Sampling.}
                            }

    Notes:
        * In this paper the ensemble is not specified. I have selected
            some very fast, basic classifiers.
        * Also, it is not clear what the authors mean by "weighted distance".
        * The original technique is not prepared for the case when no minority
            samples are classified correctly be the ensemble.
    """

    categories = [OverSampling.cat_borderline,
                  OverSampling.cat_extensive,
                  OverSampling.cat_uses_classifier,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 ensemble=[QuadraticDiscriminantAnalysis(),
                           DecisionTreeClassifier(random_state=2),
                           GaussianNB()],
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
            n_neighbors (int): control parameter of the nearest neighbor
                                component
            ensemble (list): list of classifiers, if None, default list of
                                classifiers is used
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_jobs (int): number of parallel jobs
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        try:
            len_ens = len(ensemble)
        except Exception as e:
            raise ValueError('The ensemble needs to be a list-like object')
        if len_ens == 0:
            raise ValueError('At least 1 classifier needs to be specified')
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.ensemble = ensemble
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
        ensembles = [[QuadraticDiscriminantAnalysis(),
                      DecisionTreeClassifier(random_state=2),
                      GaussianNB()]]
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'ensemble': ensembles}

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

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # training and in-sample prediction (out-of-sample by k-fold cross
        # validation might be better)
        predictions = []
        for e in self.ensemble:
            predictions.append(e.fit(X, y).predict(X))

        # constructing ensemble prediction
        pred = np.where(np.mean(np.vstack(predictions), axis=0)
                        > 0.5, 1, 0)

        # create mask of minority samples to sample
        mask_to_sample = np.where(np.logical_and(np.logical_not(
            np.equal(pred, y)), y == self.min_label))[0]
        if len(mask_to_sample) < 2:
            m = "Not enough minority samples selected %d" % len(mask_to_sample)
            _logger.warning(self.__class__.__name__ + ": " + m)
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]
        X_min_to_sample = X[mask_to_sample]

        # fitting nearest neighbors model for sampling
        n_neighbors = min([len(X_min), self.n_neighbors + 1])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_min)
        ind = nn.kneighbors(X_min_to_sample, return_distance=False)

        # doing the sampling
        samples = []
        while len(samples) < n_to_sample:
            idx = self.random_state.randint(len(X_min_to_sample))
            mean = np.mean(X_min[ind[idx][1:]], axis=0)
            samples.append(self.sample_between_points(
                X_min_to_sample[idx], mean))

        return (np.vstack([X, np.vstack([samples])]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'ensemble': self.ensemble,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

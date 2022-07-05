import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import pairwise_distances

from .._metric_tensor import (NearestNeighborsWithMetricTensor, 
                                MetricTensor, pairwise_distances_mahalanobis)
from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

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
                 nn_params={},
                 n_jobs=1, 
                 random_state=None):
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
        super().__init__()

        self.check_n_jobs(n_jobs, 'n_jobs')

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
        return [{}]

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

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        n_1 = len(X_min)
        n_0 = len(X) - n_1
        N = int(np.rint(0.5*n_0 - 0.5*n_1 + 0.5))

        if N == 0:
            return X.copy(), y.copy()

        # generating minority samples
        samples = []

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nn= NearestNeighborsWithMetricTensor(n_neighbors=1, 
                                                n_jobs=self.n_jobs, 
                                                **nn_params)
        nn.fit(X_maj)

        stds = np.std(X_min, axis=0)
        # At one point the algorithm says to keep those points which are
        # the most distant from majority samples, and not leaving any minority
        # sample isolated. This can be implemented by generating multiple
        # samples for each point and keep the one most distant from the
        # majority samples.
        for _ in range(N):
            i = self.random_state.randint(len(X_min))
            best_sample = None
            best_sample_dist = 0
            for _ in range(3):
                s = self.random_state.normal(X_min[i], stds)
                dist, ind = nn.kneighbors(s.reshape(1, -1))
                if dist[0][0] > best_sample_dist:
                    best_sample_dist = dist[0][0]
                    best_sample = s
            samples.append(best_sample)

        # Extending the minority dataset with the new samples
        X_min_extended = np.vstack([X_min, np.vstack(samples)])

        # Removing N elements from the majority dataset

        # normalize
        mms = MinMaxScaler()
        X_maj_normalized = mms.fit_transform(X_maj)

        # computing the distance matrix
        dm = pairwise_distances_mahalanobis(X_maj_normalized,
                                             X_maj_normalized, 
                                             nn_params.get('metric_tensor', None))

        # len(X_maj) offsets for the diagonal 0 elements, 2N because
        # every distances appears twice
        threshold = sorted(dm.flatten())[min(
            [len(X_maj) + 2*N, len(dm)*len(dm) - 1])]
        for i in range(len(dm)):
            dm[i, i] = np.inf

        # extracting the coordinates of pairs closer than threshold
        pairs_to_break = np.where(dm < threshold)
        pairs_to_break = np.vstack(pairs_to_break)

        # sorting the pairs, otherwise both points would be removed
        pairs_to_break.sort(axis=0)

        # uniqueing the coordinates - the final number might be less than N
        to_remove = np.unique(pairs_to_break[0])

        # removing the selected elements
        X_maj_cleaned = np.delete(X_maj, to_remove, axis=0)

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
                'random_state': self._random_state_init}


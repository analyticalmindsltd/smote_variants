from random import random
import numpy as np
from sklearn.ensemble import IsolationForest

from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['SMOTE_AMSR']

class SMOTE_AMSR(OverSampling):

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_extensive,
                  OverSampling.cat_metric_learning]

    # @_deprecate_positional_args
    def __init__(self,
                 proportion=1.0,
                 *,
                 topology='mesh',
                 random_state=None):

        super().__init__()

        self.check_greater_or_equal(proportion, "proportion", 0.0)
        self.check_isin(topology, "topology", ['star', 'bus', 'mesh'])

        self.proportion = proportion
        self.topology = topology
        self.randomsate = random_state

        self.set_random_state(random_state)

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.
        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'topology': ['bus', 'star', 'mesh'],}

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def _make_samples(
        self, X, y_type, nn_data, n_samples, method, scores, clf_anomaly
    ):
        samples = []
        if method=='star':
            X_mean = np.mean(nn_data, axis=0)
            mean_amomaly = clf_anomaly.decision_function(X_mean.reshape(1,-1))[0]
            k = max([1, int(np.rint(n_samples/len(X)))])
            MaxMin = max(scores) - min(scores)
            for ix in range(len(X)):
                diff = X_mean - X[ix]
                if mean_amomaly>=scores[ix]:
                    d = mean_amomaly-scores[ix] / MaxMin
                    step = 1.0 * self.random_state.uniform(low=0+d, high=1.0, size=(k, len(diff)))
                else:
                    d = scores[ix] - mean_amomaly / MaxMin
                    step = 1.0 * self.random_state.uniform(low=0, high=1.0-d, size=(k, len(diff)))
                for ii in range(k):
                    samples.append(X[ix] + step[ii,:].ravel()*diff)

        elif method=='bus':
            # Implementation of the bus topology
            k = max([1, int(np.rint(n_samples/len(X)))])
            MaxMin = max(scores) - min(scores)
            for ix in range(1, len(X)):
                diff = X[ix-1] - X[ix]
                if scores[ix-1]>=scores[ix]:
                    d = (scores[ix-1]-scores[ix]) / MaxMin
                    step = 1.0 * self.random_state.uniform(low=0+d, high=1.0, size=(k, len(diff)))
                else:
                    d = (scores[ix]-scores[ix-1]) / MaxMin
                    step = 1.0 * self.random_state.uniform(low=0, high=1.0-d, size=(k, len(diff)))
                for ii in range(k):
                    samples.append(X[ix] + step[ii,:].ravel()*diff)

        elif method=='mesh':
            # Implementation of the mesh topology
            Nx = X.shape[0]
            if Nx < 3:
                n_combs = Nx
            else:
                n_combs = Nx + 0.5*(Nx*(Nx-3))
            k = max([1, int(np.rint(n_samples/n_combs))])
            if k > 1:
                MaxMin = max(scores) - min(scores)
                for i in range(Nx):
                    for j in range(i+1, Nx):
                        diff = X[i] - X[j]
                        if scores[i]>=scores[j]:
                            d = (scores[i]-scores[j]) / MaxMin
                            step = 1.0 * self.random_state.uniform(low=0+d, high=1.0, size=(k, len(diff)))
                        else:
                            d = (scores[j]-scores[i]) / MaxMin
                            step = 1.0 * self.random_state.uniform(low=0, high=1.0-d, size=(k, len(diff)))
                        for ii in range(k):
                            samples.append(X[j] + step[ii,:].ravel()*diff)
            else:
                while len(samples) < n_samples:
                    random_i = self.random_state.randint(len(X))
                    random_j = self.random_state.randint(len(X))
                    diff = X[random_i] - X[random_j]
                    MaxMin = max(scores) - min(scores)
                    if scores[random_i]>=scores[random_j]:
                        d = (scores[random_i]-scores[random_j]) / MaxMin
                        step = 1.0 * self.random_state.uniform(low=0+d, high=1.0, size=(k, len(diff)))
                    else:
                        d = (scores[random_j]-scores[random_i]) / MaxMin
                        step = 1.0 * self.random_state.uniform(low=0, high=1.0-d, size=(k, len(diff)))
                    for ii in range(k):
                        samples.append(X[random_j] + step[ii,:].ravel()*diff)

        return np.vstack(samples), np.repeat(y_type, len(samples))

    def _in_danger_iforest(self, X, class_sample, y):

        clf = IsolationForest(n_estimators=100, random_state=self.randomsate)
        clf.fit(X)

        X_scores = clf.decision_function(X)
        X_class_scores = X_scores[y==class_sample]

        return X, X_class_scores, clf

    def sample(self, X, y):
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        class_sample = self.min_label
        X_class = X[y == class_sample]


        n_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])
        if n_sample == 0:
            _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X, y
        
        _, mda_scores, clf_anomaly = self._in_danger_iforest(
            X, class_sample, y
        )

        X_new, y_new = self._make_samples(
            X_class,
            class_sample,
            X_class,
            n_sample,
            self.topology,
            mda_scores,
            clf_anomaly,
        )

        return (np.vstack((X, X_new)), np.hstack((y, y_new)))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'topology': self.topology,
                'random_state': self._random_state_init}

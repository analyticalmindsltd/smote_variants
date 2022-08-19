"""
This module implements the NRAS method.
"""

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

from ..base import coalesce, coalesce_dict
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['NRAS']

class NRAS(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @article{nras,
                        title = "Noise Reduction A Priori Synthetic
                                    Over-Sampling for class imbalanced data
                                    sets",
                        journal = "Information Sciences",
                        volume = "408",
                        pages = "146 - 161",
                        year = "2017",
                        issn = "0020-0255",
                        doi = "https://doi.org/10.1016/j.ins.2017.04.046",
                        author = "William A. Rivera",
                        keywords = "NRAS, SMOTE, OUPS, Class imbalance,
                                        Classification"
                        }
    """

    categories = [OverSamplingSimplex.cat_sample_ordinary,
                  OverSamplingSimplex.cat_noise_removal,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 t=0.5,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): number of neighbors
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
            t (float): [0,1] fraction of n_neighbors as threshold
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
        self.check_in_range(t, "t", [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
        self.t = t # pylint: disable=invalid-name
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
                                  'n_neighbors': [5, 7, 9],
                                  't': [0.3, 0.5, 0.8]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def propensity_scores(self,
                            X_trans, # pylint: disable=invalid-name
                            y):
        """
        Determining the propensity scores.

        Args:
            X_trans (np.array): normalized training vectors
            y (np.array): the target labels

        Returns:
            np.array: the propensity scores
        """
        # determining propensity scores using logistic regression
        logreg = LogisticRegression(solver='lbfgs',
                                n_jobs=self.n_jobs,
                                random_state=self._random_state_init)
        logreg.fit(X_trans, y)
        propensity = logreg.predict_proba(X_trans)
        propensity = propensity[:, np.where(logreg.classes_ == self.min_label)[0][0]]

        return propensity

    def neighborhood_structure(self,
                                X_new,
                                y,
                                n_neighbors,
                                X_min_new # pylint: disable=invalid-name
                                ):
        """
        Determine neighborhood structure.

        Args:
            X_new (np.array): the extended training vectors
            y (np.array): the target labels
            n_neighbors (int): the number of neighbors
            X_min_new (np.array): the minority samples

        Returns:
            np.array: the neighborhood structure
        """
        nn_params = {**self.nn_params}
        nn_params['metric_tensor'] = \
                    self.metric_tensor_from_nn_params(nn_params, X_new, y)

        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X_new)
        ind = nnmt.kneighbors(X_min_new, return_distance=False)

        return ind

    def generate_samples(self, *, X_min, to_remove, X_trans, y,
                                                    ind, n_to_sample):
        """
        Generate samples

        Args:
            X_min (np.array): the minority samples
            to_remove (np.array): minority indices flagged as noise
            X_trans (np.array): all training samples
            y (np.array): all target labels
            ind (np.array): the neighborhood structure
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array: the generated samples
        """
        base_weights = np.repeat(1.0, len(X_min))
        base_weights[to_remove] = 0.0
        vertex_weights = np.repeat(1.0, len(X_trans))
        vertex_weights[y == self.min_label][to_remove] = 0.0

        samples = self.sample_simplex(X=X_min,
                                        indices=ind,
                                        n_to_sample=n_to_sample,
                                        X_vertices=X_trans,
                                        vertex_weights=vertex_weights,
                                        base_weights=base_weights)
        return samples

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

        # standardization is needed to make the range of the propensity scores
        # similar to that of the features
        mms = MinMaxScaler()
        X_trans = mms.fit_transform(X) # pylint: disable=invalid-name

        X_min = X_trans[y == self.min_label]

        # adding propensity scores as a new feature
        X_new = np.column_stack([X_trans, self.propensity_scores(X_trans, y)])
        X_min_new = X_new[y == self.min_label] # pylint: disable=invalid-name

        # finding nearest neighbors of minority samples
        n_neighbors = min([len(X_new), self.n_neighbors+1])

        ind = self.neighborhood_structure(X_new, y, n_neighbors, X_min_new)

        # noise removal
        t_hat = np.sum(y[ind[:, 1:]] == self.min_label, axis=1)
        to_remove = np.where(t_hat < self.t * n_neighbors)[0]

        if len(to_remove) >= len(X_min) - 1:
            return self.return_copies(X, y,
                            "most minority samples indentified as noise")

        n_to_sample = n_to_sample + to_remove.shape[0]

        samples = self.generate_samples(X_min=X_min,
                                        to_remove=to_remove,
                                        X_trans=X_trans,
                                        y=y,
                                        ind=ind,
                                        n_to_sample=n_to_sample)

        X_min = np.delete(X_min, to_remove, axis=0)

        # do the sampling
        #samples = []
        #while len(samples) < n_to_sample:
        #    idx = self.random_state.randint(len(X_min))
        #    # finding the number of minority neighbors
        #    t_hat = np.sum(y[ind[idx][1:]] == self.min_label)
        #    if t_hat < self.t*n_neighbors:
        #        # removing the minority point if the number of minority
        #        # neighbors is less then the threshold
        #        # to_remove indexes X_min
        #        if idx not in to_remove:
        #            to_remove.append(idx)
        #            # compensating the removal of the minority point
        #            n_to_sample = n_to_sample + 1
        #
        #        if len(to_remove) == len(X_min):
        #            _logger.warning(self.__class__.__name__ + ": " +
        #                            "all minority samples identified as noise")
        #            return X.copy(), y.copy()
        #    else:
        #        # otherwise do the sampling
        #        X_b = X_trans[self.random_state.choice(ind[idx][1:])]
        #        samples.append(self.sample_between_points(X_min[idx], X_b))

        return (mms.inverse_transform(np.vstack([X_trans[y == self.maj_label],
                                                 X_min,
                                                 samples])),
                np.hstack([np.repeat(self.maj_label,
                                                np.sum(y == self.maj_label)),
                           np.repeat(self.min_label, len(X_min)),
                           np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                't': self.t,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}

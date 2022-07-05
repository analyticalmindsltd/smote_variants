import numpy as np

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from scipy.stats.mstats import gmean

from .._metric_tensor import (NearestNeighborsWithMetricTensor, 
                                MetricTensor, pairwise_distances_mahalanobis)
from ._OverSampling import OverSampling

from .._logger import logger
_logger= logger

__all__= ['DSMOTE']

class DSMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{dsmote,
                            author={Mahmoudi, S. and Moradi, P. and Akhlaghian,
                                    F. and Moradi, R.},
                            booktitle={2014 4th International Conference on
                                        Computer and Knowledge Engineering
                                        (ICCKE)},
                            title={Diversity and separable metrics in
                                    over-sampling technique for imbalanced
                                    data classification},
                            year={2014},
                            volume={},
                            number={},
                            pages={152-158},
                            keywords={learning (artificial intelligence);
                                        pattern classification;sampling
                                        methods;diversity metric;separable
                                        metric;over-sampling technique;
                                        imbalanced data classification;
                                        class distribution techniques;
                                        under-sampling technique;DSMOTE method;
                                        imbalanced learning problem;diversity
                                        measure;separable measure;Iran
                                        University of Medical Science;UCI
                                        dataset;Accuracy;Classification
                                        algorithms;Vectors;Educational
                                        institutions;Euclidean distance;
                                        Data mining;Diversity measure;
                                        Separable Measure;Over-Sampling;
                                        Imbalanced Data;Classification
                                        problems},
                            doi={10.1109/ICCKE.2014.6993409},
                            ISSN={},
                            month={Oct}}

    Notes:
        * The method is highly inefficient when the number of minority samples
            is high, time complexity is O(n^3), with 1000 minority samples it
            takes about 1e9 objective function evaluations to find 1 new sample
            points. Adding 1000 samples would take about 1e12 evaluations of
            the objective function, which is unfeasible. We introduce a new
            parameter, n_step, and during the search for the new sample at
            most n_step combinations of minority samples are tried.
        * Abnormality of minority points is defined in the paper as
            D_maj/D_min, high abnormality  means that the minority point is
            close to other minority points and very far from majority points.
            This is definitely not abnormality,
            I have implemented the opposite.
        * Nothing ensures that the fisher statistics and the variance from
            the geometric mean remain comparable, which might skew the
            optimization towards one of the sub-objectives.
        * MinMax normalization doesn't work, each attribute will have a 0
            value, which will make the geometric mean of all attribute 0.
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 rate=0.1,
                 n_step=50,
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
            rate (float): [0,1] rate of minority samples to turn into majority
            n_step (int): number of random configurations to check for new
                                samples
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_in_range(rate, "rate", [0, 1])
        self.check_greater_or_equal(n_step, "n_step", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.rate = rate
        self.n_step = n_step
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
                                  'n_neighbors': [3, 5, 7],
                                  'rate': [0.1, 0.2],
                                  'n_step': [50]}
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

        if not self.check_enough_min_samples_for_sampling(3):
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        mms = MinMaxScaler(feature_range=(1e-6, 1.0 - 1e-6))
        X = mms.fit_transform(X)

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # fitting nearest neighbors model
        nn = NearestNeighborsWithMetricTensor(n_neighbors=len(X_maj), 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_maj)
        dist, ind = nn.kneighbors(X_min)

        # compute mean distances, the D_min is compenstaed for taking into
        # consideration self-distances in the mean
        D_maj = np.mean(dist, axis=1)
        D_min = np.mean(pairwise_distances_mahalanobis(X_min, tensor=nn_params.get('metric_tensor', None)), axis=1) * \
            len(X_min)/(len(X_min)-1)

        # computing degree of abnormality
        abnormality = D_min/D_maj

        # sorting minority indices in decreasing order by abnormality
        to_sort = zip(abnormality, np.arange(len(abnormality)))
        abnormality, indices = zip(*sorted(to_sort, key=lambda x: -x[0]))
        rate = int(self.rate*len(abnormality))

        if rate > 0:
            # moving the most abnormal points to the majority class
            X_maj = np.vstack([X_maj, X_min[np.array(indices[:rate])]])
            # removing the most abnormal points form the minority class
            X_min = np.delete(X_min, indices[:rate], axis=0)

        # computing the mean and variance of points in the majority class
        var_maj = np.mean(np.var(X_maj, axis=0))
        mean_maj = np.mean(X_maj)

        # this is the original objective function, however, using this
        # is very inefficient if the number of records increases above
        # approximately 1000
        # def objective(X):
        #    """
        #    The objective function to be maximized
        #
        #    Args:
        #        X (np.matrix): dataset
        #
        #    Returns:
        #        float: the value of the objective function
        #    """
        #    gm= gmean(X, axis= 0)
        #    gdiv= np.mean(np.linalg.norm(X - gm, axis= 1))
        #    fisher= (np.mean(X) - mean_maj)**2/(np.mean(np.var(X, axis= 0)) \
        #                + var_maj)
        #    return gdiv + fisher

        # in order to make the code more efficient, we do maintain some
        # variables containing the main componentes of the objective function
        # and apply only small corrections based on the new values being added
        # the effect should be identical

        # records the sum of logarithms in X_min, used to compute the geometric
        # mean
        min_log_sum = np.sum(np.log(X_min), axis=0)
        # contains the sum of values in X_min, coordinatewise
        min_sum = np.sum(X_min, axis=0)
        # contains the squares of sums of values in X_min, coordinatewise
        min_sum2 = np.sum(X_min**2, axis=0)
        # contains the sum of all numbers in X_min
        min_all_sum = np.sum(X_min)

        min_norm = np.linalg.norm(X_min)**2

        # do the sampling
        n_added = 0
        while n_added < n_to_sample:
            best_candidate = None
            highest_score = 0.0
            # we try n_step combinations of minority samples
            len_X = len(X_min)
            n_steps = min([len_X*(len_X-1)*(len_X-2), self.n_step])
            for _ in range(n_steps):
                i, j, k = self.random_state.choice(np.arange(len_X),
                                                   3,
                                                   replace=False)
                gm = gmean(X_min[np.array([i, j, k])], axis=0)

                # computing the new objective function for the new point (gm)
                #  added
                new_X_min = np.vstack([X_min, gm])

                # updating the components of the objective function
                new_min_log_sum = min_log_sum + np.log(gm)
                new_min_sum = min_sum + gm
                new_min_sum2 = min_sum2 + gm**2
                new_min_all_sum = min_all_sum + np.sum(gm)

                # computing mean, var, gmean and mean of all elements with
                # the new sample (gm)
                new_min_mean = new_min_sum/(len(new_X_min))
                new_min_var = new_min_sum2/(len(new_X_min)) - new_min_mean**2
                new_min_gmean = np.exp(new_min_log_sum/(len(new_X_min)))
                new_min_all_n = (len(new_X_min))*len(X_min[0])
                new_min_all_mean = new_min_all_sum / new_min_all_n

                new_min_norm = min_norm + np.linalg.norm(gm)

                # computing the new objective function value
                inner_prod = np.dot(new_X_min, new_min_gmean)
                gmean_norm = np.linalg.norm(new_min_gmean)**2
                term_sum = new_min_norm - 2*inner_prod + gmean_norm
                new_gdiv = np.mean(np.sqrt(term_sum))

                fisher_numerator = (new_min_all_mean - mean_maj)**2
                fisher_denominator = np.mean(new_min_var) + var_maj
                new_fisher = fisher_numerator / fisher_denominator

                score = new_gdiv + new_fisher

                # evaluate the objective function
                # score= objective(np.vstack([X_min, gm]))
                # check if the score is better than the best so far
                if score > highest_score:
                    highest_score = score
                    best_candidate = gm
                    cand_min_log_sum = new_min_log_sum
                    cand_min_sum = new_min_sum
                    cand_min_sum2 = new_min_sum2
                    cand_min_all_sum = new_min_all_sum
                    cand_min_norm = new_min_norm

            # add the best candidate to the minority samples
            X_min = np.vstack([X_min, best_candidate])
            n_added = n_added + 1

            min_log_sum = cand_min_log_sum
            min_sum = cand_min_sum
            min_sum2 = cand_min_sum2
            min_all_sum = cand_min_all_sum
            min_norm = cand_min_norm

        return (mms.inverse_transform(np.vstack([X_maj, X_min])),
                np.hstack([np.repeat(self.maj_label, len(X_maj)),
                           np.repeat(self.min_label, len(X_min))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'rate': self.rate,
                'n_step': self.n_step,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


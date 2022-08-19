"""
This module implements the DSMOTE method.
"""
from dataclasses import dataclass

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from scipy.stats.mstats import gmean

from ..base import (NearestNeighborsWithMetricTensor,
                                pairwise_distances_mahalanobis)
from ..base import OverSampling

from .._logger import logger
_logger= logger

__all__= ['DSMOTE']

@dataclass
class DSMOTEPartialResults:
    """
    This class is used to pass around some partial calculation results
    reused at various parts of the algorithm.

    Args:
        log_sum (np.array): the array of sums of logarithms of minority
                                data coordinate-wise
        sum1 (np.array): the array of the sums of minority coordinates
        sum2 (np.array): the array of the sums of squares of minority coordinates
        all_sum (float): the sum of all minority coordinate values
        norm (np.array): the norms of all minority samples
    """
    log_sum: np.array
    sum1: np.array
    sum2: np.array
    all_sum: float
    norm: np.array

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
                 random_state=None,
                 **_kwargs):
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
        super().__init__(random_state=random_state)
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

    def calculate_gdiv(self,
                        X_min_new, # pylint: disable=invalid-name
                        gmean_min,
                        partial):
        """
        Calculate the g-divergence score.

        Args:
            X_min_new (np.array): the new minority vectors
            gmean_min (np.array): the geometric mean of the minority points
            partial (DSMOTEPartialResults): partial results

        Returns:
            float: the g-divergence
        """
        inner_prod = np.dot(X_min_new, gmean_min)
        gmean_norm = np.linalg.norm(gmean_min)**2
        term_sum = partial.norm - 2*inner_prod + gmean_norm
        gdiv = np.mean(np.sqrt(term_sum))

        return gdiv

    def calculate_fisher(self, all_mean_min, mean_maj, var_min, var_maj):
        """
        Calculate the Fisher ratio

        Args:
            all_mean_min (float): mean of all minority vectors
            mean_maj (float): mean of the majority vectors
            var_min (float): mean coordinate-wise variance of all
                                minority vectors
            var_maj (float): mean coordinate-wise variances of
                                majority vectors

        Returns:
            float: the Fisher ratio
        """
        fisher_numerator = (all_mean_min - mean_maj)**2
        fisher_denominator = np.mean(var_min) + var_maj
        fisher = fisher_numerator / fisher_denominator
        return fisher

    def calculate_objective(self,
                            X_min_new, # pylint: disable=invalid-name
                            partial,
                            mean_maj,
                            var_maj):
        """
        Calculate the objective function.

        Args:
            X_min_new (np.array): new minority samples
            partial (DSMOTEPartialResults): partial results
            mean_maj (float): mean of the majority vectors
            var_maj (float): mean coordinate-wise variance of the
                                majority vectors

        Returns:
            float: the objective function value
        """
        # computing mean, var, gmean and mean of all elements with
        # the new sample (gm)
        mean_min = partial.sum1 / X_min_new.shape[0]
        var_min = partial.sum2 / X_min_new.shape[0] - mean_min**2
        gmean_min = np.exp(partial.log_sum / X_min_new.shape[0])
        all_n_min = np.prod(X_min_new.shape)
        all_mean_min = partial.all_sum / all_n_min

        # computing the new objective function value
        score = self.calculate_gdiv(X_min_new, gmean_min, partial) \
                + self.calculate_fisher(all_mean_min, mean_maj, var_min, var_maj)

        return score

    def generate_one_sample(self, X_min, mean_maj, var_maj, partial_results):
        """
        Generates one new sample

        Args:
            X_min (np.array): minority vectors
            mean_maj (float): mean of the majority vectors
            var_maj (float): mean coordinate-wise variance of the
                                majority vectors
            partial_results (DSMOTEPartialResults): partial calculation results

        Returns:
            np.array, DSMOTEPartialResults: the new vector and the
                                            partial results
        """
        best_candidate = None
        new_partial_results = {}
        highest_score = 0.0
        # we try n_step combinations of minority samples
        n_steps = np.min([X_min.shape[0] \
                        * (X_min.shape[0]-1) \
                        * (X_min.shape[0]-2), self.n_step])

        for _ in range(n_steps):
            indices = self.random_state.choice(np.arange(X_min.shape[0]),
                                                        3,
                                                        replace=False)
            gmv = gmean(X_min[indices], axis=0)

            # computing the new objective function for the new point (gm)
            #  added
            X_min_new = np.vstack([X_min, gmv]) # pylint: disable=invalid-name

            # updating the components of the objective function
            partial_tmp = DSMOTEPartialResults(
                            log_sum=partial_results.log_sum + np.log(gmv),
                            sum1=partial_results.sum1 + gmv,
                            sum2=partial_results.sum2 + gmv**2,
                            all_sum=partial_results.all_sum + np.sum(gmv),
                            norm=partial_results.norm + np.linalg.norm(gmv))

            score = self.calculate_objective(X_min_new, partial_tmp, mean_maj, var_maj)
            # evaluate the objective function
            # score= objective(np.vstack([X_min, gm]))
            # check if the score is better than the best so far
            if score > highest_score:
                highest_score = score
                best_candidate = gmv
                new_partial_results = partial_tmp

        return best_candidate, new_partial_results


    def sampling(self, X_maj, X_min, n_to_sample):
        """
        Carry out the sampling

        Args:
            X_maj (np.array): the majority samples
            X_min (np.array): the minority samples
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array: the updated minority samples
        """
        # computing the mean and variance of points in the majority class
        var_maj = np.mean(np.var(X_maj, axis=0))
        mean_maj = np.mean(X_maj)

        # records the sum of logarithms in X_min, used to compute the geometric
        # mean
        log_sum = np.sum(np.log(X_min), axis=0)
        # contains the sum of values in X_min, coordinatewise
        sum1 = np.sum(X_min, axis=0)
        # contains the squares of sums of values in X_min, coordinatewise
        sum2 = np.sum(X_min**2, axis=0)
        # contains the sum of all numbers in X_min
        all_sum = np.sum(X_min)

        norm = np.linalg.norm(X_min)**2

        partial_results = DSMOTEPartialResults(log_sum=log_sum,
                                                sum1=sum1,
                                                sum2=sum2,
                                                all_sum=all_sum,
                                                norm=norm)

        # do the sampling
        n_added = 0
        while n_added < n_to_sample:
            best_candidate, partial_results = self.generate_one_sample(X_min,
                                                                mean_maj,
                                                                var_maj,
                                                                partial_results)
            # add the best candidate to the minority samples
            X_min = np.vstack([X_min, best_candidate])
            n_added = n_added + 1

        return X_min

    def calculate_abnormality(self, X, y, X_min, X_maj):
        """
        Calculate the abnormality scores.

        Args:
            X (np.array): all features
            y (np.array): all labels
            X_min (np.array): minority samples
            X_maj (np.array): majority samples

        Returns:
            np.array: the abnormality scores
        """
        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # fitting nearest neighbors model
        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=len(X_maj),
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X_maj)
        dist, _ = nnmt.kneighbors(X_min)

        # compute mean distances, the D_min is compenstaed for taking into
        # consideration self-distances in the mean
        D_maj = np.mean(dist, axis=1) # pylint: disable=invalid-name
        D_min = np.mean(pairwise_distances_mahalanobis(X_min, # pylint: disable=invalid-name
                                tensor=nn_params.get('metric_tensor', None)), axis=1)
        D_min = D_min * len(X_min)/(len(X_min)-1) # pylint: disable=invalid-name

        # computing degree of abnormality
        abnormality = D_min/D_maj

        return abnormality

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

        mms = MinMaxScaler(feature_range=(1e-6, 1.0 - 1e-6))
        X = mms.fit_transform(X)

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        abnormality = self.calculate_abnormality(X, y, X_min, X_maj)

        # sorting minority indices in decreasing order by abnormality
        to_sort = zip(abnormality, np.arange(len(abnormality)))
        abnormality, indices = zip(*sorted(to_sort, key=lambda x: -x[0]))
        rate = int(self.rate*len(abnormality))

        if rate > 0:
            # moving the most abnormal points to the majority class
            X_maj = np.vstack([X_maj, X_min[np.array(indices[:rate])]])
            # removing the most abnormal points form the minority class
            X_min = np.delete(X_min, indices[:rate], axis=0)

        # this is the original objective function, however, using this
        # is very inefficient if the number of records increases above
        # approximately 1000
        # def objective(X):
        #    """
        #    The objective function to be maximized
        #
        #    Args:
        #        X (np.array): dataset
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

        X_min = self.sampling(X_maj, X_min, n_to_sample)

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
                **OverSampling.get_params(self)}

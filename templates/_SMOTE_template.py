"""
This module implements the SMOTE method.
"""
import logging

import numpy as np

from smote_variants.base import OverSampling

_logger = logging.getLogger('smote_variants')

__all__= ['SMOTE_template']

class SMOTE_template(OverSampling):
    """
    TODO: Add a description or a BibTex reference in the same format below

    References:
        * BibTex::

            @INPROCEEDINGS{key,
                            author={},
                            booktitle={},
                            title={},
                            year={},
                            volume={},
                            number={},
                            pages={},
                            doi={}}

    """

    # TODO: keep those categories which apply to the implemented technique
    categories = [OverSampling.cat_noise_removal,          # applies noise removal
                    OverSampling.cat_dim_reduction,        # applies dimensionality reduction
                    OverSampling.cat_uses_classifier,      # uses some advanced classifier
                    OverSampling.cat_sample_componentwise, # sampling is done coordinate-wise
                    OverSampling.cat_sample_ordinary,      # sampling is done in the SMOTE scheme
                    OverSampling.cat_sample_copy,          # sampling is done by replication
                    OverSampling.cat_memetic,              # applies some evoluationary optimization
                    OverSampling.cat_density_estimation,   # based on kernel density estimation
                    OverSampling.cat_density_based,        # estimates a density where to sample
                    OverSampling.cat_extensive,            # adds minority samples only
                    OverSampling.cat_changes_majority,     # changes the majority samples (e.g. removes some)
                    OverSampling.cat_uses_clustering,      # applies some sort of clustering
                    OverSampling.cat_borderline,           # uses the concept of borderline
                    OverSampling.cat_application,          # specific to some sort of special data
                    OverSampling.cat_metric_learning]      # metric learning is applicable (uses nearest neighbors)

    def __init__(self,
                 proportion=1.0,
                 dummy_parameter_0=1,
                 *,
                 dummy_parameter_1='dummy0',
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the SMOTE template object

        TODO: List all arguments below

        Args:
             proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            dummy_parameter_0 (int): dummy parameter 0
            dummy_parameter_1 (str): dummy parameter 1 ("dummy0"/"dummy1")
            random_state (None/int/np.random.RandomState): the random state
            _kwargs: for technical reasons and to facilitate serialization, additional
                     keyword arguments are accepted
        """

        OverSampling.__init__(self, random_state=random_state)

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(dummy_parameter_0, "dummy_parameter_0", 1)
        self.check_isin(dummy_parameter_1, "dummy_parameter_1", ["dummy0", "dummy1"])

        self.proportion = proportion
        self.dummy_parameter_0 = dummy_parameter_0
        self.dummy_parameter_1 = dummy_parameter_1

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        TODO: Add a list of reasonable values to all parameters (except random_state)

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'dummy_parameter_0': [1, 2, 3],
                                  'dummy_parameter_1': ['dummy0', 'dummy1']}

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        TODO: implement the sampling algorithm

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        # use logging in the below format
        message = "dummy message"
        _logger.info(f"{self.__class__.__name__}: {message}")

        # additional object members available at this point:
        #
        # self.min_label: the label of the minority class
        # self.maj_label: the label of the majority class
        #
        # to generate random numbers use the np.random.RandomState object
        # available as
        #
        # self.random_state.*
        #
        # for example, generate a random uniform number in [0, 1] by
        #
        # r = self.random_state.random_sample()

        # this very simple oversampling algorithm adds zero vectors
        #
        # reimplement this step with any algorithm

        X_samples = np.zeros(shape=(n_to_sample, X.shape[1]))

        return (np.vstack([X, X_samples]),
                np.hstack([y, np.hstack([self.min_label]*n_to_sample)]))

    def preprocessing_transform(self, X):
        """
        Transforms new data according to the possible transformation
        implemented by the function "sample".

        TODO: implement this if the generated samples are not in the domain of
                the input data. I.e. if the generated samples are scaled in some way,
                implement the same scaling here that can be applied as a preprocessing
                step to transform new X feature vectors accordingly. Similarly,
                if there is a dimensionality reduction, i.e. the dimensionality of the
                generated samples is less than that of the input samples, implement
                the same transformation in this function.

                In most cases this function is useless as the domain of the samples
                remains the same as that of the input feature vectors.

        Args:
            X (np.array): features

        Returns:
            np.array: transformed features
        """
        return X

    def get_params(self, deep=False):
        """
        Returns the parameters of the object

        TODO: add all parameters here

        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'dummy_parameter_0': self.dummy_parameter_0,
                'dummy_parameter_1': self.dummy_parameter_1,
                **OverSampling.get_params(self, deep)}

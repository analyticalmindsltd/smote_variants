"""
This module implements the Oversampling base class.
"""

import time
import warnings

from ..config import suppress_internal_warnings

from ._base import (StatisticsMixin, ParametersMixin, RandomStateMixin,
                    coalesce_dict)
from ._metrictensor import MetricLearningMixin
from ._simplexsampling import SimplexSamplingMixin

from .._logger import logger
_logger = logger

__all__= ['OverSampling',
            'OverSamplingSimplex',
            'OverSamplingBase',
            'RandomSamplingMixin']

class RandomSamplingMixin(RandomStateMixin):
    """
    This is the random sampling mixin class
    """
    def __init__(self, random_state=None):
        """
        Constructor of the mixin.

        Args:
            random_state (int/None/np.random.RandomState): initial random
                                                            state
        """
        RandomStateMixin.__init__(self, random_state)

    def get_params(self, deep=False):
        """
        Return the parameter dictionary.

        Args:
            deep (bool): whether it should be a deep query

        Returns:
            dict: the parameter dictionary
        """
        return RandomStateMixin.get_params(self, deep)

    def sample_between_points(self, x_vector, y_vector):
        """
        Sample randomly along the line between two points.
        Args:
            x_vector (np.array): point 1
            y_vector (np.array): point 2
        Returns:
            np.array: the new sample
        """
        return x_vector + (y_vector - x_vector)\
                             * self.random_state.random_sample()

    def sample_between_points_componentwise(self, x_vector, y_vector, mask=None):
        """
        Sample each dimension separately between the two points.
        Args:
            x_vector (np.array): point 1
            y_vector (np.array): point 2
            mask (np.array): array of 0,1s - specifies which dimensions
                                to sample
        Returns:
            np.array: the new sample being generated
        """
        if mask is None:
            return x_vector + (y_vector - x_vector)*self.random_state.random_sample()

        return x_vector + (y_vector - x_vector)*self.random_state.random_sample()*mask

    def sample_by_jittering(self, x_vector, std):
        """
        Sample by jittering.
        Args:
            x_vector (np.array): base point
            std (float): standard deviation
        Returns:
            np.array: the new sample
        """
        return x_vector + (self.random_state.random_sample() - 0.5)*2.0*std

    def sample_by_jittering_componentwise(self, x_vector, std):
        """
        Sample by jittering componentwise.
        Args:
            x_vector (np.array): base point
            std (np.array): standard deviation
        Returns:
            np.array: the new sample
        """
        return x_vector + (self.random_state.random_sample(len(x_vector))-0.5)*2.0 * std

    def sample_by_gaussian_jittering(self, x_vector, std):
        """
        Sample by Gaussian jittering
        Args:
            x_vector (np.array): base point
            std (np.array): standard deviation
        Returns:
            np.array: the new sample
        """
        return self.random_state.normal(x_vector, std)

class OverSamplingBase(StatisticsMixin,
                   ParametersMixin,
                   MetricLearningMixin):
    """
    Base class of oversampling methods
    """

    categories = []

    cat_noise_removal = 'NR'
    cat_dim_reduction = 'DR'
    cat_uses_classifier = 'Clas'
    cat_sample_componentwise = 'SCmp'
    cat_sample_ordinary = 'SO'
    cat_sample_copy = 'SCpy'
    cat_memetic = 'M'
    cat_density_estimation = 'DE'
    cat_density_based = 'DB'
    cat_extensive = 'Ex'
    cat_changes_majority = 'CM'
    cat_uses_clustering = 'Clus'
    cat_borderline = 'BL'
    cat_application = 'A'
    cat_metric_learning = 'CD'

    def __init__(self, checks=None):
        """
        Constructor of the base class.

        Args:
            checks (dict): the check list
        """
        StatisticsMixin.__init__(self)
        ParametersMixin.__init__(self)
        MetricLearningMixin.__init__(self)
        checks_default = {'min_n_min': 2,
                          'check_np': True}
        self.checks = coalesce_dict(checks, checks_default)

    def det_n_to_sample(self, strategy, n_maj=None, n_min=None):
        """
        Determines the number of samples to generate
        Args:
            strategy (str/float): if float, the fraction of the difference
                                    of the minority and majority numbers to
                                    generate, like 0.1 means that 10% of the
                                    difference will be generated if str,
                                    like 'min2maj', the minority class will
                                    be upsampled to match the cardinality
                                    of the majority class
            n_maj (int/None): the number of majority samples
            n_min (int/None): the number of minority samples

        Returns:
            int: the number of samples to generate
        """
        if n_maj is None:
            n_maj = self.class_stats[self.maj_label]
        if n_min is None:
            n_min = self.class_stats[self.min_label]

        if isinstance(strategy, (int, float)):
            return max([0, int((n_maj - n_min)*strategy)])

        raise ValueError(f"{self.__class__.__name__} Value {strategy} "\
                            "for parameter strategy is not supported")

    def fit_resample(self, X, y):
        """
        Alias of the function "sample" for compatibility with imbalanced-learn
        pipelines

        Args:
            X (np.array): the feature vectors
            y (np.array): the target labels

        Returns:
            np.array, np.array: the oversampled dataset
        """
        return self.sample(X, y)

    def sampling_algorithm(self, X, y):
        """
        The algorithm to be implemented.

        Args:
            X (np.array): features
            y (np.array): labels

        Returns:
            np.array, np.array: the oversampled dataset
        """
        return X, y

    def sample(self, X, y):
        """
        Sampling interface function.

        Args:
            X (np.array): features
            y (np.array): labels

        Returns:
            np.array, np.array: the oversampled dataset
        """
        _logger.info("%s: Running sampling via %s",
                        self.__class__.__name__,
                        self.descriptor())

        self.class_label_statistics(y)

        for key, item in self.checks.items():
            if key == 'min_n_min':
                if self.class_stats[self.min_label] <= item:
                    msg = f"{self.__class__.__name__}: Too few minority samples"\
                            " for sampling"
                    _logger.info(msg)
                    return X.copy(), y.copy()
            if key == 'min_n_dim':
                if X.shape[1] < item:
                    _logger.info("%s: not enough dimensions %d",
                                self.__class__.__name__, X.shape[1])
                    return X.copy(), y.copy()

        return self.sampling_algorithm(X, y)

    def return_copies(self, X, y, msg):
        """
        Returns copies of the data with logger message.

        Args:
            X (np.array): features
            y (np.array): labels
            msg (str): the logging message

        Returns:
            np.array, np.array: the oversampled dataset
        """
        _logger.info("%s: returning copies for %s",
                        self.__class__.__name__, msg)

        if not suppress_internal_warnings():
            warnings.warn(f"{self.__class__.__name__}: returning copies for {msg}")

        return X.copy(), y.copy()

    def sample_with_timing(self, X, y):
        """
        Execute the sampling with timing.

        Args:
            X (np.array): features
            y (np.array): labels
            msg (str): the logging message

        Returns:
            np.array, np.array: the oversampled dataset
        """
        begin = time.time()

        X_samp, y_samp = self.sample(X, y)

        _logger.info("%s: runtime: %f", self.__class__.__name__,
                                                (time.time() - begin))
        return X_samp, y_samp

    def preprocessing_transform(self, X):
        """
        Transforms new data according to the possible transformation
        implemented by the function "sample".

        Args:
            X (np.array): features

        Returns:
            np.array: transformed features
        """
        return X

    def get_params(self, deep=False):
        """
        Returns the parameters of the object as a dictionary.

        Returns:
            dict: the parameters of the object
        """
        _ = deep
        return {}

    def set_params(self, **params):
        """
        Set parameters

        Args:
            params (dict): dictionary of parameters
        """

        for key, value in params.items():
            setattr(self, key, value)

        return self

    def descriptor(self):
        """
        The descriptor of the class

        Returns:
            str: JSON description of the current sampling object
        """
        return str((self.__class__.__name__, str(self.get_params())))

    def __str__(self):
        """
        The string representation

        Returns:
            str: the descriptor
        """
        return self.descriptor()


class OverSampling(OverSamplingBase, RandomSamplingMixin):
    """
    The oversampling base class.
    """
    def __init__(self, random_state=None, checks=None):
        OverSamplingBase.__init__(self, checks)
        RandomSamplingMixin.__init__(self, random_state)

    def get_params(self, deep=False):
        return {**RandomSamplingMixin.get_params(self, deep),
                'class_name': self.__class__.__name__}


class OverSamplingSimplex(OverSamplingBase, SimplexSamplingMixin):
    """
    The oversampling simplex base class.
    """
    def __init__(self,
                *,
                n_dim=2,
                simplex_sampling='uniform',
                within_simplex_sampling='random',
                gaussian_component=None,
                random_state=None,
                checks=None):
        OverSamplingBase.__init__(self, checks)

        if checks is not None and 'simplex_dim' in checks:
            if n_dim != checks['simplex_dim']:
                warnings.warn(f"Simplex dimensions {n_dim} not supported "\
                                f"with the method {self.__class__.__name__} "\
                                f"forcing n_dim={checks['simplex_dim']}")
                n_dim = checks['simplex_dim']

        SimplexSamplingMixin.__init__(self,
                                        n_dim=n_dim,
                                        simplex_sampling=simplex_sampling,
                                        within_simplex_sampling=within_simplex_sampling,
                                        gaussian_component=gaussian_component,
                                        random_state=random_state)

    def get_params(self, deep=False):
        return {**SimplexSamplingMixin.get_params(self, deep),
                'class_name': self.__class__.__name__}

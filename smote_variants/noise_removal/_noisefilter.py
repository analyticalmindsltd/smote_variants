"""
This module implements the NoiseFilter base class.
"""

from ..base import (StatisticsMixin, ParametersMixin)
from ..base import MetricLearningMixin

from .._logger import logger
_logger= logger

__all__= ['NoiseFilter']

class NoiseFilter(StatisticsMixin,
                  ParametersMixin,
                  MetricLearningMixin):
    """
    Parent class of noise filtering methods
    """

    def __init__(self):
        """
        Constructor
        """
        StatisticsMixin.__init__(self)
        ParametersMixin.__init__(self)
        MetricLearningMixin.__init__(self)

    def remove_noise(self, X, y):
        """
        Removes noise
        Args:
            X (np.array): features
            y (np.array): target labels
        """

    def get_params(self, deep=False):
        """
        Return parameters

        Returns:
            dict: dictionary of parameters
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

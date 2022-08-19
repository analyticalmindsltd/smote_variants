"""
This module implements the UnderSampling base class.
"""

from ._base import StatisticsMixin, ParametersMixin, RandomStateMixin

__all__= ['UnderSampling']

class UnderSampling(StatisticsMixin,
                    ParametersMixin,
                    RandomStateMixin):
    """
    Base class of undersampling approaches.
    """

    def __init__(self, random_state=None):
        """
        Constructor
        """
        StatisticsMixin.__init__(self)
        ParametersMixin.__init__(self)
        RandomStateMixin.__init__(self, random_state=random_state)

    def sample(self, X, y):
        """
        Carry out undersampling
        Args:
            X (np.array): features
            y (np.array): labels
        Returns:
            np.array, np.array: sampled X and y
        """
        _, _ = X, y
        return None, None

    def get_params(self, deep=False):
        """
        Returns the parameters of the object as a dictionary.
        Returns:
            dict: the parameters of the object
        """
        return {**RandomStateMixin.get_params(self, deep)}

    def descriptor(self):
        """
        Returns:
            str: JSON description of the current sampling object
        """
        return str((self.__class__.__name__, str(self.get_params())))

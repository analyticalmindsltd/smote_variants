from .._base import StatisticsMixin, ParameterCheckingMixin, ParameterCombinationsMixin

__all__= ['UnderSampling']

class UnderSampling(StatisticsMixin,
                    ParameterCheckingMixin,
                    ParameterCombinationsMixin):
    """
    Base class of undersampling approaches.
    """

    def __init__(self):
        """
        Constructorm
        """
        super().__init__()

    def sample(self, X, y):
        """
        Carry out undersampling
        Args:
            X (np.matrix): features
            y (np.array): labels
        Returns:
            np.matrix, np.array: sampled X and y
        """
        pass

    def get_params(self, deep=False):
        """
        Returns the parameters of the object as a dictionary.
        Returns:
            dict: the parameters of the object
        """
        pass

    def descriptor(self):
        """
        Returns:
            str: JSON description of the current sampling object
        """
        return str((self.__class__.__name__, str(self.get_params())))

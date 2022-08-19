"""
This module implements some basic functionalities.
"""

import itertools
import importlib
import json
import pickle

from json import JSONDecodeError

import numpy as np

from .._logger import logger
_logger = logger

__all__= ['cov',
          'mode',
          'fix_density',
          'safe_divide',
          'equal_dicts',
          'coalesce',
          'coalesce_dict',
          'instantiate_obj',
          'load_dict',
          'dump_dict',
          'check_if_damaged',
          'StatisticsMixin',
          'RandomStateMixin',
          'ParametersMixin']

def cov(array, rowvar=True):
    """
    Wrapper over the numpy covariance function to handle
    the 1 feature case better.

    Args:
        array (np.array): to compute the covariance for
        rowvar (bool): whether the rows represent the observations
    """

    if not rowvar:
        if array.shape[1] > 1:
            return np.cov(array, rowvar=rowvar)
        return np.array([[np.cov(array, rowvar=rowvar)]])

    if array.shape[0] > 1:
        return np.cov(array, rowvar=rowvar)
    return np.array([[np.cov(array, rowvar=rowvar)]])

def fix_density(density):
    """
    Create a valid density distribution.
    """
    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)

    if np.sum(density) == 0.0:
        density = np.repeat(1.0, len(density))

    return density / np.sum(density)

def safe_divide(numerator, denominator, default=np.inf):
    """
    Safe division.

    Args:
        numerator (numeric): the numerator
        denominator (numeric): the denominator
        default (numeric): the value to return in case of zero division
    """
    if denominator == 0:
        return default
    return numerator/denominator

def equal_dicts(dict_0, dict_1):
    """
    Compare literal dictionaries.

    Args:
        dict_0 (dict): dictionary 0
        dict_1 (dict): dictionary 1

    Returns:
        bool: whether the dicts are equal
    """
    if len(dict_0) != len(dict_1):
        return False

    for key, val in dict_0.items():
        if isinstance(val, dict):
            if not equal_dicts(val, dict_1[key]):
                return False
        else:
            if val != dict_1[key]:
                return False

    return True

def coalesce(value_0, value_1):
    """
    The coalesce functionality.

    Args:
        value_0 (obj): first value
        value_1 (obj): second value

    Returns:
        obj: value_1 if value_0 is None otherwise value_0
    """
    if value_0 is None:
        return value_1
    return value_0

def coalesce_dict(dict_0, dict_1):
    """
    Coalesce dicts key by key

    Args:
        dict_0 (dict): the first dict
        dict_1 (dict): the second dict

    Returns:
        dict: the coalesced dict
    """
    if dict_0 is None:
        return dict_1

    result = {**dict_1}
    for key, item in dict_0.items():
        result[key] = item

    return result


def instantiate_obj(description):
    """
    Instantiates an object from a description

    Args:
        description (tuple): (module_name, class_name, params_dict)

    Returns:
        obj: the instantiated object
    """

    module = importlib.import_module(description[0])
    class_ = getattr(module, description[1])
    return class_(**description[2])

def mode(data):
    """
    Returns the mode of the data

    Args:
        data (np.array/list): the data to compute the mode of

    Returns:
        float: the mode of the data
    """
    values, counts = np.unique(data, return_counts=True)
    return values[np.where(counts == max(counts))[0][0]]

def load_dict(filename,
                serialization=None,
                array_to_list_map=None):
    """
    Load the contents of a file

    Args:
        filename (str): the filename to load
        serialization (str): 'json'/'pickle'/None
        array_to_list_map (list/None): list of keys to convert

    Returns:
        obj: the loaded object
    """
    if array_to_list_map is None:
        array_to_list_map = []

    if serialization == 'json':
        with open(filename, 'rt', encoding='UTF-8') as file:
            obj = json.load(file)
            for key in array_to_list_map:
                obj[key] = np.array(obj[key])
            return obj
    elif serialization == 'pickle':
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        raise ValueError(f"serialization {serialization} is not supported")

def dump_dict(obj,
                filename,
                serialization=None,
                array_to_list_map=None):
    """
    Serializes an object.

    Args:
        obj (obj): object to serialize
        filename (str): the filename
        serialization (str): 'json'/pickle'/None
        array_to_list_map (list/None): list of keys to convert
    """
    if array_to_list_map is None:
        array_to_list_map = []

    if serialization == 'json':
        for key in array_to_list_map:
            obj[key] = obj[key].astype(float).tolist()
        with open(filename, 'wt', encoding='UTF-8') as file:
            json.dump(obj, file)
    elif serialization == 'pickle':
        with open(filename, 'wb') as file:
            pickle.dump(obj, file)
    else:
        raise ValueError(f"serialization {serialization} is not supported")

def check_if_damaged(filename, serialization="json"):
    """
    Check if a file is available and not damaged

    Args:
        filename (str): the filename
        serialization (str): the type of serialization ('json'/'pickle')

    Returns:
        bool: True if the file is damaged or not available
    """
    damaged = False

    try:
        load_dict(filename, serialization)
    except FileNotFoundError:
        damaged = True
    except JSONDecodeError:
        damaged = True

    return damaged

class StatisticsMixin:
    """
    Mixin to compute class statistics and determine minority/majority labels
    """

    def class_label_statistics(self, y):
        """
        determines class sizes and minority and majority labels
        Args:
            X (np.array): features
            y (np.array): target labels
        """
        unique, counts = np.unique(y, return_counts=True)
        self.class_stats = dict(zip(unique, counts))
        self.min_label = unique[0] if counts[0] < counts[1] else unique[1]
        self.maj_label = unique[1] if counts[0] < counts[1] else unique[0]

    def get_params(self, deep=False):
        """
        Return the parameters

        Returns:
            dict: the parameters of the mixin
        """
        _ = deep
        return {}

    #def check_enough_min_samples_for_sampling(self, threshold=2):
    #    """
    #    Checks if there are enough samples for oversampling, if not,
    #    logs it and returns false
    #
    #    Args:
    #        threshold (int): the minimum number of minority samples
    #
    #    Returns:
    #        str/None: string massage about the problem
    #    """
    #    msg = None
    #    if self.class_stats[self.min_label] < threshold:
    #        msg = "The number of minority samples "\
    #                f"({self.class_stats[self.min_label]}) is not enough "\
    #                "for sampling"
    #    return msg

    #def check_n_to_sample(self, n_to_sample):
    #    """
    #    Check if there is need for sampling.
    #
    #    Args:
    #        n_to_sample (int): number of samples to generate
    #
    #    Returns:
    #        str/None: string message about the problem
    #    """
    #
    #    msg = None
    #    if n_to_sample == 0:
    #        msg = "There is no need for oversampling"
    #    return msg

class RandomStateMixin:
    """
    Mixin to set random state
    """
    def __init__(self, random_state):
        """
        Constructor of the mixin

        Args:
            random_state (int/np.random.RandomState/None): the random state
                                                                initializer
        """
        self.set_random_state(random_state)

    def set_random_state(self, random_state):
        """
        sets the random_state member of the object

        Args:
            random_state (int/np.random.RandomState/None): the random state
                                                                initializer
        """

        self._random_state_init = random_state

        if random_state is None:
            self.random_state = np.random
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        elif random_state is np.random:
            self.random_state = random_state
        else:
            raise ValueError(
                "random state cannot be initialized by " + str(random_state))

    def get_params(self, deep=False):
        """
        Returns the parameters of the object.

        Args:
            deep (bool): deep parameters

        Returns:
            dict: the parameter dictionary
        """
        _ = deep # disabling pylint reporting
        return {'random_state': self._random_state_init}

class ParametersMixin:
    """
    Mixin to check if parameters come from a valid range
    """

    def check_in_range(self, value, name, rng):
        """
        Check if parameter is in range
        Args:
            value (numeric): the parameter value
            name (str): the parameter name
            rng (list-like(2)): the lower and upper bound of a range
        Throws:
            ValueError
        """
        if value < rng[0] or value > rng[1]:
            msg = ("Value for parameter %s outside the range [%f,%f] not"
                 " allowed: %f")
            msg = msg % (name, rng[0], rng[1], value)

            raise ValueError(self.__class__.__name__ + ": " + msg)

    def check_out_range(self, value, name, rng):
        """
        Check if parameter is outside of range
        Args:
            value (numeric): the parameter value
            name (str): the parameter name
            rng (list-like(2)): the lower and upper bound of a range
        Throws:
            ValueError
        """
        if rng[0] <= value <= rng[1]:
            msg = "Value for parameter %s in the range [%f,%f] not allowed: %f"
            msg = msg % (name, rng[0], rng[1], value)

            raise ValueError(self.__class__.__name__ + ": " + msg)

    def check_less_or_equal(self, value, name, val):
        """
        Check if parameter is less than or equal to value
        Args:
            value (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if value > val:
            msg = "Value for parameter %s greater than %f not allowed: %f > %f"
            msg = msg % (name, val, value, val)

            raise ValueError(self.__class__.__name__ + ": " + msg)

    def check_less_or_equal_par(self, val_x, name_x, val_y, name_y):
        """
        Check if parameter is less than or equal to another parameter
        Args:
            val_x (numeric): the parameter value
            name_x (str): the parameter name
            val_y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if val_x > val_y:
            msg = ("Value for parameter %s greater than parameter %s not"
                 " allowed: %f > %f")
            msg = msg % (name_x, name_y, val_x, val_y)

            raise ValueError(self.__class__.__name__ + ": " + msg)

    def check_less(self, value, name, val):
        """
        Check if parameter is less than value
        Args:
            value (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if value >= val:
            msg = ("Value for parameter %s greater than or equal to %f"
                 " not allowed: %f >= %f")
            msg = msg % (name, val, value, val)

            raise ValueError(self.__class__.__name__ + ": " + msg)

    def check_less_par(self, val_x, name_x, val_y, name_y):
        """
        Check if parameter is less than another parameter
        Args:
            val_x (numeric): the parameter value
            name_x (str): the parameter name
            val_y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if val_x >= val_y:
            msg = ("Value for parameter %s greater than or equal to parameter"
                 " %s not allowed: %f >= %f")
            msg = msg % (name_x, name_y, val_x, val_y)

            raise ValueError(self.__class__.__name__ + ": " + msg)

    def check_greater_or_equal(self, value, name, val):
        """
        Check if parameter is greater than or equal to value
        Args:
            value (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if value < val:
            msg = "Value for parameter %s less than %f is not allowed: %f < %f"
            msg = msg % (name, val, value, val)

            raise ValueError(self.__class__.__name__ + ": " + msg)

    def check_greater_or_equal_par(self, val_x, name_x, val_y, name_y):
        """
        Check if parameter is less than or equal to another parameter
        Args:
            val_x (numeric): the parameter value
            name_x (str): the parameter name
            val_y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if val_x < val_y:
            msg = ("Value for parameter %s less than parameter %s is not"
                 " allowed: %f < %f")
            msg = msg % (name_x, name_y, val_x, val_y)

            raise ValueError(self.__class__.__name__ + ": " + msg)

    def check_greater(self, value, name, val):
        """
        Check if parameter is greater than value
        Args:
            value (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if value <= val:
            msg = ("Value for parameter %s less than or equal to %f not allowed"
                 " %f < %f")
            msg = msg % (name, val, value, val)

            raise ValueError(self.__class__.__name__ + ": " + msg)

    def check_greater_par(self, val_x, name_x, val_y, name_y):
        """
        Check if parameter is greater than or equal to another parameter
        Args:
            val_x (numeric): the parameter value
            name_x (str): the parameter name
            val_y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if val_x <= val_y:
            msg = ("Value for parameter %s less than or equal to parameter %s"
                 " not allowed: %f <= %f")
            msg = msg % (name_x, name_y, val_x, val_y)

            raise ValueError(self.__class__.__name__ + ": " + msg)

    def check_equal(self, value, name, val):
        """
        Check if parameter is equal to value
        Args:
            value (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if value == val:
            msg = ("Value for parameter %s equal to parameter %f is not allowed:"
                 " %f == %f")
            msg = msg % (name, val, value, val)
            raise ValueError(self.__class__.__name__ + ": " + msg)

    def check_equal_par(self, val_x, name_x, val_y, name_y):
        """
        Check if parameter is equal to another parameter
        Args:
            val_x (numeric): the parameter value
            name_x (str): the parameter name
            val_y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if val_x == val_y:
            msg = ("Value for parameter %s equal to parameter %s is not "
                 "allowed: %f == %f")
            msg = msg % (name_x, name_y, val_x, val_y)
            raise ValueError(self.__class__.__name__ + ": " + msg)

    def check_isin(self, value, name, list_):
        """
        Check if parameter is in list
        Args:
            value (numeric): the parameter value
            name (str): the parameter name
            list_ (list): list to check if parameter is in it
        Throws:
            ValueError
        """
        if value not in list_:
            msg = "Value for parameter %s not in list %s is not allowed: %s"
            msg = msg % (name, str(list_), str(value))
            raise ValueError(self.__class__.__name__ + ": " + msg)

    def check_n_jobs(self, value, name='n_jobs'):
        """
        Check n_jobs parameter
        Args:
            value (int/None): number of jobs
            name (str): the parameter name
        Throws:
            ValueError
        """
        if ((value is None)
                or (value is not None and isinstance(value, int) and value <= 0)):
            msg = "Value for parameter %s is not allowed: %s"
            msg = msg % (name, str(value))
            raise ValueError(self.__class__.__name__ + ": " + msg)

    @classmethod
    def generate_parameter_combinations(cls, dictionary, raw):
        """
        Generates reasonable parameter combinations
        Args:
            dictionary (dict): dictionary of parameter ranges
            num (int): maximum number of combinations to generate
        """
        if raw:
            return dictionary
        keys = sorted(list(dictionary.keys()))
        values = [dictionary[k] for k in keys]
        combinations = [dict(zip(keys, p))
                        for p in list(itertools.product(*values))]
        return combinations

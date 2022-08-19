"""
Testing the base functionalities.
"""

import os
import shutil
import json

import numpy as np
from sklearn.tree import DecisionTreeClassifier

import pytest

from smote_variants.base import (mode, StatisticsMixin, RandomStateMixin,
                            ParametersMixin, instantiate_obj, coalesce,
                            load_dict, dump_dict, check_if_damaged,
                            equal_dicts, coalesce_dict, safe_divide,
                            fix_density, cov)

def test_mode():
    """
    Testing the mode function.
    """
    assert mode([0, 1, 1, 2]) == 1

def test_cov():
    """
    Test the cov function.
    """
    assert len(cov(np.array([[1], [2], [3]]), rowvar=False).shape) == 2
    assert len(cov(np.array([[1, 2, 3]]), rowvar=True).shape) == 2

    assert cov(np.array([[1, 2], [2, 3]]), rowvar=False).shape[0] == 2
    assert cov(np.array([[1, 2], [2, 3]]), rowvar=True).shape[0] == 2

def test_fix_density():
    """
    Test the fix density function
    """
    assert fix_density(np.array([1.0, 1.0]))[0] == 0.5
    assert fix_density(np.array([1.0, np.nan]))[0] == 1.0
    assert fix_density(np.array([np.nan, np.inf]))[0] == 0.5

def test_safe_divide():
    """
    Test the safe_divide function
    """
    assert safe_divide(1, 2, 0.0) == 1.0/2.0
    assert safe_divide(1, 0, 5) == 5

def test_equal_dicts():
    """
    Testing the equality check of dictionaries.
    """
    assert equal_dicts({'a': 'a'}, {'a': 'a'})
    assert equal_dicts({'a': {'a': 'a'}}, {'a': {'a': 'a'}})
    assert not equal_dicts({'a': 'a'}, {'a': 'b'})
    assert not equal_dicts({'a': {'a': 'a'}}, {'a': {'a': 'b'}})
    assert not equal_dicts({'a': {'a': 'a'}}, {'a': {'a': 'b'}, 'b': 2})

def test_coalesce():
    """
    Testing the coalesce function
    """
    assert coalesce(None, 1) == 1
    assert coalesce(2, 1) == 2

def test_coalesce_dict():
    """
    Testing the coalesce dict functionality
    """
    dict_1 = {'a': 0, 'b': 1, 'd': 4}
    dict_0 = {'b': 2, 'c': 3}

    result = coalesce_dict(dict_0, dict_1)

    assert len(result) == 4
    assert result['b'] == 2

    result = coalesce_dict(None, dict_1)

    assert equal_dicts(dict_1, result)

def test_instantiate_obj():
    """
    Testing the object instantiation
    """
    obj = instantiate_obj(('sklearn.tree', 'DecisionTreeClassifier', {}))
    assert obj.__class__ == DecisionTreeClassifier

def test_load_dump_dict():
    """
    Testing the dictionary dumping
    """
    test_dir = 'test_base_dir'
    os.makedirs(test_dir, exist_ok=True)

    filename = os.path.join(test_dir, 'dump.pickle')

    obj = {'a': 2,
            'b': np.array([1, 2])}
    dump_dict(obj, filename, 'pickle', ['b'])
    obj_loaded = load_dict(filename, 'pickle', ['b'])

    np.testing.assert_array_equal(obj['b'], obj_loaded['b'])

    dump_dict(obj, filename, 'json', ['b'])
    obj_loaded = load_dict(filename, 'json', ['b'])

    np.testing.assert_array_equal(obj['b'], obj_loaded['b'])

    with pytest.raises(ValueError):
        dump_dict(obj, filename, 'nonsense', ['b'])

    with pytest.raises(ValueError):
        load_dict(filename, 'nonsense', ['b'])

    obj = {'a': 2}
    dump_dict(obj, filename, 'pickle')
    obj_loaded = load_dict(filename, 'pickle')

    assert obj['a'] == obj_loaded['a']

    shutil.rmtree(test_dir)

def test_check_if_damaged():
    """
    Testing the file availability functionality.
    """
    test_dir = 'test_base_dir'
    os.makedirs(test_dir, exist_ok=True)

    assert check_if_damaged(os.path.join(test_dir, 'nonsense.json'), 'json')

    with open(os.path.join(test_dir, 'a.json'), 'wt', encoding='utf-8') as file:
        file.write('nonsense')

    assert check_if_damaged(os.path.join(test_dir, 'a.json'), 'json')

    with open(os.path.join(test_dir, 'a.json'), 'wt', encoding='utf-8') as file:
        json.dump({'a': 2}, file)

    assert not check_if_damaged(os.path.join(test_dir, 'a.json'), 'json')

    shutil.rmtree(test_dir)


def test_statistics_mixin():
    """
    Testing the statistics mixin.
    """

    class Mock(StatisticsMixin):
        """
        Mock class
        """

    mock = Mock()

    mock.class_label_statistics([0, 0, 0, 1])

    assert mock.min_label == 1
    assert mock.maj_label == 0

    assert len(StatisticsMixin().get_params()) == 0

def test_random_state_mixin():
    """
    Testing the random state mixin.
    """
    obj = RandomStateMixin(random_state=None)
    assert obj.random_state == np.random

    obj = RandomStateMixin(random_state=5)
    assert isinstance(obj.random_state, np.random.RandomState)
    assert obj.get_params()['random_state'] == 5

    obj = RandomStateMixin(random_state=np.random.RandomState(5))
    assert isinstance(obj.random_state, np.random.RandomState)

    obj = RandomStateMixin(random_state=np.random)
    assert obj.random_state == np.random

    with pytest.raises(ValueError):
        obj = RandomStateMixin(random_state='apple')

    assert len(obj.get_params()) == 1

def test_parameters_mixin():
    """
    Testing the parameters mixin
    """
    pmixin = ParametersMixin()

    with pytest.raises(ValueError):
        pmixin.check_in_range(5, 'five', [0, 2])

    with pytest.raises(ValueError):
        pmixin.check_out_range(1, 'five', [0, 2])

    with pytest.raises(ValueError):
        pmixin.check_less_or_equal(5, 'five', 3)

    with pytest.raises(ValueError):
        pmixin.check_less_or_equal_par(5, 'five', 3, 'three')

    with pytest.raises(ValueError):
        pmixin.check_less(5, 'five', 3)

    with pytest.raises(ValueError):
        pmixin.check_less_par(5, 'five', 3, 'three')

    with pytest.raises(ValueError):
        pmixin.check_greater_or_equal(5, 'five', 8)

    with pytest.raises(ValueError):
        pmixin.check_greater_or_equal_par(5, 'five', 8, 'three')

    with pytest.raises(ValueError):
        pmixin.check_greater(5, 'five', 8)

    with pytest.raises(ValueError):
        pmixin.check_greater_par(5, 'five', 8, 'three')

    with pytest.raises(ValueError):
        pmixin.check_equal(5, 'five', 5)

    with pytest.raises(ValueError):
        pmixin.check_equal_par(5, 'five', 5, 'three')

    with pytest.raises(ValueError):
        pmixin.check_isin(5, 'five', [1, 2, 6])

    with pytest.raises(ValueError):
        pmixin.check_n_jobs(None)

def test_parameter_combinations():
    """
    Testing the parameter combinations functionality.
    """
    params_base = {'a': [1, 2],
                    'b': ['a', 'b', 'c']}
    params = ParametersMixin.generate_parameter_combinations(params_base,
                                                                raw=False)
    assert len(params) == 6

    assert len(ParametersMixin.generate_parameter_combinations(params_base,
                                                                raw=True)) == 2

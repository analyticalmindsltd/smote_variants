"""
Testing the dataset folding.
"""

import os
import shutil

from sklearn import datasets
import pytest

from smote_variants.evaluation import Folding

dataset = datasets.load_breast_cancer()
dataset['name'] = 'breast_cancer'

cache_path = os.path.join('.', 'test_path')

shutil.rmtree(cache_path, ignore_errors=True)

def test_basic():
    """
    Testing the basic functionalities.
    """
    folding = Folding(dataset,
                        cache_path=cache_path,
                        serialization='json')
    lens = [len(fold['X_test']) for fold in folding.fold()]

    folding.cache_foldings()

    assert len(lens) == 10
    assert len(folding.folding_files()) == 10

def test_json_specific_validator():
    """
    Testing the json serialization.
    """
    folding = Folding(dataset,
                        validator_params={'n_repeats': 1,
                                            'n_splits': 5,
                                            'random_state': 5},
                        cache_path=cache_path,
                        serialization='json',
                        reset=True)
    lens = [len(fold['X_test']) for fold in folding.fold()]
    assert len(lens) == 5

    folding = Folding(dataset,
                        cache_path=cache_path,
                        serialization='json',
                        reset=True)
    lens = [len(fold['X_test']) for fold in folding.fold()]
    assert len(lens) == 10

def test_pickle():
    """
    Testing the pickle serialization.
    """
    folding = Folding(dataset,
                        cache_path=cache_path,
                        serialization='pickle')
    lens = [len(fold['X_test']) for fold in folding.fold()]
    assert len(lens) == 10
    assert len(folding.folding_files()) == 10

def test_nonsense():
    """
    Testing with a nonsense parameter.
    """
    with pytest.raises(ValueError):
        Folding(dataset,
                cache_path=cache_path,
                serialization='nonsense').cache_foldings()

def test_get_params():
    """
    Testing the get_params function.
    """
    folding = Folding(dataset,
                        cache_path=cache_path,
                        serialization='json')
    assert str(folding.get_params()) == folding.descriptor()

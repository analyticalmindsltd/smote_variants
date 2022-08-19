"""
Tests oversamplers with a few samples.
"""

import pytest
import numpy as np

import smote_variants as sv

from smote_variants.datasets import (load_1_dim,
                                     load_illustration_2_class,
                                     load_normal,
                                     load_same_num,
                                     load_some_min_some_maj,
                                     load_1_min_some_maj,
                                     load_2_min_some_maj,
                                     load_3_min_some_maj,
                                     load_4_min_some_maj,
                                     load_5_min_some_maj,
                                     load_1_min_1_maj,
                                     load_repeated,
                                     load_all_min_noise,
                                     load_separable,
                                     load_linearly_dependent,
                                     load_alternating,
                                     load_high_dim)

oversamplers = [sv.SMOTE()]

@pytest.mark.parametrize("smote_obj", oversamplers)
def test_1_dim(smote_obj):
    """
    Testing oversamplers with 1 minority sample.

    Args:
        smote_obj (obj): an oversampler obj.
    """
    dataset = load_1_dim()

    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", oversamplers)
def test_1_min_some_maj(smote_obj):
    """
    Testing oversamplers with 1 minority sample.

    Args:
        smote_obj (obj): an oversampler obj.
    """
    dataset = load_1_min_some_maj()

    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", oversamplers)
def test_1_min_1_maj(smote_obj):
    """
    Testing oversamplers with 1 minority and 1 majority sample.

    Args:
        smote_obj (obj): an oversampler obj.
    """
    dataset = load_1_min_1_maj()

    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", oversamplers)
def test_repeated(smote_obj):
    """
    Testing oversamplers with repeated samples.

    Args:
        smote_obj (obj): an oversampler obj.
    """
    dataset = load_repeated()

    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", oversamplers)
def test_2_min_some_maj(smote_obj):
    """
    Testing oversamplers with 2 minority samples.

    Args:
        smote_obj (obj): an oversampler obj.
    """
    dataset = load_2_min_some_maj()

    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", oversamplers)
def test_3_min_some_maj(smote_obj):
    """
    Testing oversamplers with 3 minority samples.

    Args:
        smote_obj (obj): an oversampler obj.
    """
    dataset = load_3_min_some_maj()

    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", oversamplers)
def test_4_min_some_maj(smote_obj):
    """
    Testing oversamplers with 4 minority samples.

    Args:
        smote_obj (obj): an oversampler obj.
    """
    dataset = load_4_min_some_maj()

    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", oversamplers)
def test_5_min_some_maj(smote_obj):
    """
    Testing oversamplers with 5 minority samples.

    Args:
        smote_obj (obj): an oversampler obj.
    """
    dataset = load_5_min_some_maj()

    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", oversamplers)
def test_all_min_noise(smote_obj):
    """
    Testing oversamplers with all minority samples being noise.

    Args:
        smote_obj (obj): an oversampler obj.
    """
    dataset = load_all_min_noise()

    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", oversamplers)
def test_alternating(smote_obj):
    """
    Testing oversamplers with alternating minority samples.

    Args:
        smote_obj (obj): an oversampler obj.
    """
    dataset = load_alternating()

    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", oversamplers)
def test_high_dim(smote_obj):
    """
    Testing an oversampler with high dimensionality data.

    Args:
        smote_obj (obj): an oversampler obj
    """
    dataset = load_high_dim()

    X, y = smote_obj.sample(dataset['data'], dataset['target'])
    assert len(X) > 0
    assert X.shape[1] == smote_obj\
                                .preprocessing_transform(dataset['data']).shape[1]
    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", oversamplers)
def test_illustration(smote_obj):
    """
    Testing oversamplers with illustration data.

    Args:
        smote_obj (obj): an oversampler obj.
    """
    dataset = load_illustration_2_class()

    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", oversamplers)
def test_linearly_dependent(smote_obj):
    """
    Testing oversamplers with 2 minority samples.

    Args:
        smote_obj (obj): an oversampler obj.
    """
    dataset = load_linearly_dependent()

    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", oversamplers)
def test_normal(smote_obj):
    """
    Tests an oversmampler

    Args:
        smote_obj (obj): an oversampler obj
    """
    dataset = load_normal()

    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", oversamplers)
def test_reproducibility(smote_obj):
    """
    Tests the reproducibility of oversampling.

    Args:
        smote_obj (obj): an oversampling obj
    """
    dataset = load_normal()

    X_normal = dataset['data'] # pylint: disable=invalid-name
    y_normal = dataset['target']

    X_orig = X_normal.copy()
    y_orig = y_normal.copy()

    X_a, y_a = smote_obj.__class__(random_state=5).sample(X_normal, y_normal)
    oversampler = smote_obj.__class__(random_state=5)
    X_b, y_b = oversampler.sample(X_normal, y_normal)
    X_c, y_c = smote_obj.__class__(**oversampler.get_params()).sample(X_normal, y_normal)

    assert np.array_equal(X_a, X_b)
    assert np.array_equal(X_b, X_c)
    assert np.array_equal(X_orig, X_normal)

    assert np.array_equal(y_a, y_b)
    assert np.array_equal(y_b, y_c)
    assert np.array_equal(y_orig, y_normal)

@pytest.mark.parametrize("smote_obj", oversamplers)
def test_same_num(smote_obj):
    """
    Tests oversamplers with equalized data.

    Args:
        smote_obj (obj): an oversampling obj
    """
    dataset = load_same_num()

    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", oversamplers)
def test_separable(smote_obj):
    """
    Testing oversamplers with 3 minority samples.

    Args:
        smote_obj (obj): an oversampler obj.
    """
    dataset = load_separable()

    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", oversamplers)
def test_some_min_some_maj(smote_obj):
    """
    Tests an oversampler with only a few samples.

    Args:
        smote_obj (obj): the oversampler obj
    """
    dataset = load_some_min_some_maj()

    X, y = smote_obj.sample(dataset['data'], dataset['target'])

    assert np.unique(y).shape[0] == 2
    assert X.shape[0] > 0

@pytest.mark.parametrize("smote_obj", oversamplers)
def test_parameters(smote_obj):
    """
    Test the parameterization.

    Args:
        smote_obj (obj): an oversampling object
    """
    random_state = np.random.RandomState(5)

    par_comb = smote_obj.__class__.parameter_combinations()

    original_parameters = random_state.choice(par_comb)
    oversampler = smote_obj.__class__(**original_parameters)
    parameters = oversampler.get_params()

    assert all(v == parameters[k] for k, v in original_parameters.items())

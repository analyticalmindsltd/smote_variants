"""
Testing the classification performance metrics.
"""

import numpy as np

from smote_variants.base import (prediction_labels, calculate_atoms,
                        calculate_label_scores, calculate_prob_scores,
                        calculate_all_scores)

def test_prediction_labels():
    """
    Testing the prediction label calculation.
    """
    labels = prediction_labels(np.array([0.4]))
    assert labels[0] == 0

    labels = prediction_labels(np.array([0.4, 0.6]))
    assert labels[0] == 0 and labels[1] == 1

    labels = prediction_labels(np.array([0.5, 0.5]))
    assert labels[0] == 0 and labels[1] == 1

def test_calculate_atoms():
    """
    Testing the calculation of atoms.
    """
    atoms = calculate_atoms(np.array([0, 0, 0, 0, 1, 1]),
                            np.array([0, 1, 0, 0, 1, 1]))

    assert atoms['tp'] == 2
    assert atoms['tn'] == 3
    assert atoms['fp'] == 1
    assert atoms['fn'] == 0

def test_calculate_label_scores():
    """
    Testing the calculation of label scores.
    """
    # no fn
    atoms = calculate_atoms(np.array([0, 0, 0, 0, 1, 1]),
                            np.array([0, 1, 0, 0, 1, 1]))
    atoms = calculate_label_scores(atoms)

    assert atoms['acc'] == 5/6
    assert atoms['lfn'] is None

    # no fp
    atoms = calculate_atoms(np.array([0, 1, 0, 0, 1, 1]),
                            np.array([0, 0, 0, 0, 1, 1]))
    atoms = calculate_label_scores(atoms)

    assert atoms['lfp'] is None

    # no tp
    atoms = calculate_atoms(np.array([0, 1, 0, 0, 1, 1]),
                            np.array([0, 0, 0, 0, 0, 0]))
    atoms = calculate_label_scores(atoms)

    assert atoms['ltp'] is None

    # no tn
    atoms = calculate_atoms(np.array([0, 1, 0, 0, 1, 1]),
                            np.array([1, 1, 1, 1, 1, 1]))
    atoms = calculate_label_scores(atoms)

    assert atoms['ltn'] is None

def test_calculate_prob_scores():
    """
    Testing the calculation of probability scores.
    """
    probs = np.array([0.4, 0.6])

    scores = calculate_prob_scores(np.array([0, 1]), probs)
    assert scores['auc'] == 1.0

def test_compute_all_scores():
    """
    Testing the calculation of all scores.
    """
    probs = np.array([0.4, 0.6])

    scores = calculate_all_scores(np.array([0, 1]), probs)
    assert scores['auc'] == 1.0

    probs = np.array([0.1, 0.6, 0.3, 0.9, 0.8, 0.2, 0.3, 0.9, 0.0, 1.0])
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    scores = calculate_all_scores(labels, probs)
    assert scores['uc'] is not None

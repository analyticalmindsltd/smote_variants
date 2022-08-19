"""
This module implements some imbalanced classification metrics.
"""

import numpy as np

from sklearn.metrics import roc_auc_score, log_loss

__all__ = ['prediction_labels',
            'calculate_atoms',
            'calculate_label_scores',
            'calculate_prob_scores',
            'calculate_all_scores',
            'all_scores']

all_scores = ['acc', 'sens', 'spec', 'ppv', 'npv', 'fpr', 'fdr',
                'fnr', 'bacc', 'gacc', 'f1', 'mcc', 'l', 'ltp', 'lfp', 'lfn',
                'ltn', 'lp', 'ln', 'uc', 'informedness', 'markedness', 'p_top20',
                'brier', 'log_loss', 'auc']

def prediction_labels(probabilities_maj):
    """
    Determine the labels from the probabilities.

    Args:
        probabilities_maj (np.array): the majority probabilities

    Returns:
        np.array: the labels row-by-row
    """
    labels = (probabilities_maj > 0.5) * 1
    equals = probabilities_maj == 0.5
    indices = np.where(equals)[0]
    if len(indices) <= 1:
        return labels

    half = int(len(indices)/2)
    labels[indices[:half]] = 0
    labels[indices[half:]] = 1
    return labels

def calculate_atoms(test_labels, predicted_labels, min_label=1):
    """
    Calculate the atoms used for the measures.

    Args:
        test_labels (np.array): the true labels
        predicted_labels (np.array): the predicted labels
        min_label (int): the minority label

    Returns:
        dict: the atoms
    """

    atoms = {}

    equals = np.equal(test_labels, predicted_labels)
    not_equals = np.logical_not(equals)

    min_sample = test_labels == min_label
    maj_sample = np.logical_not(min_sample)

    atoms['tp'] = int(np.sum(np.logical_and(equals, min_sample)))
    atoms['tn'] = int(np.sum(np.logical_and(equals, maj_sample)))

    atoms['fp'] = int(np.sum(np.logical_and(not_equals, maj_sample)))
    atoms['fn'] = int(np.sum(np.logical_and(not_equals, min_sample)))

    return atoms

def _log_score(multiplier, value):
    """
    Calculates a log score and returs None if not computable.

    Args:
        multiplier (float): the multiplier
        value (float): the value to take the log of

    Returns:
        float: the score
    """
    if value > 0:
        log_value = np.log(value)
    else:
        log_value = np.nan
    if not np.isfinite(log_value):
        return None
    return float(multiplier * log_value)

def _log_score_div(numerator, denominator):
    """
    Calculates a log score and returs None if not computable.

    Args:
        nominator (float): the nominator
        denominator (float): the denominator

    Returns:
        float: the score
    """
    if denominator > 0:
        return _log_score(numerator, numerator / denominator)
    return None

def calculate_label_scores(atoms):
    """
    Calculate scores from labels.

    Args:
        atoms (dict): the atomic scores

    Returns:
        dict: the label scores
    """
    atoms['p'] = atoms['tp'] + atoms['fn']
    atoms['n'] = atoms['fp'] + atoms['tn']

    atoms['acc'] = (atoms['tp'] + atoms['tn']) / (atoms['p'] + atoms['n'])
    atoms['sens'] = atoms['tp'] / atoms['p']
    atoms['spec'] = atoms['tn'] / atoms['n']
    if atoms['tp'] + atoms['fp'] > 0:
        atoms['ppv'] = atoms['tp'] / (atoms['tp'] + atoms['fp'])
    else:
        atoms['ppv'] = 0.0
    if atoms['tn'] + atoms['fn'] > 0:
        atoms['npv'] = atoms['tn'] / (atoms['tn'] + atoms['fn'])
    else:
        atoms['npv'] = 0.0
    atoms['fpr'] = 1.0 - atoms['spec']
    atoms['fdr'] = 1.0 - atoms['ppv']
    atoms['fnr'] = 1.0 - atoms['sens']
    atoms['bacc'] = (atoms['sens'] + atoms['spec'])/2.0
    atoms['gacc'] = float(np.sqrt(atoms['sens']*atoms['spec']))
    atoms['f1'] = 2 * atoms['tp'] / (2 * atoms['tp'] + atoms['fp'] + atoms['fn'])

    tp_fp = (atoms['tp'] + atoms['fp'])
    tp_fn = (atoms['tp'] + atoms['fn'])
    tn_fp = (atoms['fp'] + atoms['tn'])
    tn_fn = (atoms['fn'] + atoms['tn'])

    mcc_num = atoms['tp']*atoms['tn'] - atoms['fp']*atoms['fn']
    mcc_denom = float(np.prod([tp_fp, tp_fn, tn_fp, tn_fn]))

    if mcc_denom == 0:
        atoms['mcc'] = None
    else:
        atoms['mcc'] = float(mcc_num/np.sqrt(mcc_denom))

    atoms['l'] = float((atoms['p'] + atoms['n']) * np.log(atoms['p'] + atoms['n']))

    atoms['ltp'] = _log_score_div(atoms['tp'], tp_fp * tp_fn)
    atoms['lfp'] = _log_score_div(atoms['fp'], tp_fp * tn_fp)
    atoms['lfn'] = _log_score_div(atoms['fn'], tp_fn * tn_fn)
    atoms['ltn'] = _log_score_div(atoms['tn'], tn_fp * tn_fn)

    atoms['lp'] = float(atoms['p'] * np.log(atoms['p']/(atoms['p'] + atoms['n'])))
    atoms['ln'] = float(atoms['n'] * np.log(atoms['n']/(atoms['p'] + atoms['n'])))

    items = [atoms['ltp'], atoms['lfp'], atoms['lfn'], atoms['ltn']]

    if np.all([item is not None for item in items]):
        uc_num = atoms['l'] + np.sum(items)
        uc_denom = atoms['l'] + atoms['lp'] + atoms['ln']
        atoms['uc'] = uc_num / uc_denom
    else:
        atoms['uc'] = None

    atoms['informedness'] = atoms['sens'] + atoms['spec'] - 1.0
    atoms['markedness'] = atoms['ppv'] + atoms['npv'] - 1.0

    return atoms

def calculate_prob_scores(test_labels, probabilities, min_label=1):
    """
    Calculate scores from probabilities.

    Args:
        test_labels (np.array): the true labels
        probabilities (np.array): the probabilities
        min_label (int): the minority label

    Returns:
        dict: the probability scores
    """
    results = {}

    thres = max(int(0.2*len(test_labels)), 1)
    results['p_top20'] = float(np.sum(test_labels[:thres] == min_label)/thres)
    results['brier'] = float(np.mean((probabilities - test_labels)**2))
    results['log_loss'] = float(log_loss(test_labels, probabilities))
    results['auc'] = float(roc_auc_score(test_labels, probabilities))

    return results

def calculate_all_scores(test_labels, probabilities, min_label=1):
    """
    Calculate all scores.

    Args:
        test_labels (np.array): the true labels
        probabilities (np.array): the probabilities
        min_label (int): the minority label

    Returns:
        dict: all scores
    """
    pred_labels = prediction_labels(probabilities)
    results = calculate_atoms(test_labels, pred_labels, min_label)
    results = calculate_label_scores(results)
    results = {**results, **calculate_prob_scores(test_labels,
                                                  probabilities,
                                                  min_label)}
    return results

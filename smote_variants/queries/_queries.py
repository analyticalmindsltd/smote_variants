"""
This module implements flexible oversampler queries.
"""

import inspect

import numpy as np

from ..base import OverSampling, SimplexSamplingMixin
from ..noise_removal import NoiseFilter
from ..oversampling import (A_SUWO, ADASYN, ADG, ADOMS, AHC, AMSCO,
AND_SMOTE, ANS, ASMOBD, Assembled_SMOTE, Borderline_SMOTE1, Borderline_SMOTE2,
CBSO, CCR, CE_SMOTE, cluster_SMOTE, CURE_SMOTE, DBSMOTE, DE_oversampling, DEAGO,
distance_SMOTE, DSMOTE, DSRBF, E_SMOTE, Edge_Det_SMOTE, G_SMOTE, GASMOTE,
Gaussian_SMOTE, polynom_fit_SMOTE_star, polynom_fit_SMOTE_bus, polynom_fit_SMOTE_poly,
polynom_fit_SMOTE_mesh, Gazzah, IPADE_ID, ISMOTE, ISOMAP_Hybrid, KernelADASYN,
kmeans_SMOTE, Lee, LLE_SMOTE, LN_SMOTE, LVQ_SMOTE, MCT, MDO, MOT2LD, MSYN,
MWMOTE, NDO_sampling, NEATER, NRAS, NRSBoundary_SMOTE, NT_SMOTE,
OUPS, PDFOS, ProWSyn, Random_SMOTE, ROSE, RWO_sampling, Safe_Level_SMOTE,
SDSMOTE, Selected_SMOTE, SL_graph_SMOTE, SMMO, SMOBD, SMOTE_Cosine,
SMOTE_D, SMOTE_ENN, SMOTE_FRST_2T, SMOTE_IPF, SMOTE_OUT, SMOTE_PSO,
SMOTE_PSOBAT, SMOTE_RSB, SMOTE_TomekLinks, SMOTE, SN_SMOTE, SOI_CJ,
SOMO, SPY, SSO, Stefanowski, SUNDO, Supervised_SMOTE, SVM_balance, SYMPROD,
TRIM_SMOTE, V_SYNTH, VIS_RST, MSMOTE)

from ._runtimes import runtimes

__all__= ['get_all_oversamplers',
          'get_metric_learning_oversamplers',
          'get_simplex_sampling_oversamplers',
          'get_multiclass_oversamplers',
          'get_all_noisefilters',
          'generate_parameter_combinations']

def determine_n_quickest(oversamplers, n_quickest):
    """
    Determine the n quickest oversamplers.

    Args:
        oversamplers (list): a list of oversampler classes
        n_quickest (int): the number of quickest to be determined

    Returns:
        list: the n quickest oversamplers
    """

    if n_quickest == -1:
        return oversamplers

    pairs = [(osampler, runtimes[osampler.__name__])
                                for osampler in oversamplers]

    sorted_pairs = sorted(pairs, key=lambda x: x[1])

    n_quickest = np.min([n_quickest, len(oversamplers)])

    return [pair[0] for pair in sorted_pairs[:n_quickest]]


def get_all_oversamplers(n_quickest=-1):
    """
    Returns all oversampling classes

    Args:
        n_quickest (int): the number of quickest oversamplers to return

    Returns:
        list(OverSampling): list of all oversampling classes

    Example::

        import smote_variants as sv

        oversamplers= sv.get_all_oversamplers()
    """

    oversamplers = [A_SUWO, ADASYN, ADG, ADOMS, AHC, AMSCO, AND_SMOTE,
            ANS, ASMOBD, Assembled_SMOTE, Borderline_SMOTE1, Borderline_SMOTE2,
           CBSO, CCR, CE_SMOTE, cluster_SMOTE, CURE_SMOTE, DBSMOTE, DE_oversampling,
           DEAGO, distance_SMOTE, DSMOTE, DSRBF, E_SMOTE, Edge_Det_SMOTE, G_SMOTE,
           GASMOTE, Gaussian_SMOTE, polynom_fit_SMOTE_star, polynom_fit_SMOTE_bus,
           polynom_fit_SMOTE_poly, polynom_fit_SMOTE_mesh, Gazzah, IPADE_ID, ISMOTE,
           ISOMAP_Hybrid, KernelADASYN, kmeans_SMOTE, Lee, LLE_SMOTE, LN_SMOTE,
           LVQ_SMOTE, MCT, MDO, MOT2LD, MSYN, MWMOTE, NDO_sampling, NEATER,
           NRAS, NRSBoundary_SMOTE, NT_SMOTE, OUPS, PDFOS, ProWSyn, Random_SMOTE,
           ROSE, RWO_sampling, Safe_Level_SMOTE, SDSMOTE, Selected_SMOTE, SL_graph_SMOTE,
           SMMO, SMOBD, SMOTE_Cosine, SMOTE_D, SMOTE_ENN, SMOTE_FRST_2T, SMOTE_IPF,
           SMOTE_OUT, SMOTE_PSO, SMOTE_PSOBAT, SMOTE_RSB, SMOTE_TomekLinks, SMOTE,
           SN_SMOTE, SOI_CJ, SOMO, SPY, SSO, Stefanowski, SUNDO, Supervised_SMOTE,
           SVM_balance, SYMPROD, TRIM_SMOTE, V_SYNTH, VIS_RST, MSMOTE]

    return determine_n_quickest(oversamplers, n_quickest)

def get_metric_learning_oversamplers(n_quickest=-1):
    """
    Returns all oversampling classes supporting the use of metric learning
    for neighborhood calculations

    Args:
        n_quickest (int): the number of quickest oversamplers to return

    Returns:
        list(OverSampling): the list of all oversampling classes supporting
                            classifier induced distance

    """
    oversamplers = get_all_oversamplers()

    oversamplers = [o for o in oversamplers if
                    OverSampling.cat_metric_learning in o.categories]

    return determine_n_quickest(oversamplers, n_quickest)

def check_within_simplex_sampling(ss_params,
                                  within_simplex_sampling):
    """
    Check if the ss_params is conformant with the specification in
    within_simplex_sampling

    Args:
        ss_params (dict): the simplex sampling parameters
        within_simplex_sampling (str/list): the specification to check against

    Returns:
        bool: whether the specification is met
    """
    if isinstance(within_simplex_sampling, str):
        return ss_params['within_simplex_sampling'] == within_simplex_sampling

    return ss_params['within_simplex_sampling'] in within_simplex_sampling

def check_gaussian_component(ss_params,
                             with_gaussian_component):
    """
    Check if the ss_params is conformant with the specification of
    with_gaussian_component

    Args:
        ss_params (dict): the simplex sampling parameters
        with_gaussian_component (bool): the specification to check against

    Returns:
        bool: whether the specification is met
    """
    if with_gaussian_component is True:
        return (ss_params['gaussian_component'] is not None)\
                            and (len(ss_params['gaussian_component']) > 0)
    return not ((ss_params['gaussian_component'] is not None)\
                            and (len(ss_params['gaussian_component']) > 0))

def check_n_dim(ss_params,
                n_dim_range):
    """
    Check if the ss_params is conformant with the specification of
    n_dim_range

    Args:
        ss_params (dict): the simplex sampling parameters
        n_dim_range (int/list): the specification to check against

    Returns:
        bool: whether the specification is met
    """
    if isinstance(n_dim_range, int):
        return ss_params['n_dim'] == n_dim_range

    return ss_params['n_dim'] in n_dim_range

def get_simplex_sampling_oversamplers(within_simplex_sampling=None,
                                      exclude_within_simplex_sampling=None,
                                      with_gaussian_component=None,
                                      n_dim_range=None,
                                      n_quickest=-1):
    """
    Returns all oversampling classes supporting simplex sampling.

    Args:
        within_simplex_sampling (str/list/None): the value or values for the
                                            within simplex sampling strategies
        within_simplex_sampling (str/list/None): the value or values for the
                                            undesired within simplex sampling
                                            strategies
        with gaussian component (bool): only those techniques not having a
                                        Guassian component set by default
        n_dim_range (int/list): the desired n_dim value(s)
        n_quickest (int): the number of quickest oversamplers to return

    Returns:
        list(OverSampling): the list of all oversampling classes supporting
                            simplex sampling
    """

    oversamplers = get_all_oversamplers()

    results = []

    for os_class in oversamplers:
        if 'ss_params' not in list(inspect.signature(os_class).parameters.keys()):
            continue

        ss_params = None
        os_obj = os_class()

        if hasattr(os_obj, 'ss_params'):
            ss_params = os_obj.ss_params
        else:
            ss_params = SimplexSamplingMixin.get_params(os_obj)['ss_params']

        checks = []

        if within_simplex_sampling is not None:
            checks.append(check_within_simplex_sampling(ss_params,
                                                    within_simplex_sampling))

        if exclude_within_simplex_sampling is not None:
            checks.append(not check_within_simplex_sampling(ss_params,
                                            exclude_within_simplex_sampling))

        if with_gaussian_component is not None:
            checks.append(check_gaussian_component(ss_params,
                                                with_gaussian_component))

        if n_dim_range is not None:
            checks.append(check_n_dim(ss_params, n_dim_range))

        if len(checks) == 0 or all(checks):
            results.append(os_class)

    return determine_n_quickest(results, n_quickest)

def multiclass_filter(sampler):
    """
    Multiclass filter

    Args:
        sampler (obj): oversampler class

    Returns:
        bool: True if suitable for multiclass oversampling
    """
    term_0 = OverSampling.cat_changes_majority not in sampler.categories
    term_1 = OverSampling.cat_dim_reduction not in sampler.categories
    term_2 = 'proportion' in sampler().get_params()

    return term_0 and term_1 and term_2

def get_multiclass_oversamplers(strategy="eq_1_vs_many_successive",
                                    n_quickest=-1):
    """
    Returns all oversampling classes which can be used with the multiclass
    strategy specified

    Args:
        strategy (str): the multiclass oversampling strategy -
                        'eq_1_vs_many_successive'/'equalize_1_vs_many'
        n_quickest (int): the number of quickest oversamplers to return

    Returns:
        list(OverSampling): list of all oversampling classes which can be used
                            with the multiclass strategy specified

    Example::

        import smote_variants as sv

        oversamplers= sv.get_multiclass_oversamplers()

    """

    oversamplers = get_all_oversamplers()

    if strategy in ('eq_1_vs_many_successive', 'equalize_1_vs_many'):
        oversamplers = [o for o in oversamplers if multiclass_filter(o)]

    return determine_n_quickest(oversamplers, n_quickest)

def get_all_noisefilters():
    """
    Returns all noise filters
    Returns:
        list(NoiseFilter): list of all noise filter classes
    """
    return NoiseFilter.__subclasses__()


def generate_parameter_combinations(classes,
                                    n_max_comb=5,
                                    result_format='smote_variants',
                                    random_seed=5):
    """
    Return random parameter combinations for a set of oversamplers.

    Args:
        list(class): list of oversampling classes
        max_n_comb (int): the maximum number of parameter combinations
                            per oversampling technique
        result_format (str): 'dict'/'smote_variants' - the format of the results
        random_seed (int): the random seed to use
    """
    random_state = np.random.RandomState(random_seed)

    results = []

    for oversampler in classes:
        parameter_combinations = oversampler.parameter_combinations()
        random_state.shuffle(parameter_combinations)
        n_comb = np.min([len(parameter_combinations), n_max_comb])
        parameter_combinations = parameter_combinations[:n_comb]
        parameter_combinations = [oversampler(**pcomb).get_params()
                                    for pcomb in parameter_combinations]
        results.extend(parameter_combinations)

    if result_format == 'dict':
        return results

    return [('smote_variants', r['class_name'], r) for r in results]

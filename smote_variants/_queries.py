from .oversampling._OverSampling import OverSampling

from ._smote_variants import *
from .noise_removal import NoiseFilter

__all__= ['get_all_oversamplers',
          'get_n_quickest_oversamplers',
          'get_metric_learning_oversamplers',
          'get_all_oversamplers_multiclass',
          'get_n_quickest_oversamplers_multiclass',
          'get_all_noisefilters']

def get_all_oversamplers():
    """
    Returns all oversampling classes

    Returns:
        list(OverSampling): list of all oversampling classes

    Example::

        import smote_variants as sv

        oversamplers= sv.get_all_oversamplers()
    """

    return OverSampling.__subclasses__()


def get_n_quickest_oversamplers(n=10):
    """
    Returns the n quickest oversamplers based on testing on the datasets of
    the imbalanced_databases package.

    Args:
        n (int): number of oversamplers to return

    Returns:
        list(OverSampling): list of the n quickest oversampling classes

    Example::

        import smote_variants as sv

        oversamplers= sv.get_n_quickest_oversamplers(10)
    """

    runtimes = {'SPY': 0.11, 'OUPS': 0.16, 'SMOTE_D': 0.20, 'NT_SMOTE': 0.20,
                'Gazzah': 0.21, 'ROSE': 0.25, 'NDO_sampling': 0.27,
                'Borderline_SMOTE1': 0.28, 'SMOTE': 0.28,
                'Borderline_SMOTE2': 0.29, 'ISMOTE': 0.30, 'SMMO': 0.31,
                'SMOTE_OUT': 0.37, 'SN_SMOTE': 0.44, 'Selected_SMOTE': 0.47,
                'distance_SMOTE': 0.47, 'Gaussian_SMOTE': 0.48, 'MCT': 0.51,
                'Random_SMOTE': 0.57, 'ADASYN': 0.58, 'SL_graph_SMOTE': 0.58,
                'CURE_SMOTE': 0.59, 'ANS': 0.63, 'MSMOTE': 0.72,
                'Safe_Level_SMOTE': 0.79, 'SMOBD': 0.80, 'CBSO': 0.81,
                'Assembled_SMOTE': 0.82, 'SDSMOTE': 0.88,
                'SMOTE_TomekLinks': 0.91, 'Edge_Det_SMOTE': 0.94,
                'ProWSyn': 1.00, 'Stefanowski': 1.04, 'NRAS': 1.06,
                'AND_SMOTE': 1.13, 'DBSMOTE': 1.17, 'polynom_fit_SMOTE': 1.18,
                'ASMOBD': 1.18, 'MDO': 1.18, 'SOI_CJ': 1.24, 'LN_SMOTE': 1.26,
                'VIS_RST': 1.34, 'TRIM_SMOTE': 1.36, 'LLE_SMOTE': 1.62,
                'SMOTE_ENN': 1.86, 'SMOTE_Cosine': 2.00, 'kmeans_SMOTE': 2.43,
                'MWMOTE': 2.45, 'V_SYNTH': 2.59, 'A_SUWO': 2.81,
                'RWO_sampling': 2.91, 'SMOTE_RSB': 3.88, 'ADOMS': 3.89,
                'SMOTE_IPF': 4.10, 'Lee': 4.16, 'SMOTE_FRST_2T': 4.18,
                'cluster_SMOTE': 4.19, 'SOMO': 4.30, 'DE_oversampling': 4.67,
                'CCR': 4.72, 'NRSBoundary_SMOTE': 5.26, 'AHC': 5.27,
                'ISOMAP_Hybrid': 6.11, 'LVQ_SMOTE': 6.99, 'CE_SMOTE': 7.45,
                'MSYN': 11.92, 'PDFOS': 15.14, 'KernelADASYN': 17.87,
                'G_SMOTE': 19.23, 'E_SMOTE': 19.50, 'SVM_balance': 24.05,
                'SUNDO': 26.21, 'GASMOTE': 31.38, 'DEAGO': 33.39,
                'NEATER': 41.39, 'SMOTE_PSO': 45.12, 'IPADE_ID': 90.01,
                'DSMOTE': 146.73, 'MOT2LD': 149.42, 'Supervised_SMOTE': 195.74,
                'SSO': 215.27, 'DSRBF': 272.11, 'SMOTE_PSOBAT': 324.31,
                'ADG': 493.64, 'AMSCO': 1502.36}

    samplers = get_all_oversamplers()
    samplers = sorted(
        samplers, key=lambda x: runtimes.get(x.__name__, 1e8))

    return samplers[:n]


def get_metric_learning_oversamplers():
    """
    Returns all oversampling classes supporting the use of classifier
    (random forest) induced distance metric for neighborhood calculations
    
    Returns:
        list(OverSampling): the list of all oversampling classes supporting
                            classifier induced distance
    """
    oversamplers = get_all_oversamplers()
    
    return [o for o in oversamplers if 
            OverSampling.cat_metric_learning in o.categories]

def get_all_oversamplers_multiclass(strategy="eq_1_vs_many_successive"):
    """
    Returns all oversampling classes which can be used with the multiclass
    strategy specified

    Args:
        strategy (str): the multiclass oversampling strategy -
                        'eq_1_vs_many_successive'/'equalize_1_vs_many'

    Returns:
        list(OverSampling): list of all oversampling classes which can be used
                            with the multiclass strategy specified

    Example::

        import smote_variants as sv

        oversamplers= sv.get_all_oversamplers_multiclass()
    """

    oversamplers = get_all_oversamplers()

    if (strategy == 'eq_1_vs_many_successive' or
            strategy == 'equalize_1_vs_many'):

        def multiclass_filter(o):
            return ((OverSampling.cat_changes_majority not in o.categories) or
                    ('proportion' in o().get_params()))

        return [o for o in oversamplers if multiclass_filter(o)]
    else:
        raise ValueError(("It is not known which oversamplers work with the"
                          " strategy %s") % strategy)


def get_n_quickest_oversamplers_multiclass(n,
                                           strategy="eq_1_vs_many_successive"):
    """
    Returns the n quickest oversamplers based on testing on the datasets of
    the imbalanced_databases package, and suitable for using the multiclass
    strategy specified.

    Args:
        n (int): number of oversamplers to return
        strategy (str): the multiclass oversampling strategy -
                        'eq_1_vs_many_successive'/'equalize_1_vs_many'

    Returns:
        list(OverSampling): list of n quickest oversampling classes which can
                    be used with the multiclass strategy specified

    Example::

        import smote_variants as sv

        oversamplers= sv.get_n_quickest_oversamplers_multiclass()
    """

    oversamplers = get_all_oversamplers()
    quickest_oversamplers = get_n_quickest_oversamplers(len(oversamplers))

    if (strategy == 'eq_1_vs_many_successive'
            or strategy == 'equalize_1_vs_many'):

        def multiclass_filter(o):
            return ((OverSampling.cat_changes_majority not in o.categories) or
                    ('proportion' in o().get_params()))

        return [o for o in quickest_oversamplers if multiclass_filter(o)][:n]
    else:
        raise ValueError("It is not known which oversamplers work with the"
                         " strategy %s" % strategy)


def get_all_noisefilters():
    """
    Returns all noise filters
    Returns:
        list(NoiseFilter): list of all noise filter classes
    """
    return NoiseFilter.__subclasses__()

"""
Plotting functionalities.
"""

import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..config import suppress_external_warnings

__all__ = ['vector_labeling',
            'plot_raw',
            'plot_oversampled',
            'plot_sampling',
            'plot_sampling_raw',
            'plot_comparison',
            'plot_comparison_raw',
            'SMOTE_VARIANTS_PALETTE',
            'FIG_SIZE_1_1',
            'FIG_SIZE_1_2',
            'FIG_SIZE_1_3',
            'SIZE_SCALE',
            'set_seaborn_theme',
            'relabel_y']

SMOTE_VARIANTS_PALETTE = ['orange', 'olive', 'red']

FIG_SIZE_1_1 = (4.5, 4.5)
FIG_SIZE_1_2 = (10, 4.5)
FIG_SIZE_1_3 = (15, 4.5)

SIZE_SCALE = (100, 300)

def set_seaborn_theme(n_labels):
    """
    Set the seaborn style

    Args:
        n_labels (int): the number of unique class

    Returns:
        dict: the style parameters
    """
    if n_labels <= 3:
        return {'style': 'white',
                'palette': SMOTE_VARIANTS_PALETTE}

    return {'style': 'white'}

def relabel_y(y):
    """
    Relabel the target labels

    Args:
        y (np.array): target labels

    Returns:
        np.array: relabeled labels
    """
    y_new = np.array(['dummy'] * len(y), dtype='object')

    unique_labels = np.unique(y)

    if len(unique_labels) == 2:
        y_new[y == 0] = 'majority'
        y_new[y == 1] = 'minority'
    else:
        y_new[y == 0] = 'majority'
        for idx, label in enumerate(unique_labels):
            if label != 0:
                y_new[y == label] = f'minority {idx}'

    return y_new

def unique_data(X, y, coordinates=(0, 1)):
    """
    Create unique dataset

    Args:
        X (np.array): all training vectors
        y (np.array): all target labels
        coordinates (tuple): the coordinates to use

    Returns:
        pd.DataFrame: the unique data
    """
    coords = [f'coordinate {idx}' for idx in coordinates]

    unified = np.round(np.vstack([X.T, y.T]).T, 4)
    unique, counts = np.unique(unified, return_counts=True, axis=0)
    X_orig, y_orig = unique[:, :-1], unique[:, -1]

    orig_pdf = pd.DataFrame(X_orig[:, coordinates], columns=coords)
    orig_pdf['y'] = y_orig
    orig_pdf['counts'] = counts

    return orig_pdf

def vector_labeling(X, y, X_samp, y_samp, coordinates=(0, 1)):
    """
    Prepares the data for plotting.

    Args:
        X (np.array): the original vectors
        y (np.array): the original target labels
        X_samp (np.array): the new vectors
        y_samp (np.array): the new target labels
        coordinates (tuple): the coordinates to consider

    Returns:
        pd.DataFrame: the data prepared for plotting
    """
    coords = [f'coordinate {idx}' for idx in coordinates]

    orig_pdf = unique_data(X, y, coordinates)
    new_pdf = unique_data(X_samp, y_samp, coordinates)

    orig_pdf = orig_pdf.rename({'counts': 'counts_old'}, axis='columns')

    orig_pdf['indicator_orig'] = 'original'
    new_pdf['indicator_new'] = 'new'

    result = new_pdf.merge(orig_pdf, how='outer', on=(coords + ['y']))

    result.loc[(result['counts_old'] == result['counts']) \
                & (result['indicator_orig'] == 'original') \
                & (result['indicator_new'] == 'new'), 'flag'] = 'untouched'
    result.loc[(result['indicator_orig'].isna()) \
                & (result['indicator_new'] == 'new'), 'flag'] = 'new'
    result.loc[(result['indicator_orig'] == 'original') \
                & (result['indicator_new'].isna()), 'flag'] = 'removed'
    result.loc[(result['counts_old'] != result['counts']) \
                & (result['indicator_orig'] == 'original') \
                & (result['indicator_new'] == 'new'), 'flag'] = 'count_changed'

    result['removed_flag'] = result['flag'] == 'removed'
    result['original_flag'] = result['indicator_orig'] == 'original'
    result['resampled_flag'] = result['flag'].isin(['new', 'untouched'])
    result['new_flag'] = result['flag'] == 'new'
    result['untouched_flag'] = result['flag'] == 'untouched'
    result['count_changed_flag'] = result['flag'] == 'count_changed'

    return result

def plot_raw(X, y, title="", coordinates=(0, 1)):
    """
    Plot a raw distribution

    Args:
        X (np.array): the training vectors
        y (np.array): the target labels
        title (str): the title
        coordinates (tuple): the coordinates to plot

    Returns:
        obj: the figure
    """
    with warnings.catch_warnings():
        if suppress_external_warnings():
            warnings.simplefilter("ignore")

        sns.set_theme(**set_seaborn_theme(len(np.unique(y))))

        fig, axis = plt.subplots(1, 1, figsize=FIG_SIZE_1_1)

        columns = [f'coordinate {idx}' for idx in coordinates]

        data = pd.DataFrame(X[:, coordinates], columns=columns)
        data['y'] = relabel_y(y)

        order = sorted(np.unique(data['y']))

        sns.scatterplot(data=data,
                        x=columns[0],
                        y=columns[1],
                        hue="y",
                        hue_order=order,
                        legend=True,
                        s=SIZE_SCALE[0])
        axis.set_title(title)

        sns.move_legend(axis, "upper left", bbox_to_anchor=(1.0, 1.0))

    return fig

def plot_oversampled(X, y, X_samp, y_samp, *, title="", coordinates=(0, 1)):
    """
    Plot a raw distribution

    Args:
        X (np.array): the training vectors
        y (np.array): the target labels
        X_samp (np.array): the oversampled vectors
        y_samp (np.array): the oversampled labels
        title (str): the title
        coordinates (tuple): the coordinates to plot

    Returns:
        obj: the figure
    """
    _ = y

    with warnings.catch_warnings():
        if suppress_external_warnings():
            warnings.simplefilter("ignore")

        sns.set_theme(**set_seaborn_theme(len(np.unique(y))))

        fig, axis = plt.subplots(1, 1, figsize=FIG_SIZE_1_1)

        columns = [f'coordinate {idx}' for idx in coordinates]

        data = pd.DataFrame(X_samp[:, coordinates], columns=columns)
        data['y'] = y_samp
        data['flag'] = 'original'
        data.loc[X.shape[0]:, 'flag'] = 'new'
        data['y'] = relabel_y(data['y'].values)

        order = np.unique(data['y'].values)

        sns.scatterplot(data=data,
                        x=columns[0],
                        y=columns[1],
                        hue="y",
                        hue_order=order,
                        style='flag',
                        legend=True,
                        s=SIZE_SCALE[0])
        axis.set_title(title)

        sns.move_legend(axis, "upper left", bbox_to_anchor=(1.0, 1.0))

    return fig

def plot_sampling(labeling, title=""):
    """
    Plots the data.

    Args:
        labeling (pd.DataFrame): data prepared for plotting
        title (str): the title

    Returns:
        obj: the figure
    """

    coords = [c for c in labeling.columns if c.startswith('coordinate')]

    with warnings.catch_warnings():
        if suppress_external_warnings():
            warnings.simplefilter("ignore")

        sizes = SIZE_SCALE

        labeling['y'] = relabel_y(labeling['y'].values)

        order = sorted(np.unique(labeling['y'].values))

        sns.set_theme(**set_seaborn_theme(len(order)))

        fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_1_2)

        axis = sns.scatterplot(data=labeling[labeling['original_flag']],
                                x=coords[0],
                                y=coords[1],
                                hue="y",
                                size="counts_old",
                                hue_order=order,
                                sizes=sizes, legend=False, ax=axes[0])
        axis.set_title("Original dataset")

        axis = sns.scatterplot(data=labeling[labeling['flag'] != 'removed'],
                                x=coords[0],
                                y=coords[1],
                                hue="y",
                                size="counts",
                                hue_order=order,
                                style='flag',
                                sizes=sizes,
                                legend='brief',
                                ax=axes[1],
                                style_order=['untouched', 'new', 'count_changed'])
        sns.scatterplot(data=labeling[labeling['flag'] == 'removed'],
                        x=coords[0],
                        y=coords[1],
                        hue="y",
                        hue_order=order,
                        legend=False,
                        alpha=0.3,
                        ax=axes[1])

        sns.move_legend(axis, "upper left", bbox_to_anchor=(1.0, 1.0))

        plt.title(title)

    return fig

def plot_sampling_raw(*, X, y, X_samp, y_samp, title, coordinates=(0, 1)):
    """
    Plot the sampling from the raw data

    Args:
        X (np.array): original training vectors
        y (np.array): original target labels
        X_samp (np.array): oversampled vectors
        y_samp (np.array): oversampled labels
        title (str): the title of the oversampling
        coordinates (tuple): the coordinates to plot
    """
    labeling = vector_labeling(X, y, X_samp, y_samp, coordinates=coordinates)

    return plot_sampling(labeling, title)

def plot_comparison(labeling0, labeling1, title0, title1):
    """
    Plot the results of two oversamplers against a base one

    Args:
        labeling0 (pd.DataFrame): the vector labeling of the first comparison
        labeling1 (pd.DataFrame): the vector labeling of the second comparison
        title0 (str): title of the first plot
        title1 (str): title of the second plot

    Returns:
        obj: the figure
    """
    coords = [c for c in labeling0.columns if c.startswith('coordinate')]

    labeling0['y'] = relabel_y(labeling0['y'].values)
    labeling1['y'] = relabel_y(labeling1['y'].values)

    with warnings.catch_warnings():
        if suppress_external_warnings():
            warnings.simplefilter("ignore")

        sizes = SIZE_SCALE

        order = sorted(np.unique(labeling0['y']))

        sns.set_theme(**set_seaborn_theme(len(order)))

        fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE_1_3)

        axis = sns.scatterplot(data=labeling0[labeling0['original_flag']],
                                x=coords[0],
                                y=coords[1],
                                hue="y",
                                size="counts_old",
                                hue_order=order,
                                sizes=sizes, legend=False, ax=axes[0])
        axis.set_title("Original dataset")

        axis = sns.scatterplot(data=labeling0[labeling0['flag'] != 'removed'],
                                x=coords[0],
                                y=coords[1],
                                hue="y",
                                size="counts",
                                hue_order=order,
                                style='flag',
                                sizes=sizes,
                                legend=False,
                                ax=axes[1],
                                style_order=['untouched', 'new', 'count_changed'])
        sns.scatterplot(data=labeling0[labeling0['flag'] == 'removed'],
                        x=coords[0],
                        y=coords[1],
                        hue="y",
                        hue_order=order,
                        legend=False,
                        alpha=0.3,
                        ax=axes[1])

        axis.set_title(title0)

        axis = sns.scatterplot(data=labeling1[labeling1['flag'] != 'removed'],
                                x=coords[0],
                                y=coords[1],
                                hue="y",
                                size="counts",
                                hue_order=order,
                                style='flag',
                                sizes=sizes,
                                legend='brief',
                                ax=axes[2],
                                style_order=['untouched', 'new', 'count_changed'])
        sns.scatterplot(data=labeling1[labeling1['flag'] == 'removed'],
                        x=coords[0],
                        y=coords[1],
                        hue="y",
                        hue_order=order,
                        legend=False,
                        alpha=0.3,
                        ax=axes[2])

        sns.move_legend(axis, "upper left", bbox_to_anchor=(1.0, 1.0))
        axis.set_title(title1)

    return fig

def plot_comparison_raw(*, X, y,
                        X_samp0, # pylint: disable=invalid-name
                        y_samp0,
                        X_samp1, # pylint: disable=invalid-name
                        y_samp1,
                        title0,
                        title1,
                        coordinates=(0, 1)):
    """
    Plot the results of two oversamplers against a base one from raw data

    Args:
        X (np.array): the base vectors
        y (np.array): the target labels
        X_samp0 (np.array): the first oversampling
        y_samp0 (np.array): the first target labels
        X_samp1 (np.array): the second oversampling
        y_samp1 (np.array): the second target labels
        title0 (str): the title of the first plot
        title1 (str): the title of the second plot
        coordinates (list): the coordinates to plot

    Returns:
        obj: the figure
    """
    labeling0 = vector_labeling(X, y, X_samp0, y_samp0, coordinates)
    labeling1 = vector_labeling(X, y, X_samp1, y_samp1, coordinates)

    return plot_comparison(labeling0, labeling1, title0, title1)

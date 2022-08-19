"""
Testing the visualization tools.
"""
import numpy as np

from smote_variants.visualization import (vector_labeling,
                                            plot_raw,
                                            plot_oversampled,
                                            plot_sampling_raw,
                                            plot_comparison_raw,
                                            relabel_y,
                                            set_seaborn_theme)

from smote_variants import SMOTE
from smote_variants.datasets import load_illustration_2_class, load_illustration_3_class

dataset = load_illustration_2_class()

X = dataset['data']
y = dataset['target']

X_samp, y_samp = SMOTE().sample(X, y)


def test_style():
    """
    Testing the setting of the color palette
    """

    assert len(set_seaborn_theme(2)) != len(set_seaborn_theme(5))


def test_relabeling():
    """
    Test the relabeling
    """
    dataset_3 = load_illustration_3_class()

    assert len(np.unique(relabel_y(dataset_3['target']))) == 3

def test_vector_labeling():
    """
    Testing the vector labeling
    """

    assert len(vector_labeling(X, y, X_samp, y_samp)) > 0

def test_plot_raw():
    """
    Testing the raw plotting
    """

    assert plot_raw(X, y) is not None

def test_plot_oversampled():
    """
    Testing the oversampled dataset plotting
    """

    assert plot_oversampled(X, y, X_samp, y_samp) is not None

def test_plot_sampling_raw():
    """
    Testing the plot sampling raw function
    """

    assert plot_sampling_raw(X=X,
                             y=y,
                             X_samp=X_samp,
                             y_samp=y_samp,
                             title="SMOTE") is not None

def test_plot_comparison_raw():
    """
    Testing the plot_comparison_raw function
    """
    X_samp1, y_samp1 = SMOTE(n_neighbors=15).sample(X, y) # pylint: disable=invalid-name

    fig = plot_comparison_raw(X=X, y=y, X_samp0=X_samp, y_samp0=y_samp,
                                X_samp1=X_samp1, y_samp1=y_samp1,
                                title0='SMOTE', title1='SMOTE15')
    assert fig is not None

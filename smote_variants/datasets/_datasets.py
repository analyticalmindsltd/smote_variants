"""
This module contains some sample data to work with.
"""

import numpy as np
from sklearn import datasets

__all__ = ['load_1_dim',
           'load_illustration_2_class',
           'load_illustration_3_class',
           'load_illustration_4_class',
           'load_normal',
           'load_same_num',
           'load_some_min_some_maj',
           'load_1_min_some_maj',
           'load_2_min_some_maj',
           'load_3_min_some_maj',
           'load_4_min_some_maj',
           'load_5_min_some_maj',
           'load_1_min_1_maj',
           'load_repeated',
           'load_all_min_noise',
           'load_separable',
           'load_linearly_dependent',
           'load_alternating',
           'load_high_dim']

X_1_dim = np.array([[1], [2], [3], [4], [5], [6], [7], [8],
                    [9], [10], [11], [12], [13]])
y_1_dim = np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0])

def load_1_dim():
    """
    Load the 1 dim test dataset

    Returns:
        dict: the dataset
    """
    return {'data': X_1_dim,
            'target': y_1_dim,
            'name': '1_dim'}

X_illustration_2_class, y_illustration_2_class = \
                        datasets.make_classification(n_samples=100,
                                                    n_features=3,
                                                    n_informative=2,
                                                    n_redundant=1,
                                                    n_repeated=0,
                                                    n_clusters_per_class=2,
                                                    weights=np.array([0.8, 0.2]),
                                                    random_state=7)

def load_illustration_2_class():
    """
    Load the 2 class illustration dataset

    Returns:
        dict: the dataset
    """
    return {'data': X_illustration_2_class,
            'target': y_illustration_2_class,
            'name': 'illustration_2_class'}

X_illustration_3_class, y_illustration_3_class = \
        datasets.make_classification(n_samples=100,
                                    n_classes=3,
                                    n_features=4,
                                    n_informative=3,
                                    n_redundant=1,
                                    n_repeated=0,
                                    n_clusters_per_class=1,
                                    weights=np.array([0.6, 0.3, 0.1]),
                                    random_state=7)

def load_illustration_3_class():
    """
    Load the 3 class illustration dataset

    Returns:
        dict: the dataset
    """
    return {'data': X_illustration_3_class,
            'target': y_illustration_3_class,
            'name': 'illustration_3_class'}

X_illustration_4_class, y_illustration_4_class = \
        datasets.make_classification(n_samples=100,
                                    n_classes=4,
                                    n_features=4,
                                    n_informative=3,
                                    n_redundant=1,
                                    n_repeated=0,
                                    n_clusters_per_class=1,
                                    weights=np.array([0.6, 0.15, 0.15, 0.1]),
                                    random_state=7)

def load_illustration_4_class():
    """
    Load the 4 class illustration dataset

    Returns:
        dict: the dataset
    """
    return {'data': X_illustration_4_class,
            'target': y_illustration_4_class,
            'name': 'illustration_4_class'}

data_min = np.array([[5.7996138, -0.25574582],
                     [3.0637093,  2.11750874],
                     [4.91444087, -0.72380123],
                     [1.06414164,  0.08694243],
                     [2.59071708,  0.75283568],
                     [3.44834937,  1.46118085],
                     [2.8036378,  0.69553702],
                     [3.57901791,  0.71870743],
                     [3.81529064,  0.62580927],
                     [3.05005506,  0.33290343],
                     [1.83674689,  1.06998465],
                     [2.08574889, -0.32686821],
                     [3.49417022, -0.92155623],
                     [2.33920982, -1.59057568],
                     [1.95332431, -0.84533309],
                     [3.35453368, -1.10178101],
                     [4.20791149, -1.41874985],
                     [2.25371221, -1.45181929],
                     [2.87401694, -0.74746037],
                     [1.84435381,  0.15715329]])

data_maj = np.array([[-1.40972752,  0.07111486],
                     [-1.1873495, -0.20838002],
                     [0.51978825,  2.1631319],
                     [-0.61995016, -0.45111475],
                     [2.6093289, -0.40993063],
                     [-0.06624482, -0.45882838],
                     [-0.28836659, -0.59493865],
                     [0.345051,  0.05188811],
                     [1.75694985,  0.16685025],
                     [0.52901288, -0.62341735],
                     [0.09694047, -0.15811278],
                     [-0.37490451, -0.46290818],
                     [-0.32855088, -0.20893795],
                     [-0.98508364, -0.32003935],
                     [0.07579831,  1.36455355],
                     [-1.44496689, -0.44792395],
                     [1.17083343, -0.15804265],
                     [1.73361443, -0.06018163],
                     [-0.05139342,  0.44876765],
                     [0.33731075, -0.06547923],
                     [-0.02803696,  0.5802353],
                     [0.20885408,  0.39232885],
                     [0.22819482,  2.47835768],
                     [1.48216063,  0.81341279],
                     [-0.6240829, -0.90154291],
                     [0.54349668,  1.4313319],
                     [-0.65925018,  0.78058634],
                     [-1.65006105, -0.88327625],
                     [-1.49996313, -0.99378106],
                     [0.31628974, -0.41951526],
                     [0.64402186,  1.10456105],
                     [-0.17725369, -0.67939216],
                     [0.12000555, -1.18672234],
                     [2.09793313,  1.82636262],
                     [-0.11711376,  0.49655609],
                     [1.40513236,  0.74970305],
                     [2.40025472, -0.5971392],
                     [-1.04860983,  2.05691699],
                     [0.74057019, -1.48622202],
                     [1.32230881, -2.36226588],
                     [-1.00093975, -0.44426212],
                     [-2.25927766, -0.55860504],
                     [-1.12592836, -0.13399132],
                     [0.14500925, -0.89070934],
                     [0.90572513,  1.23923502],
                     [-1.25416346, -1.49100593],
                     [0.51229813,  1.54563048],
                     [-1.36854287,  0.0151081],
                     [0.08169257, -0.69722099],
                     [-0.73737846,  0.42595479],
                     [0.02465411, -0.36742946],
                     [-1.14532211, -1.23217124],
                     [0.98038343,  0.59259824],
                     [-0.20721222,  0.68062552],
                     [-2.21596433, -1.96045872],
                     [-1.20519292, -1.8900018],
                     [0.47189299, -0.4737293],
                     [1.18196143,  0.85320018],
                     [0.03255894, -0.77687178],
                     [0.32485141, -0.34609381]])

X_normal = np.vstack([data_min, data_maj])
y_normal = np.hstack([np.repeat(1, len(data_min)),
                np.repeat(0, len(data_maj))])

def load_normal():
    """
    Load the normal dataset

    Returns:
        dict: the dataset
    """
    return {'data': X_normal,
            'target': y_normal,
            'name': 'normal'}

X_same_num = np.array([[1.0, 1.1],
                      [1.1, 1.2],
                      [1.05, 1.1],
                      [1.1, 1.08],
                      [1.5, 1.6],
                      [1.55, 1.55],
                      [1.5, 1.62],
                      [1.55, 1.51]])

y_same_num = np.array([0, 0, 0, 0, 1, 1, 1, 1])

def load_same_num():
    """
    Load the same number dataset

    Returns:
        dict: the dataset
    """
    return {'data': X_same_num,
            'target': y_same_num,
            'name': 'same_num'}

X_some_min_some_maj =  np.array([[1.0, 1.1],
                      [1.1, 1.2],
                      [1.05, 1.1],
                      [1.08, 1.05],
                      [1.1, 1.08],
                      [1.5, 1.6],
                      [1.55, 1.55],
                      [1.0, 1.1],
                      [1.1, 1.2],
                      [1.05, 1.1],
                      [1.08, 1.05],
                      [1.1, 1.08],
                      [1.5, 1.6],
                      [1.55, 1.55]])

y_some_min_some_maj = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0])

def load_some_min_some_maj():
    """
    Load the some minority some majority dataset

    Returns:
        dict: the dataset
    """
    return {'data': X_some_min_some_maj,
            'target': y_some_min_some_maj,
            'name': 'some_min_some_maj'}

X_3_min_some_maj =  np.array([[1.0, 1.1],
                      [1.1, 1.2],
                      [1.05, 1.1],
                      [1.08, 1.05],
                      [1.1, 1.08],
                      [1.5, 1.6],
                      [1.55, 1.55]])

y_3_min_some_maj = np.array([0, 0, 0, 0, 1, 1, 1])

def load_3_min_some_maj():
    """
    Load the 3 minosity some majority dataset

    Returns:
        dict: the dataset
    """
    return {'data': X_3_min_some_maj,
            'target': y_3_min_some_maj,
            'name': '3_min_some_maj'}

X_2_min_some_maj =  np.array([[1.0, 1.1],
                      [1.1, 1.2],
                      [1.05, 1.1],
                      [1.08, 1.05],
                      [1.1, 1.08],
                      [1.5, 1.6],
                      [1.55, 1.55]])

y_2_min_some_maj = np.array([0, 0, 0, 0, 0, 1, 1])

def load_2_min_some_maj():
    """
    Load the 2 minosity some majority dataset

    Returns:
        dict: the dataset
    """
    return {'data': X_2_min_some_maj,
            'target': y_2_min_some_maj,
            'name': '2_min_some_maj'}

X_4_min_some_maj =  np.array([[1.0, 1.1],
                      [1.1, 1.2],
                      [1.05, 1.1],
                      [1.08, 1.05],
                      [1.1, 1.08],
                      [1.5, 1.6],
                      [1.55, 1.55],
                      [1.75, 1.6],
                      [1.6, 1.55]])

y_4_min_some_maj = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])

def load_4_min_some_maj():
    """
    Load the 4 minosity some majority dataset

    Returns:
        dict: the dataset
    """
    return {'data': X_4_min_some_maj,
            'target': y_4_min_some_maj,
            'name': '4_min_some_maj'}

X_5_min_some_maj =  np.array([[1.0, 1.1],
                      [1.1, 1.2],
                      [1.05, 1.1],
                      [1.08, 1.05],
                      [1.1, 1.08],
                      [1.5, 1.6],
                      [1.55, 1.55],
                      [1.75, 1.6],
                      [1.6, 1.55],
                      [1.5, 1.3],
                      [1.3, 1.2]])

y_5_min_some_maj = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

def load_5_min_some_maj():
    """
    Load the 5 minosity some majority dataset

    Returns:
        dict: the dataset
    """
    return {'data': X_5_min_some_maj,
            'target': y_5_min_some_maj,
            'name': '5_min_some_maj'}

X_repeated =  np.array([[1.0, 1.1],
                      [1.1, 1.2],
                      [1.05, 1.1],
                      [1.0, 1.1],
                      [1.1, 1.2],
                      [1.05, 1.1],
                      [1.08, 1.05],
                      [1.1, 1.08],
                      [1.5, 1.6],
                      [1.55, 1.55],
                      [1.5, 1.6],
                      [1.55, 1.55]])

y_repeated = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

def load_repeated():
    """
    Load the dataset containing repeated entries

    Returns:
        dict: the dataset
    """
    return {'data': X_repeated,
            'target': y_repeated,
            'name': 'repeated'}

X_1_min_some_maj = np.array([[1.0, 1.1],
                      [1.1, 1.2],
                      [1.05, 1.1],
                      [1.08, 1.05],
                      [1.1, 1.08],
                      [1.55, 1.55]])

y_1_min_some_maj = np.array([0, 0, 0, 0, 0, 1])

def load_1_min_some_maj():
    """
    Load the 1 minosity some majority dataset

    Returns:
        dict: the dataset
    """
    return {'data': X_1_min_some_maj,
            'target': y_1_min_some_maj,
            'name': '1_min_some_maj'}

X_1_min_1_maj = np.array([[1.0, 1.1],
                      [1.55, 1.55]])

y_1_min_1_maj = np.array([0, 1])

def load_1_min_1_maj():
    """
    Load the 1 minosity 1 majority dataset

    Returns:
        dict: the dataset
    """
    return {'data': X_1_min_1_maj,
            'target': y_1_min_1_maj,
            'name': '1_min_1_maj'}

X_all_min_noise = np.array([[0.0, 0.1],
                            [1.0, 0.3],
                            [2.0, 0.0],
                            [3.0, 0.1],
                            [4.0, 0.2],
                            [5.0, 0.0],
                            [6.0, 0.1],
                            [7.0, 0.3],
                            [8.0, 0.0],
                            [9.0, 0.1],
                            [10.0, 0.2],
                            [11.0, 0.0],
                            [12.0, 0.1],
                            [13.0, 0.0],
                            [14.0, 0.1],
                            [15.0, 0.0],
                            [16.0, 0.1],
                            [17.0, 0.2],
                            [18.0, 0.0],
                            [19.0, 0.1],
                            [20.0, 0.0],
                            [21.0, 0.2],
                            [22.0, 0.0],
                            [23.0, 0.1],
                            [24.0, 0.0],
                            [25.0, 0.1],
                            [26.0, 0.0],
                            [27.0, 0.1],
                            [28.0, 0.2],
                            [29.0, 0.0]])
y_all_min_noise = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                            0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                            0, 0, 0, 1, 0, 0, 0, 0, 0, 1])

def load_all_min_noise():
    """
    Load the all minority noise dataset

    Returns:
        dict: the dataset
    """
    return {'data': X_all_min_noise,
            'target': y_all_min_noise,
            'name': 'all_min_noise'}

X_separable = np.array([[0.0, 0.1],
                            [0.0, 0.3],
                            [0.0, 0.0],
                            [0.0, 0.1],
                            [0.0, 0.2],
                            [0.0, 0.0],
                            [0.0, 0.1],
                            [0.0, 0.3],
                            [0.0, 0.0],
                            [0.0, 0.1],
                            [0.0, 0.2],
                            [0.0, 0.0],
                            [0.0, 0.1],
                            [0.0, 0.0],
                            [0.0, 0.1],
                            [10.0, 0.0],
                            [10.0, 0.1],
                            [10.0, 0.2],
                            [10.0, 0.0],
                            [10.0, 0.1],
                            [10.0, 0.0]])

y_separable = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

def load_separable():
    """
    Load the separable dataset

    Returns:
        dict: the dataset
    """
    return {'data': X_separable,
            'target': y_separable,
            'name': 'separable'}

X_linearly_dependent = np.vstack([[np.repeat(0.0, 21),
                                    np.arange(21)]]).T

y_linearly_dependent = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

def load_linearly_dependent():
    """
    Load the linearly dependent dataset

    Returns:
        dict: the dataset
    """
    return {'data': X_linearly_dependent,
            'target': y_linearly_dependent,
            'name': 'linearly_dependent'}

X_alternating = np.array([[1, 2], [2, 3], [3, 4], [4, 5],
                    [5, 6], [6, 7], [8, 9], [9, 10], [10, 11],
                    [11, 12], [12, 13], [13, 14], [14, 15],
                    [15, 16], [16, 17]])
y_alternating = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

def load_alternating():
    """
    Load the alternating dataset

    Returns:
        dict: the dataset
    """
    return {'data': X_alternating,
            'target': y_alternating,
            'name': 'alternating'}

np.random.seed(42)
X_high_dim = np.random.normal(size=(20, 40))
y_high_dim = np.hstack([np.repeat(1, 7), np.repeat(0, 13)])

def load_high_dim():
    """
    Load the high dimensional dataset

    Returns:
        dict: the dataset
    """
    return {'data': X_high_dim,
            'target': y_high_dim,
            'name': 'high_dim'}

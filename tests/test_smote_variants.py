#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 19:21:49 2018

@author: gykovacs
"""

import numpy as np

import logging

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.datasets as datasets

import os.path
import os

import smote_variants as sv

import pytest

_logger = logging.getLogger('smote_variants')
_logger.setLevel(logging.WARNING)


def validation(smote, X, y):
    return smote.sample(X, y)


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


class TestBasicOperationAndEdgeCases:

    def test_same_num(self):
        X = np.array([[1.0, 1.1],
                      [1.1, 1.2],
                      [1.05, 1.1],
                      [1.1, 1.08],
                      [1.5, 1.6],
                      [1.55, 1.55],
                      [1.5, 1.62],
                      [1.55, 1.51]])

        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        samplers = sv.get_all_oversamplers()

        for s in samplers:
            logging.info("testing %s" % str(s))
            X_samp, y_samp = validation(s(), X, y)
            assert len(X_samp) > 0

    def test_some_min_some_maj(self):
        X = np.array([[1.0, 1.1],
                      [1.1, 1.2],
                      [1.05, 1.1],
                      [1.08, 1.05],
                      [1.1, 1.08],
                      [1.5, 1.6],
                      [1.55, 1.55]])

        y = np.array([0, 0, 0, 0, 0, 1, 1])

        samplers = sv.get_all_oversamplers()

        for s in samplers:
            logging.info("testing %s" % str(s))
            X_samp, y_samp = validation(s(), X, y)
            assert len(X_samp) > 0

    def test_1_min_some_maj(self):
        X = np.array([[1.0, 1.1],
                      [1.1, 1.2],
                      [1.05, 1.1],
                      [1.08, 1.05],
                      [1.1, 1.08],
                      [1.55, 1.55]])

        y = np.array([0, 0, 0, 0, 0, 1])

        samplers = sv.get_all_oversamplers()

        for s in samplers:
            logging.info("testing %s" % str(s))
            X_samp, y_samp = validation(s(), X, y)
            assert len(X_samp) > 0

    def test_1_min_1_maj(self):
        X = np.array([[1.0, 1.1],
                      [1.55, 1.55]])

        y = np.array([0, 1])

        samplers = sv.get_all_oversamplers()

        for s in samplers:
            logging.info("testing %s" % str(s))
            X_samp, y_samp = validation(s(), X, y)
            assert len(X_samp) > 0

    def test_normal(self):
        X = np.vstack([data_min, data_maj])
        y = np.hstack([np.repeat(1, len(data_min)),
                       np.repeat(0, len(data_maj))])

        samplers = sv.get_all_oversamplers()

        for s in samplers:
            logging.info("testing %s" % str(s))
            X_samp, y_samp = s().sample(X, y)
            assert len(X_samp) > 0

        samplers_plus = [sv.polynom_fit_SMOTE(topology='star'),
                         sv.polynom_fit_SMOTE(topology='bus'),
                         sv.polynom_fit_SMOTE(topology='mesh'),
                         sv.polynom_fit_SMOTE(topology='poly_2'),
                         sv.Stefanowski(strategy='weak_amp'),
                         sv.Stefanowski(strategy='weak_amp_relabel'),
                         sv.Stefanowski(strategy='strong_amp'),
                         sv.G_SMOTE(method='non-linear_2.0'),
                         sv.SMOTE_PSOBAT(method='pso'),
                         sv.AHC(strategy='maj'),
                         sv.AHC(strategy='minmaj'),
                         sv.SOI_CJ(method='jittering'),
                         sv.ADG(kernel='rbf_1'),
                         sv.SMOTE_IPF(voting='consensus'),
                         sv.ASMOBD(smoothing='sigmoid')]

        for s in samplers_plus:
            logging.info("testing %s" % str(s.__class__.__name__))
            X_samp, y_samp = s.sample(X, y)
            assert len(X_samp) > 0

        nf = sv.get_all_noisefilters()

        for n in nf:
            logging.info("testing %s" % str(n))
            X_nf, y_nf = n().remove_noise(X, y)
            assert len(X_samp) > 0

    def test_high_dim(self):
        np.random.seed(42)
        X = np.random.normal(size=(20, 40))
        y = np.hstack([np.repeat(1, 7), np.repeat(0, 13)])

        samplers = sv.get_all_oversamplers()

        for s in samplers:
            logging.info("testing %s" % str(s))
            X_samp, y_samp = s().sample(X, y)
            assert len(X_samp) > 0

        samplers_plus = [sv.polynom_fit_SMOTE(topology='star'),
                         sv.polynom_fit_SMOTE(topology='bus'),
                         sv.polynom_fit_SMOTE(topology='mesh'),
                         sv.polynom_fit_SMOTE(topology='poly_2'),
                         sv.Stefanowski(strategy='weak_amp'),
                         sv.Stefanowski(strategy='weak_amp_relabel'),
                         sv.Stefanowski(strategy='strong_amp'),
                         sv.G_SMOTE(method='non-linear_2.0'),
                         sv.SMOTE_PSOBAT(method='pso'),
                         sv.AHC(strategy='maj'),
                         sv.AHC(strategy='minmaj'),
                         sv.SOI_CJ(method='jittering'),
                         sv.ADG(kernel='rbf_1'),
                         sv.SMOTE_IPF(voting='consensus'),
                         sv.ASMOBD(smoothing='sigmoid')]

        for s in samplers_plus:
            logging.info("testing %s" % str(s.__class__.__name__))
            X_samp, y_samp = s.sample(X, y)
            assert len(X_samp) > 0

        nf = sv.get_all_noisefilters()

        for n in nf:
            logging.info("testing %s" % str(n))
            X_nf, y_nf = n().remove_noise(X, y)
            assert len(X_samp) > 0

    def test_parameters(self):
        np.random.seed(42)
        samplers = sv.get_all_oversamplers()

        for s in samplers:
            logging.info("testing %s" % str(s))
            par_comb = s.parameter_combinations()
            if len(par_comb) > 0:
                original_parameters = np.random.choice(par_comb)
                sampler = s(**original_parameters)
                parameters = sampler.get_params()

                for x in original_parameters:
                    assert parameters[x] == original_parameters[x]


class TestModelSelection:
    def model_selection(self):
        X = np.vstack([data_min, data_maj])
        y = np.hstack([np.repeat(1, len(data_min)),
                       np.repeat(0, len(data_maj))])

        # setting cache path
        cache_path = os.path.join(os.path.expanduser('~'), 'smote_test')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

        # prepare dataset
        dataset = {'data': X, 'target': y, 'name': 'ballpark_data'}

        # instantiating classifiers
        knn_classifier = KNeighborsClassifier()
        dt_classifier = DecisionTreeClassifier()

        # instantiate the validation object
        oversamplers = sv.get_n_quickest_oversamplers(5)
        classifiers = [knn_classifier, dt_classifier]
        samp_obj, cl_obj = sv.model_selection(dataset=dataset,
                                              samplers=oversamplers,
                                              classifiers=classifiers,
                                              cache_path=cache_path,
                                              n_jobs=1)
        
        assert (samp_obj is not None) and (cl_obj is not None)

        results = sv.read_oversampling_results(
            datasets=[dataset], cache_path=cache_path)

        assert len(results) > 0


class TestMultiClass:
    def test_multiclass(self):
        dataset = datasets.load_wine()

        oversampler = sv.MulticlassOversampling(sv.distance_SMOTE())

        X_samp, y_samp = oversampler.sample(dataset['data'], dataset['target'])

        assert len(X_samp) > 0

        oversampler = sv.MulticlassOversampling(
            sv.distance_SMOTE(), strategy='equalize_1_vs_many')

        X_samp, y_samp = oversampler.sample(dataset['data'], dataset['target'])

        assert len(X_samp) > 0


class TestQueries:
    def test_queries(self):
        assert len(sv.get_all_oversamplers()) > 0
        assert len(sv.get_all_noisefilters()) > 0
        assert len(sv.get_n_quickest_oversamplers(5)) == 5
        assert len(sv.get_all_oversamplers_multiclass()) > 0
        assert len(sv.get_n_quickest_oversamplers_multiclass(5)) == 5


class TestMLPWrapper:
    def test_mlp_wrapper(self):
        dataset = datasets.load_wine()
        classifier = sv.MLPClassifierWrapper()
        classifier.fit(dataset['data'], dataset['target'])

        assert classifier is not None


class TestCrossValidation:

    def test_cross_validate(self):
        X = np.vstack([data_min, data_maj])
        y = np.hstack([np.repeat(1, len(data_min)),
                       np.repeat(0, len(data_maj))])

        # setting cache path
        cache_path = os.path.join(os.path.expanduser('~'), 'smote_test')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

        # prepare dataset
        dataset = {'data': X, 'target': y, 'name': 'ballpark_data'}

        # instantiating classifiers
        knn_classifier = KNeighborsClassifier()

        # instantiate the validation object
        results = sv.cross_validate(dataset=dataset,
                                    sampler=sv.SMOTE(),
                                    classifier=knn_classifier)

        assert len(results) > 0

        dataset = datasets.load_wine()

        results = sv.cross_validate(dataset=dataset,
                                    sampler=sv.SMOTE(),
                                    classifier=knn_classifier)

        assert len(results) > 0


class TestReproducibility:

    def test_reproducibility(self):
        X = np.vstack([data_min, data_maj])
        y = np.hstack([np.repeat(1, len(data_min)),
                       np.repeat(0, len(data_maj))])

        samplers = sv.get_all_oversamplers()

        for s in samplers:
            logging.info("testing %s" % str(s))

            X_orig = X.copy()
            y_orig = y.copy()

            X_samp_a, y_samp_a = s(random_state=5).sample(X, y)
            sampler = s(random_state=5)
            X_samp_b, y_samp_b = sampler.sample(X, y)
            X_samp_c, y_samp_c = s(**sampler.get_params()).sample(X, y)

            assert np.array_equal(X_samp_a, X_samp_b)
            assert np.array_equal(X_samp_a, X_samp_c)
            assert np.array_equal(X_orig, X)

        samplers_plus = [sv.polynom_fit_SMOTE(topology='star', random_state=5),
                         sv.polynom_fit_SMOTE(topology='bus', random_state=5),
                         sv.polynom_fit_SMOTE(topology='mesh', random_state=5),
                         sv.polynom_fit_SMOTE(
                             topology='poly_2', random_state=5),
                         sv.Stefanowski(strategy='weak_amp', random_state=5),
                         sv.Stefanowski(
                             strategy='weak_amp_relabel', random_state=5),
                         sv.Stefanowski(strategy='strong_amp', random_state=5),
                         sv.G_SMOTE(method='non-linear_2.0', random_state=5),
                         sv.SMOTE_PSOBAT(method='pso', random_state=5),
                         sv.AHC(strategy='maj', random_state=5),
                         sv.AHC(strategy='minmaj', random_state=5),
                         sv.SOI_CJ(method='jittering', random_state=5),
                         sv.ADG(kernel='rbf_1', random_state=5),
                         sv.SMOTE_IPF(voting='consensus', random_state=5),
                         sv.ASMOBD(smoothing='sigmoid', random_state=5)]

        for s in samplers_plus:
            logging.info("testing %s" % str(s.__class__.__name__))

            X_orig = X.copy()
            y_orig = y.copy()
            X_samp_a, y_samp_a = s.sample(X, y)
            sc = s.__class__(**s.get_params())

            X_samp_b, y_samp_b = sc.sample(X, y)

            assert np.array_equal(X_samp_a, X_samp_b)
            assert np.array_equal(X_orig, X)

        nf = sv.get_all_noisefilters()

        for n in nf:
            logging.info("testing %s" % str(n))
            X_orig, y_orig = X.copy(), y.copy()

            nf = n()
            X_nf_a, y_nf_a = nf.remove_noise(X, y)
            nf_b = n(**nf.get_params())
            X_nf_b, y_nf_b = nf_b.remove_noise(X, y)

            assert np.array_equal(X_nf_a, X_nf_b)
            assert np.array_equal(X_orig, X)


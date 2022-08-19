"""
This module implements the DEAGO method.
"""

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import numpy as np # pylint: disable=wrong-import-position
from sklearn.preprocessing import StandardScaler # pylint: disable=wrong-import-position

import tensorflow # pylint: disable=wrong-import-position

from tensorflow.keras.layers import Input, Dense, GaussianNoise # pylint: disable=wrong-import-position,no-name-in-module
from tensorflow.keras.models import Model # pylint: disable=wrong-import-position,no-name-in-module
from tensorflow.keras.callbacks import EarlyStopping # pylint: disable=wrong-import-position,no-name-in-module

from ..base import OverSampling, coalesce, coalesce_dict # pylint: disable=wrong-import-position
from ._smote import SMOTE # pylint: disable=wrong-import-position

from .._logger import logger # pylint: disable=wrong-import-position
_logger= logger

__all__= ['DEAGO']

class DEAGO(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{deago,
                            author={Bellinger, C. and Japkowicz, N. and
                                        Drummond, C.},
                            booktitle={2015 IEEE 14th International
                                        Conference on Machine Learning
                                        and Applications (ICMLA)},
                            title={Synthetic Oversampling for Advanced
                                        Radioactive Threat Detection},
                            year={2015},
                            volume={},
                            number={},
                            pages={948-953},
                            keywords={radioactive waste;advanced radioactive
                                        threat detection;gamma-ray spectral
                                        classification;industrial nuclear
                                        facilities;Health Canadas national
                                        monitoring networks;Vancouver 2010;
                                        Isotopes;Training;Monitoring;
                                        Gamma-rays;Machine learning algorithms;
                                        Security;Neural networks;machine
                                        learning;classification;class
                                        imbalance;synthetic oversampling;
                                        artificial neural networks;
                                        autoencoders;gamma-ray spectra},
                            doi={10.1109/ICMLA.2015.58},
                            ISSN={},
                            month={Dec}}

    Notes:
        * There is no hint on the activation functions and amounts of noise.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_density_estimation,
                  OverSampling.cat_application,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 e=100, # pylint: disable=invalid-name
                 h=0.3, # pylint: disable=invalid-name
                 sigma=0.1,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
            e (int): number of epochs
            h (float): fraction of number of hidden units
            sigma (float): training noise
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}

        super().__init__(random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0.0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater(e, "e", 1)
        self.check_greater(h, "h", 0)
        self.check_greater(sigma, "sigma", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
        self.ss_params = coalesce_dict(ss_params, ss_params_default)
        self.encoder_params = {'e': e, 'h': h, 'sigma': sigma}
        self.n_jobs = n_jobs

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'e': [40],
                                  'h': [0.1, 0.2, 0.3, 0.4, 0.5],
                                  'sigma': [0.05, 0.1, 0.2]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def execute_autoencoder(self,
                            X_min_normalized, # pylint: disable=invalid-name
                            X_init_normalized # pylint: disable=invalid-name
                            ):
        """
        Execute the autoencoder.

        Args:
            X_min_normalized (np.array): the normalized minority samples
            X_init_normalized (np.array): the normalized generated samples

        Returns:
            np.array: the denoised samples (all)
        """

        n_dim = X_min_normalized.shape[1]
        encoding_d = np.max([2, int(np.rint(n_dim*self.encoder_params['h']))])

        _logger.info("%s: Input dimension: %d, encoding dimension: %d",
                    self.__class__.__name__, n_dim, encoding_d)

        # constructing the autoencoder
        callbacks = [EarlyStopping(monitor='val_loss', patience=2)]

        input_layer = Input(shape=(n_dim,))
        noise = GaussianNoise(self.encoder_params['sigma'])(input_layer)
        encoded = Dense(encoding_d, activation='relu')(noise)
        decoded = Dense(n_dim, activation='linear')(encoded)

        dae = Model(input_layer, decoded)
        dae.compile(optimizer='adadelta', loss='mean_squared_error')
        actual_epochs = np.max([self.encoder_params['e'],
                                int(5000.0/X_min_normalized.shape[0])])

        if X_min_normalized.shape[0] > 10:
            val_num = int(0.2 * X_min_normalized.shape[0])
            X_min_train = X_min_normalized[:-val_num] # pylint: disable=invalid-name
            X_min_val = X_min_normalized[-val_num:] # pylint: disable=invalid-name

            dae.fit(X_min_train,
                    X_min_train,
                    epochs=actual_epochs,
                    validation_data=(X_min_val, X_min_val),
                    callbacks=callbacks,
                    verbose=0)
        else:
            dae.fit(X_min_normalized, X_min_normalized,
                    epochs=actual_epochs, verbose=0)

        return dae.predict(X_init_normalized)

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed.")

        tensorflow.random.set_seed(self._random_state_init)

        # sampling by smote
        X_samp, _ = SMOTE(proportion=self.proportion,
                            n_neighbors=self.n_neighbors,
                            nn_params=self.nn_params,
                            ss_params=self.ss_params,
                            n_jobs=self.n_jobs,
                            random_state=self._random_state_init).sample(X, y)

        # samples to map to the manifold extracted by the autoencoder
        X_init = X_samp[len(X):] # pylint: disable=invalid-name

        # normalizing
        X_min = X[y == self.min_label]
        scaler = StandardScaler()
        X_min_normalized = scaler.fit_transform(X_min) # pylint: disable=invalid-name
        X_init_normalized = scaler.transform(X_init) # pylint: disable=invalid-name

        # mapping the initial samples to the manifold
        samples = scaler.inverse_transform(self.execute_autoencoder(X_min_normalized,
                                                                    X_init_normalized))

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'e': self.encoder_params['e'],
                'h': self.encoder_params['h'],
                'sigma': self.encoder_params['sigma'],
                'ss_params': self.ss_params,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}

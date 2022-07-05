import numpy as np

from sklearn.preprocessing import StandardScaler

from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

from .._logger import logger
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
                 nn_params={},
                 e=100,
                 h=0.3,
                 sigma=0.1,
                 n_jobs=1,
                 random_state=None):
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
            e (int): number of epochs
            h (float): fraction of number of hidden units
            sigma (float): training noise
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0.0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater(e, "e", 1)
        self.check_greater(h, "h", 0)
        self.check_greater(sigma, "sigma", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.e = e
        self.h = h
        self.sigma = sigma
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

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

    def sample(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # ugly hack to get reproducible results from keras with
        # tensorflow backend
        if isinstance(self._random_state_init, int):
            import os
            os.environ['PYTHONHASHSEED'] = str(self._random_state_init)
            #import keras as K
            from tensorflow import keras as K
            np.random.seed(self._random_state_init)
            import random
            random.seed(self._random_state_init)
            # from tensorflow import set_random_seed
            import tensorflow
            try:
                tensorflow.set_random_seed(self._random_state_init)
            except Exception as e:
                tensorflow.random.set_seed(self._random_state_init)
        else:
            seed = 127
            import os
            os.environ['PYTHONHASHSEED'] = str(seed)
            #import keras as K
            from tensorflow import keras as K
            np.random.seed(seed)
            import random
            random.seed(seed)
            # from tensorflow import set_random_seed
            import tensorflow
            try:
                tensorflow.compat.v1.set_random_seed(seed)
            except Exception as e:
                tensorflow.random.set_seed(self._random_state_init)

        from tensorflow.keras import backend as K
        import tensorflow as tf
        try:
            session_conf = tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
            sess = tf.compat.v1.Session(
                graph=tf.compat.v1.get_default_graph(), config=session_conf)
            K.set_session(sess)
        except Exception as e:
            session_conf = tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
            sess = tf.compat.v1.Session(
                graph=tf.compat.v1.get_default_graph(), config=session_conf)
            tf.compat.v1.keras.backend.set_session(sess)

        if not hasattr(self, 'Input'):
            from tensorflow.keras.layers import Input, Dense, GaussianNoise
            from tensorflow.keras.models import Model
            from tensorflow.keras.callbacks import EarlyStopping

            self.Input = Input
            self.Dense = Dense
            self.GaussianNoise = GaussianNoise
            self.Model = Model
            self.EarlyStopping = EarlyStopping

        # sampling by smote
        X_samp, y_samp = SMOTE(proportion=self.proportion,
                               n_neighbors=self.n_neighbors,
                               nn_params=self.nn_params,
                               n_jobs=self.n_jobs,
                               random_state=self._random_state_init).sample(X, y)

        # samples to map to the manifold extracted by the autoencoder
        X_init = X_samp[len(X):]

        if len(X_init) == 0:
            return X.copy(), y.copy()

        # normalizing
        X_min = X[y == self.min_label]
        ss = StandardScaler()
        X_min_normalized = ss.fit_transform(X_min)
        X_init_normalized = ss.transform(X_init)

        # extracting dimensions
        d = len(X[0])
        encoding_d = max([2, int(np.rint(d*self.h))])

        message = "Input dimension: %d, encoding dimension: %d"
        message = message % (d, encoding_d)
        _logger.info(self.__class__.__name__ + ": " + message
                     )

        # constructing the autoencoder
        callbacks = [self.EarlyStopping(monitor='val_loss', patience=2)]

        input_layer = self.Input(shape=(d,))
        noise = self.GaussianNoise(self.sigma)(input_layer)
        encoded = self.Dense(encoding_d, activation='relu')(noise)
        decoded = self.Dense(d, activation='linear')(encoded)

        dae = self.Model(input_layer, decoded)
        dae.compile(optimizer='adadelta', loss='mean_squared_error')
        actual_epochs = max([self.e, int(5000.0/len(X_min))])

        if len(X_min) > 10:
            val_perc = 0.2
            val_num = int(val_perc*len(X_min))
            X_min_train = X_min_normalized[:-val_num]
            X_min_val = X_min_normalized[-val_num:]

            dae.fit(X_min_train,
                    X_min_train,
                    epochs=actual_epochs,
                    validation_data=(X_min_val, X_min_val),
                    callbacks=callbacks,
                    verbose=0)
        else:
            dae.fit(X_min_normalized, X_min_normalized,
                    epochs=actual_epochs, verbose=0)

        # mapping the initial samples to the manifold
        samples = ss.inverse_transform(dae.predict(X_init_normalized))

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'e': self.e,
                'h': self.h,
                'sigma': self.sigma,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

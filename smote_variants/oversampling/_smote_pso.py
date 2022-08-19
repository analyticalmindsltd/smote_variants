"""
This module implements the SMOTE_PSO method.
"""

import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSampling
from .._logger import logger
_logger= logger

__all__= ['SMOTE_PSO']

class SMOTE_PSO(OverSampling):
    """
    References:
        * BibTex::

            @article{smote_pso,
                        title = "PSO-based method for SVM classification on
                                    skewed data sets",
                        journal = "Neurocomputing",
                        volume = "228",
                        pages = "187 - 197",
                        year = "2017",
                        note = "Advanced Intelligent Computing: Theory and
                                    Applications",
                        issn = "0925-2312",
                        doi = "https://doi.org/10.1016/j.neucom.2016.10.041",
                        author = "Jair Cervantes and Farid Garcia-Lamont and
                                    Lisbeth Rodriguez and Asdrúbal López and
                                    José Ruiz Castilla and Adrian Trueba",
                        keywords = "Skew data sets, SVM, Hybrid algorithms"
                        }

    Notes:
        * I find the description of the technique a bit confusing, especially
            on the bounds of the search space of velocities and positions.
            Equations 15 and 16 specify the lower and upper bounds, the lower
            bound is in fact a vector while the upper bound is a distance.
            I tried to implement something meaningful.
        * I also find the setting of accelerating constant 2.0 strange, most
            of the time the velocity will be bounded due to this choice.
        * Also, training and predicting probabilities with a non-linear
            SVM as the evaluation function becomes fairly expensive when the
            number of training vectors reaches a couple of thousands. To
            reduce computational burden, minority and majority vectors far
            from the other class are removed to reduce the size of both
            classes to a maximum of 500 samples. Generally, this shouldn't
            really affect the results as the technique focuses on the samples
            near the class boundaries.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_memetic,
                  OverSampling.cat_uses_classifier,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 k=3,
                 *,
                 nn_params={},
                 eps=0.01,
                 n_pop=10,
                 w=1.0,
                 c1=2.0,
                 c2=2.0,
                 num_it=20,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            k (int): number of neighbors in nearest neighbors component, this
                        is also the multiplication factor of minority support
                        vectors
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            eps (float): use to specify the initially generated support
                            vectors along minority-majority lines
            n_pop (int): size of population
            w (float): intertia constant
            c1 (float): acceleration constant of local optimum
            c2 (float): acceleration constant of population optimum
            num_it (int): number of iterations
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state)
        self.check_greater_or_equal(k, "k", 1)
        self.check_greater(eps, "eps", 0)
        self.check_greater_or_equal(n_pop, "n_pop", 1)
        self.check_greater_or_equal(w, "w", 0)
        self.check_greater_or_equal(c1, "c1", 0)
        self.check_greater_or_equal(c2, "c2", 0)
        self.check_greater_or_equal(num_it, "num_it", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.k = k # pylint: disable=invalid-name
        self.nn_params = nn_params
        self.params = {'eps': eps,
                       'n_pop': n_pop,
                       'w': w,
                       'c1': c1,
                       'c2': c2,
                       'num_it': num_it}
        self.n_jobs = n_jobs

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        return cls.generate_parameter_combinations({'k': [3, 5, 7],
                                                    'eps': [0.01],
                                                    'n_pop': [5],
                                                    'w': [0.5, 1.0],
                                                    'c1': [1.0, 2.0],
                                                    'c2': [1.0, 2.0],
                                                    'num_it': [20]}, raw)

    # def remove_majority(self,
    #                     performance_th,
    #                     nn_params,
    #                     X_scaled, # pylint: disable=invalid-name
    #                     y):
    #     """
    #     Remove majority samples

    #     Args:
    #         performance_th (int): performance threshold
    #         nn_params (dict): nearest neighbor parameters
    #         X_scaled (np.array): scaled training vectors
    #         y (np.array): target labels

    #     Returns:
    #         np.array, np.array: the filtered training vectors and target labels
    #     """
    #     n_maj_to_remove = np.sum(y == self.maj_label) - performance_th

    #     if n_maj_to_remove <= 0:
    #         return X_scaled, y

    #     # if majority samples are to be removed
    #     nnmt= NearestNeighborsWithMetricTensor(n_neighbors=1,
    #                                             n_jobs=self.n_jobs,
    #                                             **nn_params)
    #     nnmt.fit(X_scaled[y == self.min_label])
    #     dist, _ = nnmt.kneighbors(X_scaled)

    #     dist = np.vstack([dist[0], np.arange(X_scaled.shape[0])]).T
    #     dist = dist[y == self.maj_label]
    #     dist = dist[dist[:, 0].argsort()[::-1]]

    #     to_remove = dist[:, 1][:n_maj_to_remove]

    #     # removing the samples
    #     X_scaled = np.delete(X_scaled, to_remove, axis=0)
    #     y = np.delete(y, to_remove)

    #     return X_scaled, y

    # def remove_minority(self,
    #                     performance_th,
    #                     nn_params,
    #                     X_scaled, # pylint: disable=invalid-name
    #                     y):
    #     """
    #     Remove minority samples

    #     Args:
    #         performance_th (int): performance threshold
    #         nn_params (dict): nearest neighbor parameters
    #         X_scaled (np.array): scaled training vectors
    #         y (np.array): target labels

    #     Returns:
    #         np.array, np.array: the filtered training vectors and target labels
    #     """

    #     n_min_to_remove = np.sum(y == self.min_label) - performance_th

    #     if n_min_to_remove <= 0:
    #         return X_scaled, y

    #     # if majority samples are to be removed
    #     nnmt = NearestNeighborsWithMetricTensor(n_neighbors=1,
    #                                             n_jobs=self.n_jobs,
    #                                             **nn_params)
    #     nnmt.fit(X_scaled[y == self.maj_label])

    #     dist, _ = nnmt.kneighbors(X_scaled)
    #     dist = np.vstack([dist[0], np.arange(X_scaled.shape[0])]).T
    #     dist = dist[y == self.min_label]
    #     dist = dist[dist[:, 0].argsort()[::-1]]

    #     to_remove = dist[:, 1][:n_min_to_remove]

    #     # removing the samples
    #     X_scaled = np.delete(X_scaled, to_remove, axis=0)
    #     y = np.delete(y, to_remove)

    #     return X_scaled, y

    def determine_support(self,
                            X_scaled, # pylint: disable=invalid-name
                            y):
        """
        Find the support.

        Args:
            X_scaled (np.array): all training vectors
            y (np.array): all target labels

        Returns:
            np.array, np.array: the minority and majority support vectors
        """
        # fitting SVM to extract initial support vectors
        svc = SVC(kernel='rbf', probability=True,
                  gamma='auto', random_state=self._random_state_init)
        svc.fit(X_scaled, y)

        # extracting the support vectors
        SV_min = np.array([i for i in svc.support_ if y[i] == self.min_label]) # pylint: disable=invalid-name
        SV_maj = np.array([i for i in svc.support_ if y[i] == self.maj_label]) # pylint: disable=invalid-name

        X_SV_min = X_scaled[SV_min] # pylint: disable=invalid-name
        X_SV_maj = X_scaled[SV_maj] # pylint: disable=invalid-name

        return X_SV_min, X_SV_maj

    def determine_indices(self, n_neighbors, nn_params, X_SV_maj, X_SV_min):
        """
        Determine the neighborhood relations.

        Args:
            n_neighbors (int): the number of neighbors to use
            nn_params (dict): nearest neighbors parameters
            X_SV_maj (np.array): majority vectors
            X_SV_min (np.array): minority vectors

        Returns:
            np.array: indices
        """
        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **nn_params)
        nnmt.fit(X_SV_maj)
        _, ind = nnmt.kneighbors(X_SV_min)

        return ind

    def initialize(self,
                    X_scaled, # pylint: disable=invalid-name
                    y,
                    nn_params):
        """
        Initialize the search

        Args:
            X_scaled (np.array): the scaled training vectors
            y (np.array): the target labels
            nn_params (dict): the nearest neighbor parameters

        Returns:
            np.array, np.array, np.array, np.array: the minority generators,
                                                    the search space,
                                                    the initial velocities,
                                                    and the search bounds
        """
        X_SV_min, X_SV_maj = self.determine_support(X_scaled, y) # pylint: disable=invalid-name

        # finding nearest majority support vectors
        n_neighbors = np.min([X_SV_maj.shape[0], self.k])
        ind = self.determine_indices(n_neighbors, nn_params, X_SV_maj, X_SV_min)

        min_vector = X_SV_min[np.repeat(np.arange(len(X_SV_min)), n_neighbors)]
        maj_vector = X_SV_maj[ind.flatten()]
        upper_bound = X_SV_maj[ind[np.repeat(np.arange(len(X_SV_min)), n_neighbors), 0]]
        init_velocity = self.params['eps'] * (maj_vector - min_vector)
        X_min_gen = min_vector + init_velocity # pylint: disable=invalid-name

        search_bound = np.linalg.norm(min_vector - upper_bound, axis=1)

        search_space = np.hstack([min_vector, maj_vector]).reshape(min_vector.shape[0],
                                                                    2,
                                                                    min_vector.shape[1])

        return X_min_gen, search_space, search_bound, init_velocity

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Trains support vector classifier and evaluates it

        Args:
            X_train (np.array): training vectors
            y_train (np.array): target labels
            X_test (np.array): test vectors
            y_test (np.array): test labels
        """
        svc = SVC(kernel='rbf', probability=True,
                  gamma='auto', random_state=self._random_state_init)

        svc.fit(X_train, y_train)
        class_idx = np.where(svc.classes_ == self.min_label)[0][0]
        y_pred = svc.predict_proba(X_test)[:, class_idx]
        return roc_auc_score(y_test, y_pred)

    def evaluate_particle(self,
                            X_scaled, # pylint: disable=invalid-name
                            y,
                            part):
        """
        Evaluate a particle.

        Args:
            X_scaled (np.array): the training vectors
            y (np.array): the target labels
            part (np.array): a particle
        """
        X_extended = np.vstack([X_scaled, part]) # pylint: disable=invalid-name
        y_extended = np.hstack([y, np.repeat(self.min_label, len(part))])
        return self.evaluate(X_extended, y_extended, X_scaled, y)

    def update_velocities(self, *, particle_swarm, velocities,
                            search_bound, local_best, global_best):
        """
        Update the velocities.

        Args:
            particle_swarm (np.array): the particles
            velocities (np.array): the velocities
            search_bound (np.array): the search bounds
            local_best (np.array): the local best results
            global_best (np.array): the global best result

        Returns:
            list: the updated velocities
        """
        # update velocities
        #for idx, part in enumerate(particle_swarm):
        #    term_0 = self.params['w'] * velocities[idx]
        #    rand = self.random_state.random_sample(2)
        #    term_1 = self.params['c1'] * rand[0] * (local_best[idx] - part)
        #    term_2 = self.params['c2'] * rand[1] * (global_best - part)
        #
        #    velocities[idx] = term_0 + term_1 + term_2
        #    #velocities[idx] = term_0

        term_0 = self.params['w'] * velocities
        rand = self.random_state.random_sample((2, velocities.shape[0]))
        term_1 = self.params['c1'] * (local_best - particle_swarm) \
                                                        * rand[0][:, None, None]
        term_2 = self.params['c2'] * (global_best - particle_swarm) \
                                                        * rand[1][:, None, None]

        velocities = term_0 + term_1 + term_2

        velocity_norms = np.linalg.norm(velocities, axis=2)

        mask = (velocity_norms > search_bound[None, :] / 2.0)

        search_bounds = np.tile(search_bound[:, None], (velocity_norms.shape[0],)).T

        multiplier = search_bounds[mask] / velocity_norms[mask] / 2.0

        velocities[mask] = velocities[mask] * multiplier[:, None]

        # bound velocities according to search space constraints
        #for vel in velocities:
        #    for idx, vel_idx in enumerate(vel):
        #        v_i_norm = np.linalg.norm(vel_idx)
        #        if v_i_norm > search_bound[idx]/2.0:
        #            vel[idx] = vel_idx / v_i_norm * search_bound[idx] / 2.0

        return velocities

    def update_positions(self, particle_swarm, velocities, search_space, search_bound):
        """
        Update the positions.

        Args:
            particle_swarm (np.array): the particles
            velocities (np.array): the velocities
            search_space (np.array): the search space
            search_bound (np.array): the search bounds

        Returns:
            np.array: the updated particle swarm
        """
        # update positions
        particle_swarm = particle_swarm + velocities

        trans_vectors = particle_swarm - search_space[:, 0][None, :, :]
        trans_norm = np.linalg.norm(trans_vectors, axis=2)

        mask = trans_norm > search_bound[None, :]

        search_space_0 = np.tile(search_space[:, 0][None, :, :], (particle_swarm.shape[0], 1, 1))

        search_bounds = np.tile(search_bound, (particle_swarm.shape[0], 1))

        multiplier = search_bounds[mask] / trans_norm[mask]

        normed_trans = trans_vectors[mask] * multiplier[:, None]

        particle_swarm[mask] = search_space_0[mask] + normed_trans

        #for idx, part in enumerate(particle_swarm):
        #    particle_swarm[idx] = particle_swarm[idx] + velocities[idx]

        # bound positions according to search space constraints
        #for part in particle_swarm:
        #    for idx, p_idx in enumerate(part):
        #        search_idx = search_space[idx]
        #
        #        trans_vector = p_idx - search_idx[0]
        #        trans_norm = np.linalg.norm(trans_vector)
        #
        #        if trans_norm > search_idx[2]:
        #            normed_trans = trans_vector / trans_norm
        #            part[idx] = search_idx[0] + normed_trans * search_idx[2]

        return particle_swarm

    def update_scores(self, *, X_scaled, y, particle_swarm,
                local_best_scores, local_best, global_best_score, global_best):
        """
        Evaluate and update the scores.

        Args:
            X_scaled (np.array): the training vectors
            y (np.array): the target labels
            particle_swarm (np.array): the particle swarm
            local_best_scores (np.array): the local best scores
            local_best (np.array): the local best particles
            global_best_score (float): the global best score
            global_best (obj): the globally best particle

        Returns:
            np.array, np.array, float, obj: the updated local best scores,
                        local best, global best score and global best
                        particle
        """

        scores = np.array([self.evaluate_particle(X_scaled, y, p)
                      for p in particle_swarm])

        local_mask = scores > local_best_scores
        local_best_scores[local_mask] = scores[local_mask]
        local_best[local_mask] = particle_swarm[local_mask]

        max_idx = np.argmax(scores)
        if scores[max_idx] > global_best_score:
            global_best_score = scores[max_idx]
            global_best = particle_swarm[max_idx]

        # update best scores
        #for idx, score in enumerate(scores):
        #    if score > local_best_scores[idx]:
        #        local_best_scores[idx] = score
        #        local_best[idx] = particle_swarm[idx]
        #    if score > global_best_score:
        #        global_best_score = score
        #        global_best = particle_swarm[idx]

        return local_best_scores, local_best, global_best_score, global_best

    def search(self,
                X_scaled, # pylint: disable=invalid-name
                y,
                nn_params):
        """
        The optimization search.

        Args:
            X_scaled (np.array): the scaled training vectors
            y (np.array): the target label
            nn_params (dict): the nearest neighbor parameters

        Returns:
            np.array: the generated samples
        """
        X_min_gen, search_space, search_bound, init_velocity = \
                    self.initialize(X_scaled, y, nn_params) # pylint: disable=invalid-name

        # initializing the particle swarm and the particle and population level
        # memory
        particle_swarm = np.array([X_min_gen.copy()] * self.params['n_pop'])
        velocities = np.array([init_velocity.copy()] * self.params['n_pop'])
        local_best = np.array([X_min_gen.copy()] * self.params['n_pop'])
        local_best_scores = np.array([0.0]*self.params['n_pop'])
        global_best = X_min_gen.copy()
        global_best_score = 0.0

        for iteration in range(self.params['num_it']):
            _logger.info("%s: Iteration %d", self.__class__.__name__, iteration)
            # evaluate population
            local_best_scores, local_best, global_best_score, global_best = \
                self.update_scores(X_scaled=X_scaled,
                                    y=y,
                                    particle_swarm=particle_swarm,
                                    local_best_scores=local_best_scores,
                                    local_best=local_best,
                                    global_best_score=global_best_score,
                                    global_best=global_best)

            velocities = self.update_velocities(particle_swarm=particle_swarm,
                                                velocities=velocities,
                                                search_bound=search_bound,
                                                local_best=local_best,
                                                global_best=global_best)

            particle_swarm = self.update_positions(particle_swarm,
                                                    velocities,
                                                    search_space,
                                                    search_bound)
        return global_best

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        if np.sum(y == self.min_label) >= np.sum(y == self.maj_label):
            return self.return_copies(X, y, "Sampling is not needed.")

        # saving original dataset
        X_orig = X
        y_orig = y

        # scaling the records
        #mms = MinMaxScaler()
        mms = StandardScaler()
        X_scaled = mms.fit_transform(X) # pylint: disable=invalid-name

        # removing majority and minority samples far from the training data if
        # needed to increase performance
        #performance_th = 500

        nn_params = {**self.nn_params}
        nn_params['metric_tensor'] = \
                self.metric_tensor_from_nn_params(nn_params, X_scaled, y)

        #X_scaled, y = self.remove_majority(performance_th, nn_params, X_scaled, y)

        #X_scaled, y = self.remove_minority(performance_th, nn_params, X_scaled, y)

        new_items = [np.zeros(shape=(0, X.shape[1]))]

        while np.sum(y == self.min_label) < np.sum(y == self.maj_label):
            global_best = self.search(X_scaled, y, nn_params)
            new_items.append(global_best)
            X_scaled = np.vstack([X_scaled, global_best]) # pylint: disable=invalid-name
            y = np.hstack([y, np.repeat(self.min_label, len(global_best))])

        X_new = np.vstack(new_items)
        X_ret = np.vstack([X_orig, mms.inverse_transform(X_new)]) # pylint: disable=invalid-name
        y_ret = np.hstack([y_orig, np.repeat(self.min_label, len(X_new))])

        return (X_ret, y_ret)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'k': self.k,
                'nn_params': self.nn_params,
                'eps': self.params['eps'],
                'n_pop': self.params['n_pop'],
                'w': self.params['w'],
                'c1': self.params['c1'],
                'c2': self.params['c2'],
                'num_it': self.params['num_it'],
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}

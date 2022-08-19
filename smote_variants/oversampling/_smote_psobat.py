"""
This module implements the SMOTE_PSOBAT method.
"""

import numpy as np

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from ..base import coalesce, coalesce_dict
from ..base import OverSampling
from ._smote import SMOTE

from .._logger import logger
_logger= logger

__all__= ['SMOTE_PSOBAT']

class SMOTE_PSOBAT(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{smote_psobat,
                            author={Li, J. and Fong, S. and Zhuang, Y.},
                            booktitle={2015 3rd International Symposium on
                                        Computational and Business
                                        Intelligence (ISCBI)},
                            title={Optimizing SMOTE by Metaheuristics with
                                    Neural Network and Decision Tree},
                            year={2015},
                            volume={},
                            number={},
                            pages={26-32},
                            keywords={data mining;particle swarm
                                        optimisation;pattern classification;
                                        data mining;classifier;metaherustics;
                                        SMOTE parameters;performance
                                        indicators;selection optimization;
                                        PSO;particle swarm optimization
                                        algorithm;BAT;bat-inspired algorithm;
                                        metaheuristic optimization algorithms;
                                        nearest neighbors;imbalanced dataset
                                        problem;synthetic minority
                                        over-sampling technique;decision tree;
                                        neural network;Classification
                                        algorithms;Neural networks;Decision
                                        trees;Training;Optimization;Particle
                                        swarm optimization;Data mining;SMOTE;
                                        Swarm Intelligence;parameter
                                        selection optimization},
                            doi={10.1109/ISCBI.2015.12},
                            ISSN={},
                            month={Dec}}

    Notes:
        * The parameters of the memetic algorithms are not specified.
        * I have checked multiple paper describing the BAT algorithm, but the
            meaning of "Generate a new solution by flying randomly" is still
            unclear.
        * It is also unclear if best solutions are recorded for each bat, or
            the entire population.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_memetic,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 *,
                 maxit=50,
                 c1=0.3,
                 c2=0.1,
                 c3=0.1,
                 alpha=0.9,
                 gamma=0.9,
                 method='bat',
                 nn_params=None,
                 ss_params=None,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            maxit (int): maximum number of iterations
            c1 (float): intertia weight of PSO
            c2 (float): attraction of local maximums in PSO
            c3 (float): attraction of global maximum in PSO
            alpha (float): alpha parameter of the method
            gamma (float): gamma parameter of the method
            method (str): optimization technique to be used
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}

        super().__init__(random_state=random_state)
        self.check_greater_or_equal(maxit, "maxit", 1)
        self.check_greater_or_equal(c1, "c1", 0)
        self.check_greater_or_equal(c2, "c2", 0)
        self.check_greater_or_equal(c3, "c3", 0)
        self.check_greater_or_equal(alpha, "alpha", 0)
        self.check_greater_or_equal(gamma, "gamma", 0)
        self.check_isin(method, "method", ['pso', 'bat'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.params = {'maxit': maxit,
                        'c1': c1,
                        'c2': c2,
                        'c3': c3,
                        'alpha': alpha,
                        'gamma': gamma,
                        'n_pop': 10,
                        'n_keep': 5,
                        'f_max': 10}
        self.method = method
        self.nn_params = coalesce(nn_params, {})
        self.ss_params = coalesce_dict(ss_params, ss_params_default)
        self.n_jobs = n_jobs

        # a reasonable range of proportions
        self.proportion_range = np.array([0.1, 2.0])
        self.k_range = np.array([1, 10])

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        bat_pc = cls.generate_parameter_combinations({'maxit': [50],
                                                      'alpha': [0.7, 0.9],
                                                      'gamma': [0.7, 0.9],
                                                      'method': ['bat']}, raw)
        pso_pc = cls.generate_parameter_combinations({'maxit': [50],
                                                      'c1': [0.2, 0.5],
                                                      'c2': [0.1, 0.2],
                                                      'c3': [0.1, 0.2],
                                                      'method': ['pso']}, raw)
        if not raw:
            bat_pc.extend(pso_pc)
            return bat_pc

        return {'maxit': [50],
                'alpha': [0.7, 0.9],
                'gamma': [0.7, 0.9],
                'method': ['bat', 'pso'],
                'c1': [0.2, 0.5],
                'c2': [0.1, 0.2],
                'c3': [0.1, 0.2]}

    def pred_test_vectors(self, *, proportion, K, nn_params, X, y):
        """
        Determine the prediction and test vectors.

        Args:
            proportion (float): the proportion to sample
            K (int): the number of neighbors
            nn_params (dict): the nearest neighbor parameters
            X (np.array): all training vectors
            y (np.array): all target labels

        Returns:
            np.array, np.array: the predictions and the test labels
        """
        ss_params = self.ss_params
        ss_params['n_dim'] = np.min([ss_params['n_dim'], int(K)])

        X_samp, y_samp = SMOTE(proportion=proportion,
                            n_neighbors=K,
                            nn_params=nn_params,
                            ss_params=ss_params,
                            n_jobs=self.n_jobs,
                            random_state=self._random_state_init).sample(X, y)

        # doing k-fold cross validation
        kfold = KFold(5)
        preds = []
        tests = []
        for train, test in kfold.split(X_samp):
            dtree = DecisionTreeClassifier(random_state=self._random_state_init)
            dtree.fit(X_samp[train], y_samp[train])
            preds.append(dtree.predict(X_samp[test]))
            tests.append(y_samp[test])
        preds = np.hstack(preds)
        tests = np.hstack(tests)

        return preds, tests

    def calculate_scores(self, truep, truen, falsep, falsen):
        """
        Calculate the target scores.

        Args:
            truep (int): the number of true positives
            truen (int): the number of true negatives
            falsep (int): the number of false positives
            falsen (int): the number of false negatives

        Returns:
            float, float: kappa, accuracy
        """
        p_o = (truep + truen) / (truep + falsen + truen + falsep)
        term_0 = (truep + falsen) * (truep + falsep) \
                / (truep + falsen + truen + falsep)**2
        term_1 = (falsep + truen)*(falsen + truen) \
                / (truep + falsen + truen + falsep)**2
        p_e =  term_0 + term_1

        kappa = (p_o - p_e)/(1.0 - p_e)

        return kappa, p_o

    def evaluate(self, *, K, proportion, nn_params, X, y):
        """
        Evaluate given configuration

        Args:
            K (int): number of neighbors in nearest neighbors component
            proportion (float): proportion of missing data to generate
            nn_params (dict): nearest neighbor parameters
            X (np.array): the training vectors
            y (np.array): the target labels

        Returns:
            float, float: kappa and accuracy scores
        """
        preds, tests = self.pred_test_vectors(proportion=proportion,
                                                K=K,
                                                nn_params=nn_params,
                                                X=X,
                                                y=y)

        # computing the kappa score
        truep = np.sum(np.logical_and(preds == tests,
                                    tests == self.min_label))
        falsen = np.sum(np.logical_and(preds != tests,
                                    tests == self.min_label))
        truen = np.sum(np.logical_and(preds == tests,
                                    tests == self.maj_label))
        falsep = np.sum(np.logical_and(preds != tests,
                                    tests == self.maj_label))

        return self.calculate_scores(truep, truen, falsep, falsen)

    def init_particle_or_bat(self):
        """
        Initialize a PSO particle.

        Args:
            k_range (tuple): the range of k values
            proportion_range (tuple): the range of proportion values

        Returns:
            np.array: an initial particle
        """
        k_rand = self.random_state.randint(self.k_range[0], self.k_range[1])
        rand = self.random_state.random_sample()
        diff = self.proportion_range[1] - self.proportion_range[0]
        vect = rand * diff + self.proportion_range[0]
        return np.array([k_rand, vect])

    def init_population(self):
        """
        Initialize the population.

        Returns:
            np.array: the initial population
        """
        return np.array([self.init_particle_or_bat() \
                            for _ in range(self.params['n_pop'])])

    def pso_update_local_scores(self,
                                scores,
                                particles,
                                local_scores,
                                local_best):
        """
        Update the local scores according to the rule in the paper.

        Args:
            scores (np.array): the new scores
            particles (np.array): the particles
            local_scores (np.array): the local best scores
            local_best (np.array): the local best particles

        Returns:
            np.array, np.array, bool: local scores, local best particles,
                                        not changed flag
        """
        not_changed = True

        mask = ((np.min(np.vstack([local_scores[:, 0],
                                  scores[:, 0]]).T, axis=1) > 0.4)
                                & (local_scores[:, 1] > scores[:, 1]))

        local_scores[mask] = scores[mask].copy()
        local_best[mask] = particles[mask].copy()
        if np.any(mask):
            not_changed = False

        mask = (~mask) & (scores[:, 0] > 0.4) & (local_scores[:, 0] <= 0.4)
        local_scores[mask] = scores[mask].copy()
        local_best[mask] = particles[mask].copy()

        if np.any(mask):
            not_changed = False

        return local_scores, local_best, not_changed

    def pso_update_global_scores(self,
                                    scores,
                                    particles,
                                    global_scores,
                                    global_best):
        """
        Update global scores for PSO

        Args:
            scores (np.array): the actual scores
            particles (np.array): the particles
            global_scores (np.array): the global scores
            global_best (np.array): the globally best particle

        Returns:
            np.array, np.array, bool: the global best score, best particle,
                                        the not changed flag
        """
        not_changed = True
        mask = ((np.min(np.vstack([np.repeat(global_scores[0], len(scores)),
                                                scores[:, 0]]).T, axis=1) > 0.4)
                    & (global_scores[1] > scores[:, 1]))
        if np.any(mask):
            max_idx = np.argmax(scores[mask, 1])
            global_scores = scores[mask][max_idx].copy()
            global_best = particles[mask][max_idx].copy()
            not_changed = False

        mask = (~mask) & (scores[:, 0] > 0.4) & (global_scores[0] <= 0.4)
        if np.any(mask):
            max_idx = np.argmax(scores[mask, 1])
            global_scores = scores[mask][max_idx].copy()
            global_best = particles[mask][max_idx].copy()
            not_changed = False

        return (global_scores, global_best, not_changed)

    def pso_update_velocities(self, particles, velocities, local_best, global_best):
        """
        Update the PSO velocities

        Args:
            particles (np.array): the particles
            velocities (np.array): the actual velocities
            local_best (np.array): the locally best particles
            global_best (np.array): the globally best particle

        Returns:
            np.array: the updated velocities
        """
        # update velocities
        velocities = self.params['c1'] * velocities \
                        + (local_best - particles) * self.params['c2'] \
                        + (global_best - particles) * self.params['c3']

        ratios = np.abs(velocities[:, 0]) / (self.k_range[1] - self.k_range[0])
        velocities[ratios > 1.0, 0] = \
                velocities[ratios > 1.0, 0] / np.ceil(ratios[ratios > 1.0])

        proportion_diff = self.proportion_range[1] - self.proportion_range[0]
        ratios = np.abs(velocities[:, 1]) / proportion_diff
        velocities[ratios > 1.0, 1] = \
                velocities[ratios > 1.0, 1] / np.ceil(ratios[ratios > 1.0])

        return velocities

    def pso_vectorized(self, X, y, nn_params):
        """
        PSO optimization

        Args:
            X (np.array): all training vectors
            y (np.array): all target labels
            nn_params (dict): the nearest neighbor parameters

        Returns:
            np.array: the best K and proportion values
        """

        # initial particles

        particles = self.init_population()
        # initial velocities
        velocities = np.array([np.array([0, 0])
                                        for _ in range(self.params['n_pop'])])

        local_best = np.array([particles[i].copy()
                                        for i in range(self.params['n_pop'])])
        local_scores = np.zeros((self.params['n_pop'], 2))
        global_best = particles[0].copy()
        global_scores = np.array([0.0, 0.0])

        # executing the particle swarm optimization
        not_changed = 0
        for _ in range(self.params['maxit']):
            # if the configurations didn't change for 10 iterations, stop
            if not_changed > len(particles):
                break

            not_changed = not_changed + 1

            scores = np.array([self.evaluate(K=int(particle[0]),
                                        proportion=particle[1],
                                        nn_params=nn_params,
                                        X=X, y=y) for particle in particles])

            (local_scores, local_best, changed) = \
                self.pso_update_local_scores(scores,
                                                particles,
                                                local_scores,
                                                local_best)
            (global_scores, global_best, changed) = \
                self.pso_update_global_scores(scores,
                                                particles,
                                                global_scores,
                                                global_best)

            not_changed = not_changed * changed

            velocities = self.pso_update_velocities(particles,
                                                    velocities,
                                                    local_best,
                                                    global_best)

            # update positions
            particles = particles + velocities
            particles[:, 0] = np.clip(particles[:, 0],
                                        self.k_range[0],
                                        self.k_range[1])
            particles[:, 1] = np.clip(particles[:, 1],
                                        self.proportion_range[0],
                                        self.proportion_range[1])

        return global_best

    def bat_check_global(self, scores, global_scores):
        """
        Check the global performances for the bat optimization

        Args:
            scores (np.array): the actual scores
            global_scores (np.array): the globally best scores

        Returns:
            int: the index maximizing the global score or -1 if no improvement
                 happened
        """
        max_idx = -1
        tmp0 = np.repeat(global_scores[0], len(scores))
        tmp1 = scores[:, 0]

        mask = ((np.min(np.vstack([tmp0, tmp1]).T, axis=1) > 0.4)
                    & (global_scores[1] > scores[:, 1]))
        if np.any(mask):
            max_idx = np.argmax(scores[mask, 1])

        mask = (~mask) & (scores[:, 0] > 0.4) & (global_scores[0] <= 0.4)
        if np.any(mask):
            max_idx = np.argmax(scores[mask, 1])

        return max_idx

    def bat_check_local(self, scores, local_scores):
        """
        Check the performance locally.

        Args:
            scores (np.array): the actual scores
            local_scores (np.array): the local scores

        Returns:
            np.array: the mask of particles that improved
        """
        local_scores = np.array([local[0] for local in local_scores])
        mask = ((np.min(np.vstack([local_scores[:, 0], scores[:, 0]]).T, axis=1) > 0.4)
                                & (local_scores[:, 1] > scores[:, 1]))

        mask = (~mask) & (scores[:, 0] > 0.4) & (local_scores[:, 0] <= 0.4)

        return mask

    def bat_local_search(self, bats, local_best, local_scores, pulse_loudness):
        """
        The local search algorithm.

        Args:
            bats (np.array): the bats
            local_best (np.array): the locally best bats
            local_scores (np.array): the local scores
            pulse_loudness (np.array): the pulse and loudness parameters

        Returns:
            np.array: the updated bats
        """
        n_pop = self.params['n_pop']

        local_lens = np.sum(local_scores[:, :, 0] >= 0.0, axis=1)
        mask = self.random_state.random_sample(n_pop) > pulse_loudness[:, 1]

        random_best_idx = np.floor(local_lens \
                * self.random_state.random_sample(n_pop)).astype(int)
        random_best = local_best[np.arange(n_pop), random_best_idx][mask]
        rand = (self.random_state.random_sample(size=random_best.shape) - 0.5) * 2.0
        rand = rand / 20.0
        rand[:, 0] = rand[:, 0] * (self.k_range[1] - self.k_range[0])
        rand[:, 1] = rand[:, 1] * (self.proportion_range[1] - self.proportion_range[0])
        bats[mask] = random_best + rand * np.mean(pulse_loudness[:, 2])

        bats[:, 0] = np.clip(bats[:, 0],
                                self.k_range[0],
                                self.k_range[1])
        bats[:, 1] = np.clip(bats[:, 1],
                                self.proportion_range[0],
                                self.proportion_range[1])

        return bats

    def bat_update_local(self, *, bats, scores, local_best, local_scores, improved_local):
        """
        Update the local scores for the bat optimization

        Args:
            bats (np.array): the bats
            scores (np.array): the actual scores
            local_best (np.array): the locally best bats
            local_scores (np.array): the local scores
            improved_local (np.array): the mask of particles that improved locally

        Returns:
            np.array, np.array: the updated local scores, and the locally best bats
        """
        local_lens = np.sum(local_scores[:, :, 0] >= 0.0, axis=1)
        local_lens[improved_local] = np.minimum(local_lens[improved_local] + 1,
                                                self.params['n_keep'])
        local_best[improved_local, local_lens[improved_local]] = \
                                                        bats[improved_local]
        local_scores[improved_local, local_lens[improved_local]] = \
                                                        scores[improved_local]

        sorting = local_scores[:,:,0].argsort(axis=1)[:,::-1]
        sorting = sorting + \
                        (np.arange(self.params['n_pop']) * self.params['n_keep'])[:, None]
        n_total = self.params['n_pop']*self.params['n_keep']

        local_scores = local_scores.reshape(n_total, 2)[sorting]
        local_scores = local_scores.reshape(self.params['n_pop'],
                                            self.params['n_keep'],
                                            2)

        local_best = local_best.reshape(n_total, 2)[sorting]
        local_best = local_best.reshape(self.params['n_pop'],
                                        self.params['n_keep'],
                                        2)

        return local_scores, local_best

    def bat_update_pulse_loudness(self,
                                    pulse_loudness,
                                    improved_local,
                                    iteration):
        """
        Update the pulse and loudness values.

        Args:
            pulse_loudness (np.array): the pulse and loudness values
            improved_local (np.array): the mask of bats that improved locally
            iteration (int): the iteration number

        Returns:
            np.array: the updated pulse-loudness parameters
        """
        random = self.random_state.random_sample(self.params['n_pop'])
        mask = random[improved_local] < pulse_loudness[improved_local, 1]
        pulse_loudness[improved_local, 2][mask] = \
                pulse_loudness[improved_local, 2][mask] * self.params['alpha']
        pulse_loudness[improved_local, 1][mask] = \
                pulse_loudness[improved_local, 1][mask] \
                    * (1 - np.exp(-self.params['gamma'] * iteration))

        return pulse_loudness

    def bat_update_local_scores(self,
                                    *,
                                    bats,
                                    scores,
                                    local_best,
                                    local_scores,
                                    pulse_loudness,
                                    iteration):
        """
        Update the local scores in the bat algorithm.

        Args:
            bats (np.array): all bats
            scores (np.array): the actual scores
            local_best (np.array): the locally best bats
            local_scores (np.array): the local scores
            pulse_loudness (np.array): the pulse-loudness parameters
            iteration (int): the iteration number

        Returns:
            np.array, np.array, np.array, bool: the updated locally best bats,
                                                local scores, pulse-loudness
                                                values and the flag indicating
                                                any update
        """
        improved_local = self.bat_check_local(scores, local_scores)
        local_scores, local_best = self.bat_update_local(bats=bats,
                                                        scores=scores,
                                                        local_best=local_best,
                                                        local_scores=local_scores,
                                                        improved_local=improved_local)
        pulse_loudness = self.bat_update_pulse_loudness(pulse_loudness,
                                                        improved_local,
                                                        iteration)

        return local_best, local_scores, pulse_loudness, np.any(improved_local)

    def bat_update_global_scores(self,
                                    *,
                                    bats,
                                    scores,
                                    global_best,
                                    global_scores,
                                    pulse_loudness):
        """
        Update the global scores in the bat algorithm.

        Args:
            bats (np.array): the bats
            scores (np.array): the actual scores
            global_best (np.array): the globally best bat
            global_scores (np.array): the globally best scores
            pulse_loudness (np.array): the pulse-loudness parameters

        Returns:
            np.array, np.array, bool: the updated globally best bat, globally
                                        best scores and the improvement flag
        """
        improved_global = self.bat_check_global(scores, global_scores)

        rand = self.random_state.random_sample()

        if (improved_global >= 0 and rand < pulse_loudness[improved_global, 2]):
            global_best = bats[improved_global]
            global_scores = scores[improved_global]

        return global_best, global_scores, improved_global >= 0

    def bat_vectorized(self, X, y, nn_params):
        """
        BAT optimization

        Args:
            X (np.array): all training vectors
            y (np.array): all target labels
            nn_params (dict): the nearest neighbor parameters

        Returns:
            np.array: the best K and proportion values
        """

        # initial bat positions
        bats = self.init_population()
        # initial velocities
        velocities = np.zeros(shape=(self.params['n_pop'], 2))

        # best configurations of particles
        local_best = np.array([[bats[i].copy()]*self.params['n_keep']
                                for i in range(len(bats))])

        local_scores = np.zeros(shape=(self.params['n_pop'], self.params['n_keep'], 2))
        local_scores[:, :, :] = -1

        # scores of best configurations of particles
        global_best = bats[0].copy()
        global_scores = np.array([0.0, 0.0])

        # columns: pulse frequency, pulse rate, loudness
        pulse_loudness = self.random_state.random_sample(size=(self.params['n_pop'], 3))

        not_changed = 0
        for iteration in range(self.params['maxit']):
            not_changed = not_changed + 1

            if not_changed > 10:
                break

            # update frequencies
            pulse_loudness[:, 0] = \
                self.random_state.random_sample(size=self.params['n_pop']) \
                                                            * self.params['f_max']

            velocities = velocities + (bats - global_best) \
                                            * pulse_loudness[:, 0, None]

            bats = bats + velocities

            bats = self.bat_local_search(bats,
                                            local_best,
                                            local_scores,
                                            pulse_loudness)

            scores = np.array([self.evaluate(K=int(bat[0]),
                                             proportion=bat[1],
                                             nn_params=nn_params,
                                             X=X, y=y) for bat in bats])

            (local_best, local_scores, pulse_loudness, changed) = \
                self.bat_update_local_scores(bats=bats,
                                                scores=scores,
                                                local_best=local_best,
                                                local_scores=local_scores,
                                                pulse_loudness=pulse_loudness,
                                                iteration=iteration)

            global_best, global_scores, changed = \
                    self.bat_update_global_scores(bats=bats,
                                                    scores=scores,
                                                    global_best=global_best,
                                                    global_scores=global_scores,
                                                    pulse_loudness=pulse_loudness)

            not_changed = not_changed * changed

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

        nn_params = {**self.nn_params}
        nn_params['metric_tensor'] = \
                self.metric_tensor_from_nn_params(nn_params, X, y)

        # a reasonable range of nearest neighbors to use with SMOTE
        self.k_range = np.array([1, np.min([np.sum(y == self.min_label),
                                            self.k_range[1]])])

        if self.method == 'pso':
            best_combination = self.pso_vectorized(X, y, nn_params)
        elif self.method == 'bat':
            best_combination = self.bat_vectorized(X, y, nn_params)

        ss_params = self.ss_params
        ss_params['n_dim'] = np.min([ss_params['n_dim'], int(best_combination[0])])

        return SMOTE(proportion=best_combination[1],
                     n_neighbors=int(best_combination[0]),
                     nn_params=nn_params,
                     ss_params=ss_params,
                     n_jobs=self.n_jobs,
                     random_state=self._random_state_init).sample(X, y)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'maxit': self.params['maxit'],
                'c1': self.params['c1'],
                'c2': self.params['c2'],
                'c3': self.params['c3'],
                'alpha': self.params['alpha'],
                'gamma': self.params['gamma'],
                'method': self.method,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}

import numpy as np

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from .._metric_tensor import MetricTensor
from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

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
                 nn_params={},
                 n_jobs=1,
                 random_state=None):
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
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(maxit, "maxit", 1)
        self.check_greater_or_equal(c1, "c1", 0)
        self.check_greater_or_equal(c2, "c2", 0)
        self.check_greater_or_equal(c3, "c3", 0)
        self.check_greater_or_equal(alpha, "alpha", 0)
        self.check_greater_or_equal(gamma, "gamma", 0)
        self.check_isin(method, "method", ['pso', 'bat'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.maxit = maxit
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.alpha = alpha
        self.gamma = gamma
        self.method = method
        self.nn_params = nn_params
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

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
        else:
            bat_pc = {**bat_pc, **pso_pc}
        return bat_pc

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

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        def evaluate(K, proportion):
            """
            Evaluate given configuration

            Args:
                K (int): number of neighbors in nearest neighbors component
                proportion (float): proportion of missing data to generate

            Returns:
                float, float: kappa and accuracy scores
            """
            smote = SMOTE(proportion=proportion,
                          n_neighbors=K,
                          nn_params=nn_params,
                          n_jobs=self.n_jobs,
                          random_state=self._random_state_init)
            X_samp, y_samp = smote.sample(X, y)

            # doing k-fold cross validation
            kfold = KFold(5)
            preds = []
            tests = []
            for train, test in kfold.split(X_samp):
                dt = DecisionTreeClassifier(random_state=self._random_state_init)
                dt.fit(X_samp[train], y_samp[train])
                preds.append(dt.predict(X_samp[test]))
                tests.append(y_samp[test])
            preds = np.hstack(preds)
            tests = np.hstack(tests)
            # computing the kappa score
            tp = np.sum(np.logical_and(preds == tests,
                                       tests == self.min_label))
            fn = np.sum(np.logical_and(preds != tests,
                                       tests == self.min_label))
            tn = np.sum(np.logical_and(preds == tests,
                                       tests == self.maj_label))
            fp = np.sum(np.logical_and(preds != tests,
                                       tests == self.maj_label))

            p_o = (tp + tn)/(tp + fn + tn + fp)
            p_e = (tp + fn)*(tp + fp)/(tp + fn + tn + fp)**2 + \
                (fp + tn)*(fn + tn)/(tp + fn + tn + fp)**2

            kappa = (p_o - p_e)/(1.0 - p_e)

            return kappa, p_o

        def PSO():
            """
            PSO optimization

            Returns:
                int, float: the best K and proportion values
            """
            # a reasonable range of nearest neighbors to use with SMOTE
            k_range = [2, min([np.sum(y == self.min_label), 10])]
            # a reasonable range of proportions
            proportion_range = [0.1, 2.0]
            # population size
            n_pop = 10

            # initial particles
            def init_particle():
                k_rand = self.random_state.randint(k_range[0], k_range[1])
                r = self.random_state.random_sample()
                diff = proportion_range[1] - proportion_range[0]
                vect = r*diff + proportion_range[0]
                return np.array([k_rand, vect])
            ps = [init_particle() for _ in range(n_pop)]
            # initial velocities
            velocities = [np.array([0, 0]) for _ in range(n_pop)]
            # best configurations of particles
            local_best = [ps[i].copy() for i in range(n_pop)]
            # scores of best configurations of particles
            local_scores = [(0, 0) for _ in range(n_pop)]
            # global best configuration of particles
            global_best = ps[0].copy()
            # global best score
            global_scores = (0, 0)

            # executing the particle swarm optimization
            not_changed = 0
            for _ in range(self.maxit):
                # if the configurations didn't change for 10 iterations, stop
                if not_changed > len(ps)*10:
                    break
                # evaluating each of the configurations
                for i in range(len(ps)):
                    scores = evaluate(np.int(ps[i][0]), ps[i][1])
                    # recording if the best scores didn't change
                    not_changed = not_changed + 1
                    # registering locally and globally best scores
                    if (min([local_scores[i][0], scores[0]]) > 0.4
                            and local_scores[i][1] > scores[1]):
                        local_scores[i] = scores
                        local_best[i] = ps[i].copy()
                        not_changed = 0
                    elif scores[0] > 0.4 and local_scores[i][0] <= 0.4:
                        local_scores[i] = scores
                        local_best[i] = ps[i].copy()
                        not_changed = 0

                    if (min([global_scores[0], scores[0]]) > 0.4
                            and global_scores[1] > scores[1]):
                        global_scores = scores
                        global_best = ps[i].copy()
                        not_changed = 0
                    elif scores[0] > 0.4 and global_scores[0] <= 0.4:
                        global_scores = scores
                        global_best = ps[i].copy()
                        not_changed = 0

                # update velocities
                for i in range(len(ps)):
                    velocities[i] = self.c1*velocities[i] + \
                        (local_best[i] - ps[i])*self.c2 + \
                        (global_best - ps[i])*self.c3
                    # clipping velocities if required
                    while abs(velocities[i][0]) > k_range[1] - k_range[0]:
                        velocities[i][0] = velocities[i][0]/2.0
                    diff = proportion_range[1] - proportion_range[0]
                    while abs(velocities[i][1]) > diff:
                        velocities[i][1] = velocities[i][1]/2.0

                # update positions
                for i in range(len(ps)):
                    ps[i] = ps[i] + velocities[i]
                    # clipping positions according to the specified ranges
                    ps[i][0] = np.clip(ps[i][0], k_range[0], k_range[1])
                    ps[i][1] = np.clip(ps[i][1],
                                       proportion_range[0],
                                       proportion_range[1])

            return global_best

        def BAT():
            """
            BAT optimization

            Returns:
                int, float: the best K and proportion values
            """

            if sum(y == self.min_label) < 2:
                return X.copy(), y.copy()

            # a reasonable range of nearest neighbors to use with SMOTE
            k_range = [1, min([np.sum(y == self.min_label), 10])]
            # a reasonable range of proportions
            proportion_range = [0.1, 2.0]
            # population size
            n_pop = 10
            # maximum frequency
            f_max = 10

            def init_bat():
                k_rand = self.random_state.randint(k_range[0], k_range[1])
                r = self.random_state.random_sample()
                diff = proportion_range[1] - proportion_range[0]
                return np.array([k_rand, r*diff + proportion_range[0]])

            # initial bat positions
            bats = [init_bat() for _ in range(n_pop)]
            # initial velocities
            velocities = [np.array([0, 0]) for _ in range(10)]
            # best configurations of particles
            local_best = [[[[0.0, 0.0], bats[i].copy()]]
                          for i in range(len(bats))]
            # scores of best configurations of particles
            global_best = [[0.0, 0.0], bats[0].copy()]
            # pulse frequencies
            f = self.random_state.random_sample(size=n_pop)*f_max
            # pulse rates
            r = self.random_state.random_sample(size=n_pop)
            # loudness
            A = self.random_state.random_sample(size=n_pop)

            # gamma parameter according to the BAT paper
            gamma = self.gamma
            # alpha parameter according to the BAT paper
            alpha = self.alpha

            # initial best solution
            bat_star = bats[0].copy()

            not_changed = 0
            for t in range(self.maxit):
                not_changed = not_changed + 1

                if not_changed > 10:
                    break

                # update frequencies
                f = self.random_state.random_sample(size=n_pop)*f_max

                # update velocities
                for i in range(len(velocities)):
                    velocities[i] = velocities[i] + (bats[i] - bat_star)*f[i]

                # update bats
                for i in range(len(bats)):
                    bats[i] = bats[i] + velocities[i]
                    bats[i][0] = np.clip(bats[i][0], k_range[0], k_range[1])
                    bats[i][1] = np.clip(
                        bats[i][1], proportion_range[0], proportion_range[1])

                for i in range(n_pop):
                    # generate local solution
                    if self.random_state.random_sample() > r[i]:
                        n_rand = min([len(local_best[i]), 5])
                        rand_int = self.random_state.randint(n_rand)
                        random_best_sol = local_best[i][rand_int][1]
                        rr = self.random_state.random_sample(
                            size=len(bat_star))
                        bats[i] = random_best_sol + rr*A[i]

                # evaluate and do local search
                for i in range(n_pop):
                    scores = evaluate(int(bats[i][0]), bats[i][1])

                    # checking if the scores are better than the global score
                    # implementation of the multi-objective criterion in the
                    # SMOTE-PSOBAT paper
                    improved_global = False
                    if (min([global_best[0][0], scores[0]]) > 0.4
                            and global_best[0][1] > scores[1]):
                        improved_global = True
                        not_changed = 0
                    elif scores[0] > 0.4 and global_best[0][0] <= 0.4:
                        improved_global = True
                        not_changed = 0

                    # checking if the scores are better than the local scores
                    # implementation of the multi-objective criterion in the
                    # SMOTE-PSOBAT paper
                    improved_local = False
                    if (min([local_best[i][0][0][0], scores[0]]) > 0.4
                            and local_best[i][0][0][1] > scores[1]):
                        improved_local = True
                    elif scores[0] > 0.4 and local_best[i][0][0][0] <= 0.4:
                        improved_local = True

                    # local search in the bet algorithm
                    if (self.random_state.random_sample() < A[i]
                            and improved_local):
                        local_best[i].append([scores, bats[i].copy()])
                        A[i] = A[i]*alpha
                        r[i] = r[i]*(1 - np.exp(-gamma*t))
                    if (self.random_state.random_sample() < A[i]
                            and improved_global):
                        global_best = [scores, bats[i].copy()]

                    # ranking local solutions to keep track of the best 5
                    local_best[i] = sorted(
                        local_best[i], key=lambda x: -x[0][0])
                    local_best[i] = local_best[i][:min(
                        [len(local_best[i]), 5])]

                t = t + 1

            return global_best[1]

        if self.method == 'pso':
            best_combination = PSO()
        elif self.method == 'bat':
            best_combination = BAT()
        else:
            message = "Search method %s not supported yet." % self.method
            raise ValueError(self.__class__.__name__ + ": " + message)

        return SMOTE(proportion=best_combination[1],
                     n_neighbors=int(best_combination[0]),
                     nn_params=nn_params,
                     n_jobs=self.n_jobs,
                     random_state=self._random_state_init).sample(X, y)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'maxit': self.maxit,
                'c1': self.c1,
                'c2': self.c2,
                'c3': self.c3,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'method': self.method,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

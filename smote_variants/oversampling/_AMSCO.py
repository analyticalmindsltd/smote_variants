import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

from .._metric_tensor import MetricTensor
from ._OverSampling import OverSampling

from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['AMSCO']

class AMSCO(OverSampling):
    """
    References:
        * BibTex::

            @article{amsco,
                        title = "Adaptive multi-objective swarm fusion for
                                    imbalanced data classification",
                        journal = "Information Fusion",
                        volume = "39",
                        pages = "1 - 24",
                        year = "2018",
                        issn = "1566-2535",
                        doi = "https://doi.org/10.1016/j.inffus.2017.03.007",
                        author = "Jinyan Li and Simon Fong and Raymond K.
                                    Wong and Victor W. Chu",
                        keywords = "Swarm fusion, Swarm intelligence
                                    algorithm, Multi-objective, Crossover
                                    rebalancing, Imbalanced data
                                    classification"
                        }

    Notes:
        * It is not clear how the kappa threshold is used, I do use the RA
            score to drive all the evolution. Particularly:

            "In the last phase of each iteration, the average Kappa value
            in current non-inferior set is compare with the latest threshold
            value, the threshold is then increase further if the average value
            increases, and vice versa. By doing so, the non-inferior region
            will be progressively reduced as the Kappa threshold lifts up."

        I don't see why would the Kappa threshold lift up if the kappa
        thresholds are decreased if the average Kappa decreases ("vice versa").

        * Due to the interpretation of kappa threshold and the lack of detailed
            description of the SIS process, the implementation is not exactly
            what is described in the paper, but something very similar.
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_memetic,
                  OverSampling.cat_uses_classifier,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 *,
                 n_pop=5,
                 n_iter=15,
                 omega=0.1,
                 r1=0.1,
                 r2=0.1,
                 nn_params={},
                 n_jobs=1,
                 classifier=DecisionTreeClassifier(random_state=2),
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            n_pop (int): size of populations
            n_iter (int): optimization steps
            omega (float): intertia of PSO
            r1 (float): force towards local optimum
            r2 (float): force towards global optimum
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
        self.check_greater_or_equal(n_pop, "n_pop", 1)
        self.check_greater_or_equal(n_iter, "n_iter", 1)
        self.check_greater_or_equal(omega, "omega", 0)
        self.check_greater_or_equal(r1, "r1", 0)
        self.check_greater_or_equal(r2, "r2", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_pop = n_pop
        self.n_iter = n_iter
        self.omega = omega
        self.r1 = r1
        self.r2 = r2
        self.nn_params = nn_params
        self.n_jobs = n_jobs
        self.classifier = classifier

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        # as the method is an overall optimization, 1 reasonable settings
        # should be enough

        classifiers = [DecisionTreeClassifier(random_state=2)]
        parameter_combinations = {'n_pop': [5],
                                  'n_iter': [15],
                                  'omega': [0.1],
                                  'r1': [0.1],
                                  'r2': [0.1],
                                  'classifier': classifiers}

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

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        n_cross_val = min([4, len(X_min)])
        
        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)
        
        def fitness(X_min, X_maj):
            """
            Calculating fitness function

            Args:
                X_min (np.matrix): minority samples
                X_maj (np.matrix): majority samples

            Returns:
                float, float: kappa, accuracy
            """
            kfold = StratifiedKFold(n_cross_val)

            # prepare assembled dataset
            X_ass = np.vstack([X_min, X_maj])
            y_ass = np.hstack([np.repeat(self.min_label, len(X_min)),
                               np.repeat(self.maj_label, len(X_maj))])

            preds = []
            tests = []
            for train, test in kfold.split(X_ass, y_ass):
                self.classifier.fit(X_ass[train], y_ass[train])
                preds.append(self.classifier.predict(X))
                tests.append(y)
            preds = np.hstack(preds)
            tests = np.hstack(tests)

            # calculate kappa and accuracy scores
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
            accuracy = (tp + tn)/(tp + fn + tn + fp)

            return kappa, accuracy

        def OSMOTE(X_min, X_maj):
            """
            Executing OSMOTE phase

            Args:
                X_min (np.matrix): minority samples
                X_maj (np.matrix): majority samples

            Returns:
                np.matrix, np.matrix: new minority and majority datasets
            """

            # initialize particles, first coordinate represents proportion
            # parameter of SMOTE
            # the second coordinate represents the number of neighbors to
            # take into consideration
            def init_pop():
                proportion = self.random_state.random_sample()/2.0+0.5
                n_neighbors = self.random_state.randint(3, 10)
                return np.array([proportion, n_neighbors])
            particles = [init_pop() for _ in range(self.n_pop)]
            # velocities initialized
            velocities = [np.array([0.1, 1]) for _ in range(self.n_pop)]
            # setting the limits of the search space
            limits = [np.array([0.25, 3]), np.array([4.0, 10])]
            # local best results
            local_best = [particles[i].copy() for i in range(self.n_pop)]
            # local best scores
            local_score = [(0.0, 0.0)]*self.n_pop
            # global best result
            global_best = particles[0].copy()
            # global best score
            global_score = (0.0, 0.0)
            # best dataset
            best_dataset = None

            # running the optimization
            for _ in range(self.n_iter):
                # update velocities
                for i in range(len(velocities)):
                    diff1 = (local_best[i] - velocities[i])
                    diff2 = (global_best - velocities[i])
                    velocities[i] = (velocities[i]*self.omega +
                                     self.r1 * diff1 + self.r2*diff2)
                    # clipping velocities using the upper bounds of the
                    # particle search space
                    velocities[i][0] = np.clip(
                        velocities[i][0], -limits[1][0]/2, limits[1][0]/2)
                    velocities[i][1] = np.clip(
                        velocities[i][1], -limits[1][1]/2, limits[1][1]/2)

                # update particles
                for i in range(len(particles)):
                    particles[i] = particles[i] + velocities[i]
                    # clipping the particle positions using the lower and
                    # upper bounds
                    particles[i][0] = np.clip(
                        particles[i][0], limits[0][0], limits[1][0])
                    particles[i][1] = np.clip(
                        particles[i][1], limits[0][1], limits[1][1])

                # evaluate
                scores = []
                for i in range(len(particles)):
                    # apply SMOTE
                    smote = SMOTE(particles[i][0],
                                  int(np.rint(particles[i][1])),
                                  nn_params=nn_params,
                                  n_jobs=self.n_jobs,
                                  random_state=self._random_state_init)
                    X_to_sample = np.vstack([X_maj, X_min])
                    y_to_sample_maj = np.repeat(
                        self.maj_label, len(X_maj))
                    y_to_sample_min = np.repeat(
                        self.min_label, len(X_min))
                    y_to_sample = np.hstack([y_to_sample_maj, y_to_sample_min])
                    X_samp, _ = smote.sample(X_to_sample, y_to_sample)

                    # evaluate
                    scores.append(fitness(X_samp[len(X_maj):],
                                          X_samp[:len(X_maj)]))

                    # update scores according to the multiobjective setting
                    if (scores[i][0]*scores[i][1] >
                            local_score[i][0]*local_score[i][1]):
                        local_best[i] = particles[i].copy()
                        local_score[i] = scores[i]
                    if (scores[i][0]*scores[i][1] >
                            global_score[0]*global_score[1]):
                        global_best = particles[i].copy()
                        global_score = scores[i]
                        best_dataset = (X_samp[len(X_maj):],
                                        X_samp[:len(X_maj)])

            return best_dataset[0], best_dataset[1]

        def SIS(X_min, X_maj):
            """
            SIS procedure

            Args:
                X_min (np.matrix): minority dataset
                X_maj (np.matrix): majority dataset

            Returns:
                np.matrix, np.matrix: new minority and majority datasets
            """
            min_num = len(X_min)
            max_num = len(X_maj)
            if min_num >= max_num:
                return X_min, X_maj

            # initiate particles
            def init_particle():
                num = self.random_state.randint(min_num, max_num)
                maj = self.random_state.choice(np.arange(len(X_maj)), num)
                return maj

            particles = [init_particle() for _ in range(self.n_pop)]
            scores = [fitness(X_min, X_maj[particles[i]])
                      for i in range(self.n_pop)]
            best_score = (0.0, 0.0)
            best_dataset = None

            for _ in range(self.n_iter):
                # mutate and evaluate
                # the way mutation or applying PSO is not described in the
                # paper in details
                for i in range(self.n_pop):
                    # removing some random elements
                    domain = np.arange(len(particles[i]))
                    n_max = min([10, len(particles[i])])
                    n_to_choose = self.random_state.randint(0, n_max)
                    to_remove = self.random_state.choice(domain, n_to_choose)
                    mutant = np.delete(particles[i], to_remove)

                    # adding some random elements
                    maj_set = set(np.arange(len(X_maj)))
                    part_set = set(particles[i])
                    diff = list(maj_set.difference(part_set))
                    n_max = min([10, len(diff)])
                    n_to_choose = self.random_state.randint(0, n_max)
                    diff_elements = self.random_state.choice(diff, n_to_choose)
                    mutant = np.hstack([mutant, np.array(diff_elements)])
                    # evaluating the variant
                    score = fitness(X_min, X_maj[mutant])
                    if score[1] > scores[i][1]:
                        particles[i] = mutant.copy()
                        scores[i] = score
                    if score[1] > best_score[1]:
                        best_score = score
                        best_dataset = mutant.copy()

            return X_min, X_maj[best_dataset]

        # executing the main optimization procedure
        current_min = X_min
        current_maj = X_maj
        for it in range(self.n_iter):
            _logger.info(self.__class__.__name__ + ": " +
                         'staring iteration %d' % it)
            new_min, _ = OSMOTE(X_min, current_maj)
            _, new_maj = SIS(current_min, X_maj)

            # calculating fitness values of the four combinations
            fitness_0 = np.prod(fitness(new_min, current_maj))
            fitness_1 = np.prod(fitness(current_min, current_maj))
            fitness_2 = np.prod(fitness(new_min, new_maj))
            fitness_3 = np.prod(fitness(current_min, new_maj))

            # selecting the new current_maj and current_min datasets
            message = 'fitness scores: %f %f %f %f'
            message = message % (fitness_0, fitness_1, fitness_2, fitness_3)
            _logger.info(self.__class__.__name__ + ": " + message)
            max_fitness = np.max([fitness_0, fitness_1, fitness_2, fitness_3])
            if fitness_1 == max_fitness or fitness_3 == max_fitness:
                current_maj = new_maj
            if fitness_0 == max_fitness or fitness_2 == max_fitness:
                current_min = new_min

        return (np.vstack([current_maj, current_min]),
                np.hstack([np.repeat(self.maj_label, len(current_maj)),
                           np.repeat(self.min_label, len(current_min))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'n_pop': self.n_pop,
                'n_iter': self.n_iter,
                'omega': self.omega,
                'r1': self.r1,
                'r2': self.r2,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'classifier': self.classifier,
                'random_state': self._random_state_init}

import numpy as np

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['GASMOTE']

class GASMOTE(OverSampling):
    """
    References:
        * BibTex::

            @Article{gasmote,
                        author="Jiang, Kun
                        and Lu, Jing
                        and Xia, Kuiliang",
                        title="A Novel Algorithm for Imbalance Data
                                Classification Based on Genetic
                                Algorithm Improved SMOTE",
                        journal="Arabian Journal for Science and
                                    Engineering",
                        year="2016",
                        month="Aug",
                        day="01",
                        volume="41",
                        number="8",
                        pages="3255--3266",
                        issn="2191-4281",
                        doi="10.1007/s13369-016-2179-2",
                        url="https://doi.org/10.1007/s13369-016-2179-2"
                        }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_memetic,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 maxn=7,
                 n_pop=10,
                 popl3=5,
                 pm=0.3,
                 pr=0.2,
                 Ge=10,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            n_neighbors (int): number of neighbors
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            maxn (int): maximum number of samples to generate per minority
                        instances
            n_pop (int): size of population
            popl3 (int): number of crossovers
            pm (float): mutation probability
            pr (float): selection probability
            Ge (int): number of generations
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(maxn, "maxn", 1)
        self.check_greater_or_equal(n_pop, "n_pop", 1)
        self.check_in_range(pm, "pm", [0, 1])
        self.check_in_range(pr, "pr", [0, 1])
        self.check_greater_or_equal(Ge, "Ge", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.maxn = maxn
        self.n_pop = n_pop
        self.popl3 = popl3
        self.pm = pm
        self.pr = pr
        self.Ge = Ge
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        return cls.generate_parameter_combinations({'n_neighbors': [7],
                                                    'maxn': [2, 3, 4],
                                                    'n_pop': [10],
                                                    'popl3': [4],
                                                    'pm': [0.3],
                                                    'pr': [0.2],
                                                    'Ge': [10]}, raw)

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

        # fitting nearest neighbors model to find minority neighbors of
        #  minority samples
        n_neighbors = min([self.n_neighbors + 1, len(X_min)])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                        n_jobs=self.n_jobs, 
                                                        **(nn_params))
        nn.fit(X_min)
        ind = nn.kneighbors(X_min, return_distance=False)
        kfold = KFold(min([len(X), 5]))

        def fitness(conf):
            """
            Evluate fitness of configuration

            Args:
                conf (list(list)): configuration
            """
            # generate new samples
            samples = []
            for i in range(len(conf)):
                for _ in range(conf[i]):
                    X_b = X_min[self.random_state.choice(ind[i][1:])]
                    samples.append(self.sample_between_points(X_min[i], X_b))

            if len(samples) == 0:
                # if no samples are generated
                X_new = X
                y_new = y
            else:
                # construct dataset
                X_new = np.vstack([X, np.vstack(samples)])
                y_new = np.hstack(
                    [y, np.repeat(self.min_label, len(samples))])

            # execute kfold cross validation
            preds, tests = [], []
            for train, test in kfold.split(X_new):
                dt = DecisionTreeClassifier(random_state=self._random_state_init)
                dt.fit(X_new[train], y_new[train])
                preds.append(dt.predict(X_new[test]))
                tests.append(y_new[test])
            preds = np.hstack(preds)
            tests = np.hstack(tests)

            # compute fitness measure
            tp = np.sum(np.logical_and(
                tests == self.min_label, tests == preds))
            tn = np.sum(np.logical_and(
                tests == self.maj_label, tests == preds))
            fp = np.sum(np.logical_and(
                tests == self.maj_label, tests != preds))
            fn = np.sum(np.logical_and(
                tests == self.min_label, tests != preds))
            sens = tp/(tp + fn)
            spec = tn/(fp + tn)

            return np.sqrt(sens*spec)

        def crossover(conf_a, conf_b):
            """
            Crossover

            Args:
                conf_a (list(list)): configuration to crossover
                conf_b (list(list)): configuration to crossover

            Returns:
                list(list), list(list): the configurations after crossover
            """
            for _ in range(self.popl3):
                k = self.random_state.randint(len(conf_a))
                conf_a = np.hstack([conf_a[:k], conf_b[k:]])
                conf_b = np.hstack([conf_b[:k], conf_a[k:]])
            return conf_a, conf_b

        def mutation(conf, ge):
            """
            Mutation

            Args:
                conf (list(list)): configuration to mutate
                ge (int): iteration number
            """
            conf = conf.copy()
            if self.random_state.random_sample() < self.pm:
                pass
            else:
                for i in range(len(conf)):
                    r = self.random_state.random_sample()
                    r = r**((1 - ge/self.Ge)**3)
                    if self.random_state.randint(2) == 0:
                        conf[i] = int(conf[i] + (self.maxn - conf[i])*r)
                    else:
                        conf[i] = int(conf[i] - (conf[i] - 0)*r)
            return conf

        # generate initial population
        def init_pop():
            return self.random_state.randint(self.maxn, size=len(X_min))

        population = [[init_pop(), 0] for _ in range(self.n_pop)]

        # calculate fitness values
        for p in population:
            p[1] = fitness(p[0])

        # start iteration
        ge = 0
        while ge < self.Ge:
            # sorting population in descending order by fitness scores
            population = sorted(population, key=lambda x: -x[1])

            # selection operation (Step 2)
            pp = int(self.n_pop*self.pr)
            population_new = []
            for i in range(pp):
                population_new.append(population[i])
            population_new.extend(population[:(self.n_pop - pp)])
            population = population_new

            # crossover
            for _ in range(int(self.n_pop/2)):
                pop_0 = population[self.random_state.randint(self.n_pop)][0]
                pop_1 = population[self.random_state.randint(self.n_pop)][0]
                conf_a, conf_b = crossover(pop_0, pop_1)
                population.append([conf_a, fitness(conf_a)])
                population.append([conf_b, fitness(conf_b)])

            # mutation
            for _ in range(int(self.n_pop/2)):
                pop_0 = population[self.random_state.randint(self.n_pop)][0]
                conf = mutation(pop_0, ge)
                population.append([conf, fitness(conf)])

            ge = ge + 1

        # sorting final population
        population = sorted(population, key=lambda x: -x[1])

        # get best configuration
        conf = population[0][0]

        # generate final samples
        samples = []
        for i in range(len(conf)):
            for _ in range(conf[i]):
                samples.append(self.sample_between_points(
                    X_min[i], X_min[self.random_state.choice(ind[i][1:])]))

        if len(samples) == 0:
            return X.copy(), y.copy()

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'maxn': self.maxn,
                'n_pop': self.n_pop,
                'popl3': self.popl3,
                'pm': self.pm,
                'pr': self.pr,
                'Ge': self.Ge,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

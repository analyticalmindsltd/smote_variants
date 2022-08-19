"""
This module implements the GASMOTE method.
"""

import numpy as np

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from ..base import coalesce, coalesce_dict
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['GASMOTE']

class GASMOTE(OverSamplingSimplex):
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

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_memetic,
                  OverSamplingSimplex.cat_sample_ordinary,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 maxn=7,
                 n_pop=10,
                 popl3=5,
                 pm=0.3,
                 pr=0.2,
                 Ge=10,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            n_neighbors (int): number of neighbors
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
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
        nn_params = coalesce(nn_params, {})

        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(maxn, "maxn", 1)
        self.check_greater_or_equal(n_pop, "n_pop", 1)
        self.check_in_range(pm, "pm", [0, 1])
        self.check_in_range(pr, "pr", [0, 1])
        self.check_greater_or_equal(Ge, "Ge", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.ga_params = {'maxn': maxn,
                            'n_pop': n_pop,
                            'popl3': popl3,
                            'pm': pm,
                            'pr': pr,
                            'Ge': Ge}
        self.n_jobs = n_jobs

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

    def samples_from_conf(self, X_min, ind, conf):
        """
        Generate samples from a configuration.

        Args:
            X_min (np.array): the minority samples
            ind (np.array): the neighborhoods
            conf (np.array): a configuration

        Returns:
            np.array: the generated samples
        """
        samples = [np.zeros(shape=(0, X_min.shape[1]), dtype=float)]
        for idx, conf_i in enumerate(conf):
            samples.append(self.sample_simplex(X=X_min[[idx]],
                                                indices=ind[[idx]],
                                                n_to_sample=conf_i,
                                                X_vertices=X_min))
        return np.vstack(samples)

    def calculate_score(self, tests, preds):
        """
        Calculate the objective function score

        Args:
            tests (np.array): the test labels
            preds (np.array): the predicted labels

        Returns:
            float: the score
        """
        # compute fitness measure
        t_p = np.sum(np.logical_and(tests == self.min_label, tests == preds))
        t_n = np.sum(np.logical_and(tests == self.maj_label, tests == preds))
        f_p = np.sum(np.logical_and(tests == self.maj_label, tests != preds))
        f_n = np.sum(np.logical_and(tests == self.min_label, tests != preds))
        sens = t_p/(t_p + f_n)
        spec = t_n/(f_p + t_n)

        return np.sqrt(sens*spec)

    def fitness(self, *, conf, X_min, ind, X, y, kfold):
        """
        Evluate fitness of configuration

        Args:
            conf (list(list)): configuration
            X_min (np.array): the minority samples
            ind (np.array): the neighborhood relation
            X (np.array): all vectors
            y (np.array): all target labels
            kfold (obj): cross validation object

        Returns:
            float: the fitness score
        """
        # generate new samples
        samples = self.samples_from_conf(X_min, ind, conf)

        #samples = []
        #for idx, conf_i in enumerate(conf):
        #    for _ in range(conf_i):
        #        X_b = X_min[self.random_state.choice(ind[idx][1:])]
        #        samples.append(self.sample_between_points(X_min[idx], X_b))

        X_new = np.vstack([X, samples])
        y_new = np.hstack([y, np.repeat(self.min_label, len(samples))])

        # execute kfold cross validation
        preds, tests = [], []
        for train, test in kfold.split(X_new):
            clas = DecisionTreeClassifier(random_state=self._random_state_init)
            clas.fit(X_new[train], y_new[train])
            preds.append(clas.predict(X_new[test]))
            tests.append(y_new[test])
        preds = np.hstack(preds)
        tests = np.hstack(tests)

        return self.calculate_score(tests, preds)

    def crossover(self, conf_a, conf_b):
        """
        Crossover

        Args:
            conf_a (list): configuration to crossover
            conf_b (list): configuration to crossover

        Returns:
            list, list: the configurations after crossover
        """
        for _ in range(self.ga_params['popl3']):
            kdx = self.random_state.randint(len(conf_a))
            conf_a = np.hstack([conf_a[:kdx], conf_b[kdx:]])
            conf_b = np.hstack([conf_b[:kdx], conf_a[kdx:]])
        return conf_a, conf_b

    def mutation(self, conf, ge_):
        """
        Mutation

        Args:
            conf (list): configuration to mutate
            ge (int): iteration number

        Returns:
            list: the mutated configuration
        """
        conf = conf.copy()
        if self.random_state.random_sample() >= self.ga_params['pm']:
            for idx, conf_i in enumerate(conf):
                rand = self.random_state.random_sample()
                rand = rand**((1 - ge_/self.ga_params['Ge'])**3)
                if self.random_state.randint(2) == 0:
                    tmp = (self.ga_params['maxn'] - conf_i) * rand
                    conf[idx] = int(conf_i + tmp)
                else:
                    conf[idx] = int(conf_i - (conf_i - 0)*rand)
        return conf

    # generate initial population
    def init_pop(self, X_min):
        """
        Initialize the population

        Args:
            X_min (np.array): the minority vectors

        Returns:
            np.array: an initial population
        """
        return self.random_state.randint(self.ga_params['maxn'], size=X_min.shape[0])

    def genetic_algorithm(self, *, X_min, ind, X, y, kfold):
        """
        The genetic algorithm

        Args:
            X_min (np.array): the minority vectors
            ind (np.array): the neighborhood relations
            X (np.array): all feature vectors
            y (np.array): all target labels
            kfold (obj): a k-fold cross-validation object
        """
        population = [[self.init_pop(X_min), 0] for _ in range(self.ga_params['n_pop'])]

        # calculate fitness values
        for pop in population:
            pop[1] = self.fitness(conf=pop[0],
                                    X_min=X_min,
                                    ind=ind,
                                    X=X,
                                    y=y,
                                    kfold=kfold)

        # start iteration
        ge_ = 0
        while ge_ < self.ga_params['Ge']:
            # sorting population in descending order by fitness scores
            population = sorted(population, key=lambda x: -x[1])

            # selection operation (Step 2)
            n_keep = int(self.ga_params['n_pop']*self.ga_params['pr'])
            population_new = []
            population_new.extend(population[:n_keep])
            population_new.extend(population[:(self.ga_params['n_pop'] - n_keep)])
            population = population_new

            # crossover
            for _ in range(int(self.ga_params['n_pop']/2)):
                pop_0 = population[self.random_state.randint(self.ga_params['n_pop'])][0]
                pop_1 = population[self.random_state.randint(self.ga_params['n_pop'])][0]
                conf_a, conf_b = self.crossover(pop_0, pop_1)
                population.append([conf_a, self.fitness(conf=conf_a,
                                                        X_min=X_min,
                                                        ind=ind,
                                                        X=X,
                                                        y=y,
                                                        kfold=kfold)])
                population.append([conf_b, self.fitness(conf=conf_b,
                                                        X_min=X_min,
                                                        ind=ind,
                                                        X=X,
                                                        y=y,
                                                        kfold=kfold)])

            # mutation
            for _ in range(int(self.ga_params['n_pop']/2)):
                pop_0 = population[self.random_state.randint(self.ga_params['n_pop'])][0]
                conf_a = self.mutation(pop_0, ge_)
                population.append([conf_a, self.fitness(conf=conf_a,
                                                        X_min=X_min,
                                                        ind=ind,
                                                        X=X,
                                                        y=y,
                                                        kfold=kfold)])

            ge_ = ge_ + 1

        # sorting final population
        population = sorted(population, key=lambda x: -x[1])

        # return best configuration
        return population[0][0]

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        X_min = X[y == self.min_label]

        # fitting nearest neighbors model to find minority neighbors of
        #  minority samples
        n_neighbors = min([self.n_neighbors + 1, len(X_min)])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                        n_jobs=self.n_jobs,
                                                        **(nn_params))
        nnmt.fit(X_min)
        ind = nnmt.kneighbors(X_min, return_distance=False)
        kfold = KFold(min([len(X), 5]))

        conf = self.genetic_algorithm(X_min=X_min,
                                        ind=ind,
                                        X=X,
                                        y=y,
                                        kfold=kfold)

        # generate final samples
        samples = self.samples_from_conf(X_min, ind, conf)

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'maxn': self.ga_params['maxn'],
                'n_pop': self.ga_params['n_pop'],
                'popl3': self.ga_params['popl3'],
                'pm': self.ga_params['pm'],
                'pr': self.ga_params['pr'],
                'Ge': self.ga_params['Ge'],
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}

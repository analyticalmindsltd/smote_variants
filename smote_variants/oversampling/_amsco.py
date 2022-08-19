"""
This module implements the AMSCO method.
"""
from dataclasses import dataclass

import numpy as np

from sklearn.model_selection import StratifiedKFold

from ..base import instantiate_obj, coalesce_dict, coalesce
from ..base import OverSampling

from ._smote import SMOTE

from .._logger import logger
_logger= logger

__all__= ['AMSCO']

@dataclass
class AMSCOParams:
    """
    AMSCO parameters class
    """
    n_pop: int
    n_iter: int
    omega: float
    r1: float # pylint: disable=invalid-name
    r2: float # pylint: disable=invalid-name

@dataclass
class Particle:
    """
    Particle class
    """
    position: np.array
    velocity: np.array
    local_best: np.array
    local_score: tuple

@dataclass
class OSMOTEStatus:
    """
    OSMOTE status object
    """
    particles : list
    limits : list
    global_best : object
    global_score : tuple
    best_dataset : tuple

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
                 nn_params=None,
                 ss_params=None,
                 n_jobs=1,
                 classifier=('sklearn.tree',
                                'DecisionTreeClassifier',
                                {'random_state': 2}),
                 random_state=None,
                 **_kwargs):
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
            ss_params (dict): simplex sampling parameters
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(random_state=random_state,
                            checks={'min_n_min': 4})
        self.check_greater_or_equal(n_pop, "n_pop", 1)
        self.check_greater_or_equal(n_iter, "n_iter", 1)
        self.check_greater_or_equal(omega, "omega", 0)
        self.check_greater_or_equal(r1, "r1", 0)
        self.check_greater_or_equal(r2, "r2", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.params = AMSCOParams(n_pop, n_iter, omega, r1, r2)
        self.nn_params = coalesce(nn_params, {})
        self.ss_params = ss_params
        self.n_jobs = n_jobs
        self.classifier = classifier

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        # as the method is an overall optimization, 1 reasonable settings
        # should be enough

        classifiers = [('sklearn.tree',
                        'DecisionTreeClassifier',
                        {'random_state': 2})]
        parameter_combinations = {'n_pop': [5],
                                  'n_iter': [15],
                                  'omega': [0.1],
                                  'r1': [0.1],
                                  'r2': [0.1],
                                  'classifier': classifiers}

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def determine_scores(self, preds, tests):
        """
        Determine performance scores

        Args:
            preds (np.array): predictions
            tests (np.array): tested labels

        Returns:
            float, float: kappa, accuracy
        """
        # calculate kappa and accuracy scores
        true_p = np.sum(np.logical_and(preds == tests,
                                    tests == self.min_label))
        false_n = np.sum(np.logical_and(preds != tests,
                                    tests == self.min_label))
        true_n = np.sum(np.logical_and(preds == tests,
                                    tests == self.maj_label))
        false_p = np.sum(np.logical_and(preds != tests,
                                    tests == self.maj_label))

        p_o = (true_p + true_n)/(true_p + false_n + true_n + false_p)
        p_e = (true_p + false_n)*(true_p + false_p)/\
                (true_p + false_n + true_n + false_p)**2 + \
                (false_p + true_n)*(false_n + true_n)/\
                (true_p + false_n + true_n + false_p)**2

        kappa = (p_o - p_e)/(1.0 - p_e)
        accuracy = (true_p + true_n)/(true_p + false_n + true_n + false_p)

        return kappa, accuracy

    def fitness(self, *, X_min, X_maj, X, y, n_cross_val):
        """
        Calculating fitness function

        Args:
            X_min (np.array): minority samples
            X_maj (np.array): majority samples

        Returns:
            float, float: kappa, accuracy
        """
        kfold = StratifiedKFold(n_splits=np.max([2, np.min([n_cross_val, len(X_min), len(X_maj)])]),
                                shuffle=False)

        # prepare assembled dataset
        X_assembled = np.vstack([X_min, X_maj]) # pylint: disable=invalid-name
        y_assembled = np.hstack([np.repeat(self.min_label, len(X_min)),
                                 np.repeat(self.maj_label, len(X_maj))])

        preds = []
        tests = []

        for train, _ in kfold.split(X_assembled, y_assembled):
            classifier_obj = instantiate_obj(self.classifier)
            classifier_obj.fit(X_assembled[train], y_assembled[train])
            preds.append(classifier_obj.predict(X))
            tests.append(y)
        preds = np.hstack(preds)
        tests = np.hstack(tests)

        return self.determine_scores(preds, tests)

    def update_velocities_osmote(self, ostatus):
        """
        Update the velocities.

        Args:
            ostatus (OSMOTEStatus): status object of the search
        """
        for _, part in enumerate(ostatus.particles):
            diff1 = (part.local_best - part.velocity)
            diff2 = (ostatus.global_best - part.velocity)
            new_velocity = (part.velocity*self.params.omega +
                            self.params.r1 * diff1 +
                            self.params.r2 * diff2)
            # clipping velocities using the upper bounds of the
            # particle search space
            new_velocity[0] = np.clip(new_velocity[0],
                                        -ostatus.limits[1][0]/2,
                                        ostatus.limits[1][0]/2)
            new_velocity[1] = np.clip(new_velocity[1],
                                        -ostatus.limits[1][1]/2,
                                        ostatus.limits[1][1]/2)
            part.velocity = new_velocity

    def update_particles_osmote(self, ostatus):
        """
        Update the particles.

        Args:
            ostatus (OSMOTEStatus): status object of the search
        """
        # update particles
        for _, part in enumerate(ostatus.particles):
            new_position = part.position + part.velocity
            # clipping the particle positions using the lower and
            # upper bounds
            new_position[0] = np.clip(new_position[0],
                                        ostatus.limits[0][0],
                                        ostatus.limits[1][0])
            new_position[1] = np.clip(new_position[1],
                                        ostatus.limits[0][1],
                                        ostatus.limits[1][1])
            part.position = new_position

    def init_pop_osmote(self):
        """
        Initialize a population.

        Returns:
            np.array: the initial population
        """
        proportion = self.random_state.random_sample()/2.0+0.5
        n_neighbors = self.random_state.randint(3, 10)
        return Particle(position=np.array([proportion, n_neighbors]),
                        velocity=np.array([0.1, 1]),
                        local_best=np.array([proportion, n_neighbors]),
                        local_score=(0.0,0.0))

    def update_scores_osmote(self, ostatus, particle, score, dataset):
        """
        Update the scores based on the results for particle idx

        Args:
            ostatus (OSMOTEStatus): status object of the search
            particle (Particle): the particle
            score (tuple): score of the particle
            dataset (tuple): dataset represented by the particle
        """
        joint_local = np.prod(particle.local_score)
        joint_global = np.prod(ostatus.global_score)
        joint_score = np.prod(score)

        if joint_score > joint_local:
            particle.local_best = particle.position.copy()
            particle.local_score = score
        if joint_score > joint_global:
            ostatus.global_best = particle.position.copy()
            ostatus.global_score = score
            ostatus.best_dataset = dataset

    def evaluate_osmote(self, ostatus, *, X_maj, X_min, X, y,
                            n_cross_val, nn_params):
        """
        Evaluation

        Args:
            ostatus (OSMOTEStatus): status object of the search
            X_maj (np.array): majority samples
            X_min (np.array): minority samples
            X (np.array): all samples
            y (np.array): all targets
            n_cross_val (int): number of cross validations
            nn_params (dict): nearest neighbors parameters
        """
        # evaluate
        for _, particle in enumerate(ostatus.particles):
            # apply SMOTE
            smote = SMOTE(particle.position[0],
                            int(np.rint(particle.position[1])),
                            nn_params=nn_params,
                            ss_params=self.ss_params,
                            n_jobs=self.n_jobs,
                            random_state=self._random_state_init)

            y_to_sample = np.hstack([np.repeat(self.maj_label, len(X_maj)),
                                        np.repeat(self.min_label, len(X_min))])
            X_samp, _ = smote.sample(np.vstack([X_maj, X_min]),
                                    y_to_sample) # pylint: disable=invalid-name

            # evaluate
            score = self.fitness(X_min=X_samp[len(X_maj):],
                                    X_maj=X_samp[:len(X_maj)],
                                    X=X,
                                    y=y,
                                    n_cross_val=n_cross_val)

            # update scores according to the multiobjective setting
            self.update_scores_osmote(ostatus, particle, score,
                        dataset=(X_samp[len(X_maj):], X_samp[:len(X_maj)]))

    def osmote(self, *, X_min, X_maj, X, y, n_cross_val, nn_params):
        """
        Executing OSMOTE phase

        Args:
            X_min (np.array): minority samples
            X_maj (np.array): majority samples

        Returns:
            np.array, np.array: new minority and majority datasets
        """

        # initialize particles, first coordinate represents proportion
        # parameter of SMOTE
        # the second coordinate represents the number of neighbors to
        # take into consideration

        particles = [self.init_pop_osmote()\
                                    for _ in range(self.params.n_pop)]

        ostatus = OSMOTEStatus(
                    particles=particles,
                    limits=[np.array([0.25, 3]), np.array([4.0, 10])],
                    global_best=particles[0].position.copy(),
                    global_score=(0.0, 0.0),
                    best_dataset=None)

        # running the optimization
        for _ in range(self.params.n_iter):
            # update velocities
            self.update_velocities_osmote(ostatus)
            self.update_particles_osmote(ostatus)
            self.evaluate_osmote(ostatus=ostatus,
                                    X_maj=X_maj,
                                    X_min=X_min,
                                    X=X,
                                    y=y,
                                    n_cross_val=n_cross_val,
                                    nn_params=nn_params)

        return ostatus.best_dataset[0], ostatus.best_dataset[1]

    # initiate particles
    def init_particle_sis(self, min_num, max_num, X_maj):
        """
        Initialize particle for SIS.

        Args:
            min_num (int): minimum number
            max_num (int): maximum number
            X_maj (np.array): majority samples

        Returns:
            np.array: majority indices
        """
        num = self.random_state.randint(min_num, max_num)
        maj = self.random_state.choice(np.arange(len(X_maj)), num)
        return maj

    def remove_elements(self, particle):
        """
        Removes some random elements

        Args:
            particle (np.array): a particle

        Returns:
            np.array: the mutant particle
        """
        domain = np.arange(len(particle))
        #n_max = min([10, len(particle)])
        n_max = np.max([0, int(np.sqrt(len(particle)))])

        n_to_choose = self.random_state.randint(0, n_max)

        if len(particle) - n_to_choose < 4:
            return particle

        to_remove = self.random_state.choice(domain, n_to_choose)

        return np.delete(particle, to_remove)

    def add_elements(self, particle, mutant, X_maj):
        """
        Adds some new elements to the particle

        Args:
            particle (np.array): the particle
            mutant (np.array): the already mutated particle (
                                by removing elements)
            X_maj (np.array): the majority samples

        Returns:
            np.array: the mutated particle
        """
        maj_set = set(np.arange(len(X_maj)))
        part_set = set(particle)
        diff = list(maj_set.difference(part_set))
        n_max = min([10, len(diff)])
        n_to_choose = self.random_state.randint(0, n_max)
        diff_elements = self.random_state.choice(diff, n_to_choose)
        return np.hstack([mutant, np.array(diff_elements)])

    def sis(self, *, X_min, X_maj, X, y, n_cross_val):
        """
        SIS procedure

        Args:
            X_min (np.array): minority dataset
            X_maj (np.array): majority dataset

        Returns:
            np.array, np.array: new minority and majority datasets
        """
        min_num = len(X_min)
        max_num = len(X_maj)

        if min_num >= max_num:
            return X_min, X_maj

        particles = [self.init_particle_sis(min_num, max_num, X_maj)
                                    for _ in range(self.params.n_pop)]
        scores = [self.fitness(X_min=X_min,
                                X_maj=X_maj[particles[i]],
                                X=X,
                                y=y,
                                n_cross_val=n_cross_val)
                    for i in range(self.params.n_pop)]
        best_score = (0.0, 0.0)
        best_dataset = None

        for _ in range(self.params.n_iter):
            # mutate and evaluate
            # the way mutation or applying PSO is not described in the
            # paper in details
            for idx in range(self.params.n_pop):
                # removing some random elements
                mutant= self.remove_elements(particles[idx])
                # adding some random elements
                mutant = self.add_elements(particles[idx],
                                            mutant,
                                            X_maj)

                # evaluating the variant
                score = self.fitness(X_min=X_min,
                                    X_maj=X_maj[mutant],
                                    X=X,
                                    y=y,
                                    n_cross_val=n_cross_val)
                if score[1] > scores[idx][1]:
                    particles[idx] = mutant.copy()
                    scores[idx] = score
                if score[1] > best_score[1]:
                    best_score = score
                    best_dataset = mutant.copy()

        return X_min, X_maj[best_dataset]

    def evaluate(self, *, current, new_min, new_maj, X, y, n_cross_val):
        """
        Evaluate the actual configurations.

        Args:
            current (tuple): current best minority and majority
            new_min (np.array): new minority
            new_maj (np.array): new majority
            X (np.array): features
            y (np.array): target
            n_cross_val (int): cross validation
        """
        fitness_0 = np.prod(self.fitness(X_min=new_min,
                                            X_maj=current[1],
                                            X=X,
                                            y=y,
                                            n_cross_val=n_cross_val))
        fitness_1 = np.prod(self.fitness(X_min=current[0],
                                            X_maj=current[1],
                                            X=X,
                                            y=y,
                                            n_cross_val=n_cross_val))
        fitness_2 = np.prod(self.fitness(X_min=new_min,
                                            X_maj=new_maj,
                                            X=X,
                                            y=y,
                                            n_cross_val=n_cross_val))
        fitness_3 = np.prod(self.fitness(X_min=current[0],
                                            X_maj=new_maj,
                                            X=X,
                                            y=y,
                                            n_cross_val=n_cross_val))

        # selecting the new current_maj and current_min datasets
        _logger.info("%s: fitness scores: %f %f %f %f",
                        self.__class__.__name__, fitness_0,
                            fitness_1, fitness_2, fitness_3)

        max_fitness = np.max([fitness_0, fitness_1, fitness_2, fitness_3])
        if max_fitness in (fitness_1, fitness_3):
            current = (current[0], new_maj)
        if max_fitness in (fitness_0, fitness_2):
            current = (new_min, current[1])

        return current

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
        X_maj = X[y == self.maj_label]

        n_cross_val = min([4, len(X_min)])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor'] = \
                        self.metric_tensor_from_nn_params(nn_params, X, y)

        # executing the main optimization procedure
        current = (X_min, X_maj)
        for iteration in range(self.params.n_iter):
            _logger.info("%s starting iteration %d",
                            self.__class__.__name__, iteration)
            new_min, _ = self.osmote(X_min=X_min,
                                        X_maj=current[1],
                                        X=X,
                                        y=y,
                                        n_cross_val=n_cross_val,
                                        nn_params=nn_params)
            _, new_maj = self.sis(X_min=current[0],
                                    X_maj=X_maj,
                                    X=X,
                                    y=y,
                                    n_cross_val=n_cross_val)

            current = self.evaluate(current=current,
                                    new_min=new_min,
                                    new_maj=new_maj,
                                    X=X,
                                    y=y,
                                    n_cross_val=n_cross_val)

        return (np.vstack([current[1], current[0]]),
                np.hstack([np.repeat(self.maj_label, len(current[1])),
                           np.repeat(self.min_label, len(current[0]))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'n_pop': self.params.n_pop,
                'n_iter': self.params.n_iter,
                'omega': self.params.omega,
                'r1': self.params.r1,
                'r2': self.params.r2,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'classifier': self.classifier,
                **OverSampling.get_params(self)}

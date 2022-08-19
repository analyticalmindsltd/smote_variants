"""
This module implements the DSRBF method.
"""
import warnings

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from ..config import suppress_external_warnings
from ..base import OverSampling
from ..base import RandomStateMixin, coalesce, coalesce_dict
from ._smote import SMOTE

from .._logger import logger
_logger= logger

__all__= ['DSRBF']

class RBFNeuron(RandomStateMixin):
    """
    This class abstracts a neuron of an RBF network
    """

    def __init__(self,
                 *,
                 center,
                 i_b,
                 o_b,
                 ranges,
                 range_mins,
                 init_conn_mask,
                 init_conn_weights,
                 beta=None,
                 mask=None,
                 input_weights=None,
                 radius=None,
                 random_state=None):
        """
        Constructor of the neuron

        Args:
            center (np.array): center of the hidden unit
            i_b (float): upper bound on the absolute values of input weights
            o_b (float): upper bound on the absolute values of output weights
            ranges (np.array): ranges widths of parameters
            range_min (np.array): lower bounds of parameter ranges
            init_conn_mask (np.array): initial input connections
            init_conn_weights (np.array): initial weights of input connections
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        RandomStateMixin.__init__(self, random_state)
        self.center = center

        self.beta = coalesce(beta, (self.random_state.random_sample()-0.5) * o_b)
        self.init_conn_mask = coalesce(mask, init_conn_mask)
        self.input_weights = coalesce(input_weights, init_conn_weights)
        self.mask = init_conn_mask
        self.radius = coalesce(radius, self.random_state.random_sample())

        self.params = {'i_b': i_b,
                        'o_b': o_b,
                        'init_conn_mask': init_conn_mask,
                        'init_conn_weights': init_conn_weights,
                        'ranges': ranges,
                        'range_mins': range_mins,
                        'beta': beta,
                        'input_weights': input_weights
                        }

    def clone(self):
        """
        Clones the neuron

        Returns:
            RBFNeuron: an identical neuron
        """
        return RBFNeuron(**(self.params),
                        center=self.center,
                        random_state=self._random_state_init,
                        mask=self.mask.copy(),
                        radius=self.radius)

    def evaluate(self, X):
        """
        Evaluates the system on dataset X

        Args:
            X (np.array): dataset to evaluate on

        Returns:
            np.array: the output of the network
        """
        weighted = X[:, self.mask]*self.input_weights
        shifted = weighted - self.center[self.mask]
        term_exp = -np.linalg.norm(shifted, axis=1)**2/self.radius**2
        return self.beta*np.exp(term_exp)

    def mutate(self):
        """
        Mutates the neuron
        """
        n_dim = self.center.shape[0]
        rand = self.random_state.random_sample()
        if rand < 0.2:
            # centre creep
            self.center = self.random_state.normal(self.center,
                                                    self.radius)
        elif rand < 0.4:
            # radius creep
            tmp = self.random_state.normal(self.radius,
                                            np.var(self.params['ranges']))
            if tmp > 0:
                self.radius = tmp
        elif rand < 0.6:
            # randomize centers
            self.center = self.random_state.random_sample(size=n_dim)\
                            * self.params['ranges'] + self.params['range_mins']
        elif rand < 0.8:
            # randomize radii
            self.radius = self.random_state.random_sample()\
                             * np.mean(self.params['ranges'])
        else:
            # randomize output weight
            self.beta = self.random_state.normal(self.beta, self.params['o_b'])

    def add_connection(self):
        """
        Adds a random input connection to the neuron
        """
        n_dim = self.center.shape[0]
        if len(self.mask) < n_dim:
            d_set = set(range(n_dim))
            mask_set = set(self.mask.tolist())

            domain = list(d_set.difference(mask_set))

            additional_elements = np.array(self.random_state.choice(domain))

            self.mask = np.hstack([self.mask,
                                    additional_elements])

            random_weight = (self.random_state.random_sample() - 0.5)\
                                                     * self.params['i_b']

            self.input_weights = np.hstack([self.input_weights,
                                            random_weight])

    def delete_connection(self):
        """
        Deletes a random input connection
        """
        if len(self.mask) > 1:
            idx = self.random_state.randint(len(self.mask))
            self.mask = np.delete(self.mask, idx)
            self.input_weights = np.delete(self.input_weights, idx)

class RBF(RandomStateMixin):
    """
    RBF network abstraction
    """

    def __init__(self,
                 *,
                 X,
                 m_min,
                 m_max,
                 i_b,
                 o_b,
                 init_conn_mask,
                 init_conn_weights,
                 range_mins=None,
                 ranges=None,
                 neurons=None,
                 beta_0=None,
                 random_state=None):
        """
        Initializes the RBF network

        Args:
            X (np.array): dataset to work with
            m_min (int): minimum number of hidden neurons
            m_max (int): maximum number of hidden neurons
            Ib (float): maximum absolute value of input weights
            Ob (float): maximum absolute value of output weights
            init_conn_mask (np.array): initial input connections
            init_conn_weights (np.array): initial input weights
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        RandomStateMixin.__init__(self, random_state)

        self.X = X

        self.range_mins = coalesce(range_mins, np.min(X, axis=0))
        self.ranges = coalesce(ranges, np.max(X, axis=0) - self.range_mins)

        beta_0 = coalesce(beta_0, (self.random_state.random_sample() - 0.5) * o_b)

        self.params = {'m_min': m_min,
                        'm_max': m_max,
                        'i_b': i_b,
                        'o_b': o_b,
                        'init_conn_mask': init_conn_mask,
                        'init_conn_weights': init_conn_weights,
                        'beta_0': beta_0}

        if neurons is not None:
            self.neurons = neurons
        else:
            self.neurons = []
            num_neurons = self.random_state.randint(m_min, m_max)
            for _ in range(num_neurons):
                self.neurons.append(self.create_new_node())

    def clone(self):
        """
        Clones the entire network

        Returns:
            RBF: the cloned network
        """
        return RBF(X=self.X,
                    **self.params,
                    neurons=[n.clone() for n in self.neurons],
                    range_mins=self.range_mins.copy(),
                    ranges=self.ranges.copy(),
                    random_state=self._random_state_init)

    def create_new_node(self):
        """
        Creates a new node.

        Returns:
            RBFNeuron: a new hidden neuron
        """
        return RBFNeuron(center=self.X[self.random_state.randint(self.X.shape[0])],
                         i_b=self.params['i_b'],
                         o_b=self.params['o_b'],
                         ranges=self.ranges,
                         range_mins=self.range_mins,
                         init_conn_mask=self.params['init_conn_mask'],
                         init_conn_weights=self.params['init_conn_weights'],
                         random_state=self._random_state_init)

    def update_data(self, X):
        """
        Updates the data to work with

        Returns:
            RBF: the updated RBF object
        """
        self.X = X
        for neuron in self.neurons:
            neuron.X = X
        return self

    def improve_centers(self):
        """
        Improves the center locations by kmeans clustering

        Returns:
            RBF: the RBF object with updated centers
        """
        if len(np.unique(self.X, axis=0)) > len(self.neurons):
            cluster_init = np.vstack([neuron.center for neuron in self.neurons])
            kmeans = KMeans(n_clusters=len(self.neurons),
                            init=cluster_init,
                            n_init=1,
                            max_iter=30,
                            random_state=self._random_state_init)
            with warnings.catch_warnings():
                if suppress_external_warnings():
                    warnings.simplefilter("ignore")
                kmeans.fit(self.X)

            for idx, neuron in enumerate(self.neurons):
                neuron.center = kmeans.cluster_centers_[idx]

        return self

    def evaluate(self, X, y):
        """
        Evaluates the target function

        Returns:
            float: the target function value
        """
        evaluation = np.column_stack([n.evaluate(X) for n in self.neurons])
        func = self.params['beta_0'] + np.sum(evaluation, axis=1)
        L_star = np.mean(abs(y[y == 1] - func[y == 1])) # pylint: disable=invalid-name
        L_star += np.mean(abs(y[y == 0] - func[y == 0])) # pylint: disable=invalid-name
        return L_star

    def mutation(self):
        """
        Mutates the neurons

        Returns:
            RBF: a new, mutated RBF network
        """
        rbf = self.clone()
        for neuron in rbf.neurons:
            neuron.mutate()
        return rbf

    def structural_mutation(self):
        """
        Applies structural mutation

        Returns:
            RBF: a new, structurally mutated network
        """
        # in the binary case the removal of output connections is the same as
        # removing hidden nodes
        rbf = self.clone()
        rand = self.random_state.random_sample()
        if rand < 0.5:
            if len(rbf.neurons) < rbf.params['m_max']:
                rbf.neurons.append(rbf.create_new_node())
            elif len(rbf.neurons) > rbf.params['m_min']:
                del rbf.neurons[self.random_state.randint(len(rbf.neurons))]
        else:
            rbf.neurons[self.random_state.randint(len(rbf.neurons))]\
                                                        .delete_connection()
            rbf.neurons[self.random_state.randint(len(rbf.neurons))]\
                                                        .add_connection()

        return rbf

    def recombine(self, rbf):
        """
        Recombines two networks

        Args:
            rbf (RBF): another network

        Returns:
            RBF: the result of recombination
        """
        # the order of neurons doesn't matter, so the logic can be simplified
        new_rbf = self.clone()

        if self.random_state.random_sample() < 0.5:
            n_random = self.random_state.randint(1, len(new_rbf.neurons))
            new_neurons_0 = self.random_state.choice(new_rbf.neurons, n_random)

            n_random = self.random_state.randint(1, len(rbf.neurons))
            new_neurons_1 = self.random_state.choice(rbf.neurons, n_random)

            new_rbf.neurons = [n.clone() for n in new_neurons_0]
            new_rbf.neurons.extend([n.clone() for n in new_neurons_1])

            while len(new_rbf.neurons) > self.params['m_max']:
                n_neurons = len(new_rbf.neurons)
                del new_rbf.neurons[self.random_state.randint(n_neurons)]
        else:
            for idx, _ in enumerate(new_rbf.neurons):
                if self.random_state.random_sample() < 0.2:
                    n_random = self.random_state.randint(len(rbf.neurons))
                    new_rbf.neurons[idx] = rbf.neurons[n_random].clone()

        return new_rbf


class DSRBF(OverSampling):
    """
    References:
        * BibTex::

            @article{dsrbf,
                        title = "A dynamic over-sampling procedure based on
                                    sensitivity for multi-class problems",
                        journal = "Pattern Recognition",
                        volume = "44",
                        number = "8",
                        pages = "1821 - 1833",
                        year = "2011",
                        issn = "0031-3203",
                        doi = "https://doi.org/10.1016/j.patcog.2011.02.019",
                        author = "Francisco Fernández-Navarro and César
                                    Hervás-Martínez and Pedro Antonio
                                    Gutiérrez",
                        keywords = "Classification, Multi-class, Sensitivity,
                                    Accuracy, Memetic algorithm, Imbalanced
                                    datasets, Over-sampling method, SMOTE"
                        }

    Notes:
        * It is not entirely clear why J-1 output is supposed where J is the
            number of classes.
        * The fitness function is changed to a balanced mean loss, as I found
            that it just ignores classification on minority samples
            (class label +1) in the binary case.
        * The iRprop+ optimization is not implemented.
        * The original paper proposes using SMOTE incrementally. Instead of
            that, this implementation applies SMOTE to generate all samples
            needed in the sampling epochs and the evolution of RBF networks
            is used to select the sampling providing the best results.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_classifier,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_memetic,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 hidden_range=(4, 10),
                 i_b=2,
                 o_b=2,
                 n_pop=500,
                 n_init_pop=5000,
                 n_iter=40,
                 n_sampling_epoch=5,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): number of neighbors in the SMOTE sampling
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
            m_min (int): minimum number of hidden units
            m_max (int): maximum number of hidden units
            i_b (float): input weight range
            o_b (float): output weight range
            n_pop (int): size of population
            n_init_pop (int): size of initial population
            n_iter (int): number of iterations
            n_sampling_epoch (int): resampling after this many iterations
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state)

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(hidden_range[0], "m_min", 1)
        self.check_greater_or_equal(hidden_range[1], "m_max", 1)
        self.check_greater(i_b, "Ib", 0)
        self.check_greater(o_b, "Ob", 0)
        self.check_greater_or_equal(n_pop, "n_pop", 2)
        self.check_greater_or_equal(n_init_pop, "n_pop", 2)
        self.check_greater_or_equal(n_iter, "n_iter", 0)
        self.check_greater_or_equal(n_sampling_epoch, "n_sampling_epoch", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
        self.ss_params = coalesce_dict(ss_params, {'n_dim': 2,
                                        'simplex_sampling': 'uniform',
                                        'within_simplex_sampling': 'random',
                                        'gaussian_component': None})
        self.params = {'m_min': hidden_range[0],
                        'm_max': hidden_range[1],
                        'i_b': i_b,
                        'o_b': o_b}
        self.search_params = {'n_pop': n_pop,
                                'n_init_pop': n_init_pop,
                                'n_iter': n_iter,
                                'n_sampling_epoch': n_sampling_epoch}
        self.n_jobs = n_jobs

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        # as the technique optimizes, it is unnecessary to check various
        # combinations except one specifying a decent workspace with a large
        # number of iterations
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'hidden_range': [(4, 10)],
                                  'i_b': [2.0],
                                  'o_b': [2.0],
                                  'n_pop': [100],
                                  'n_init_pop': [1000],
                                  'n_iter': [40],
                                  'n_sampling_epoch': [8]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def init_pop(self, X, domain, n_random, m_max):
        """
        Initialize an element of the population.

        Args:
            X (np.array): input data
            domain (np.array): domain of connections
            n_random (np.array): the random cardinality of connections
            m_max (int): maximum number of hidden units
        """
        init_conn_mask = self.random_state.choice(domain, n_random)
        init_conn_weights = self.random_state.random_sample(size=n_random)

        return RBF(X=X,
                    m_min=self.params['m_min'],
                    m_max=m_max,
                    i_b=self.params['i_b'],
                    o_b=self.params['o_b'],
                    init_conn_mask=init_conn_mask,
                    init_conn_weights=init_conn_weights,
                    random_state=self._random_state_init)

    def recombination(self, population):
        """
        Do the recombination.

        Args:
            population (list): the population to recombine

        Returns:
            list: the recombined population
        """
        p_recombination = population[:int(0.1*self.search_params['n_pop'])]
        for pop in p_recombination:
            domain = range(len(p_recombination))
            p_rec = p_recombination[self.random_state.choice(domain)][0]
            population.append([pop[0].recombine(p_rec), pop[1], pop[2], np.inf])

        return population

    def execute_optimization(self, m_max, nn_params, X, y):
        """
        Execute the optimization.

        Args:
            m_max (int): the maximum number of hidden units
            nn_params (dict): the nearest neighbors parameters
            X (np.array): all vectors
            y (np.array): all target labels

        Returns:
            list: the best constituent of the final population
        """
        n_pop = self.search_params['n_pop']
        n_iter = self.search_params['n_iter']

        smote = SMOTE(proportion=self.proportion,
                              n_neighbors=self.n_neighbors,
                              nn_params=nn_params,
                              ss_params=self.ss_params,
                              n_jobs=self.n_jobs,
                              random_state=self._random_state_init)

        X_orig, y_orig = X, y

        X, y = smote.sample(X, y)

        # setting epoch lengths
        epoch_len = int(self.search_params['n_iter']\
                        / self.search_params['n_sampling_epoch'])

        # generating initial population

        population = [self.init_pop(X=X,
                                    domain=np.arange(X.shape[1]),
                                    n_random=int(X.shape[1]/2),
                                    m_max=m_max) \
                            for _ in range(self.search_params['n_init_pop'])]
        population = [[pop, X, y, np.inf] for pop in population]
        population = sorted([[pop[0], pop[1], pop[2], pop[0].evaluate(pop[1], pop[2])]
                             for pop in population], key=lambda x: x[3])
        population = population[:n_pop]

        # executing center improval in the hidden units
        for pop in population:
            pop[0].improve_centers()

        # executing the optimization process
        for iteration in range(n_iter):
            _logger.info("%s: Iteration %d/%d, loss: %f, data size: %d",
                        self.__class__.__name__, iteration,
                        n_iter, population[0][3], len(population[0][1]))

            # evaluating non-evaluated elements
            for pop in population:
                if pop[3] == np.inf:
                    pop[3] = pop[0].evaluate(pop[1], pop[2])

            # sorting the population by the loss values
            population = sorted((pop for pop in population), key=lambda x: x[3])
            population = population[:n_pop]

            # executing mutation
            for pop in population[:int(0.1*n_pop)]:
                population.append([pop[0].mutation(), pop[1], pop[2], np.inf])

            # executing structural mutation
            for pop in population[:int(0.9*n_pop-1)]:
                population.append(
                    [pop[0].structural_mutation(), pop[1], pop[2], np.inf])

            # executing recombination
            self.recombination(population)

            # do the sampling
            if iteration % epoch_len == 0:
                X, y = smote.sample(X_orig, y_orig)
                extension = []
                for pop in population:
                    extension.append([pop[0].clone()\
                                        .update_data(X)\
                                        .improve_centers(), X, y, np.inf])
                population.extend(extension)

        # evaluate unevaluated elements of the population
        for pop in population:
            if pop[3] == np.inf:
                pop[3] = pop[0].evaluate(pop[1], pop[2])

        # sorting the population
        population = sorted((pop for pop in population),
                            key=lambda x: x[3])[:n_pop]

        return population[0]

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
            return self.return_copies(X, y, "Sampling is not needed")

        m_max = min(len(X), self.params['m_max'])

        if self.params['m_min'] >= m_max or len(X) < self.params['m_min'] + 1:
            return self.return_copies(X, y,
                    "Range of the number of hidden units is not suitable "\
                    f"{self.descriptor()}")

        # Standardizing the data to let the network work with comparable
        # attributes

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        p_best = self.execute_optimization(m_max, nn_params, X, y)

        return scaler.inverse_transform(p_best[1]), p_best[2]

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'ss_params': self.ss_params,
                'hidden_range': (self.params['m_min'], self.params['m_max']),
                'i_b': self.params['i_b'],
                'o_b': self.params['o_b'],
                'n_pop': self.search_params['n_pop'],
                'n_init_pop': self.search_params['n_init_pop'],
                'n_iter': self.search_params['n_iter'],
                'n_sampling_epoch': self.search_params['n_sampling_epoch'],
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}

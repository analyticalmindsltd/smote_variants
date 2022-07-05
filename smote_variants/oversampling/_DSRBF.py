import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._base import RandomStateMixin
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['DSRBF']

class RBFNeuron(RandomStateMixin):
    """
    This class abstracts a neuron of an RBF network
    """

    def __init__(self,
                 c,
                 Ib,
                 Ob,
                 ranges,
                 range_mins,
                 init_conn_mask,
                 init_conn_weights,
                 random_state=None):
        """
        Constructor of the neuron

        Args:
            c (np.array): center of the hidden unit
            Ib (float): upper bound on the absolute values of input weights
            Ob (float): upper bound on the absolute values of output weights
            ranges (np.array): ranges widths of parameters
            range_min (np.array): lower bounds of parameter ranges
            init_conn_mask (np.array): initial input connections
            init_conn_weights (np.array): initial weights of input connections
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        self.d = len(c)
        self.c = c
        self.Ib = Ib
        self.Ob = Ob
        self.init_conn_mask = init_conn_mask
        self.init_conn_weights = init_conn_weights
        self.ranges = ranges
        self.range_mins = range_mins

        self.set_random_state(random_state)

        self.beta = (self.random_state.random_sample()-0.5)*Ob
        self.mask = init_conn_mask
        self.input_weights = init_conn_weights
        self.r = self.random_state.random_sample()

    def clone(self):
        """
        Clones the neuron

        Returns:
            RBFNeuron: an identical neuron
        """
        r = RBFNeuron(self.c,
                      self.Ib,
                      self.Ob,
                      self.ranges,
                      self.range_mins,
                      self.init_conn_mask,
                      self.init_conn_weights,
                      random_state=self._random_state_init)
        r.beta = self.beta
        r.mask = self.mask.copy()
        r.input_weights = self.input_weights.copy()
        r.r = self.r

        return r

    def evaluate(self, X):
        """
        Evaluates the system on dataset X

        Args:
            X (np.matrix): dataset to evaluate on

        Returns:
            np.array: the output of the network
        """
        wX = X[:, self.mask]*self.input_weights
        term_exp = -np.linalg.norm(wX - self.c[self.mask], axis=1)**2/self.r**2
        return self.beta*np.exp(term_exp)

    def mutate(self):
        """
        Mutates the neuron
        """
        r = self.random_state.random_sample()
        if r < 0.2:
            # centre creep
            self.c = self.random_state.normal(self.c, self.r)
        elif r < 0.4:
            # radius creep
            tmp = self.random_state.normal(self.r, np.var(self.ranges))
            if tmp > 0:
                self.r = tmp
        elif r < 0.6:
            # randomize centers
            self.c = self.random_state.random_sample(
                size=len(self.c))*self.ranges + self.range_mins
        elif r < 0.8:
            # randomize radii
            self.r = self.random_state.random_sample()*np.mean(self.ranges)
        else:
            # randomize output weight
            self.beta = self.random_state.normal(self.beta, self.Ob)

    def add_connection(self):
        """
        Adds a random input connection to the neuron
        """
        if len(self.mask) < self.d:
            d_set = set(range(self.d))
            mask_set = set(self.mask.tolist())
            domain = list(d_set.difference(mask_set))
            additional_elements = np.array(self.random_state.choice(domain))
            self.mask = np.hstack([self.mask, additional_elements])
            random_weight = (self.random_state.random_sample()-0.5)*self.Ib
            self.input_weights = np.hstack([self.input_weights, random_weight])

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
                 X,
                 m_min,
                 m_max,
                 Ib,
                 Ob,
                 init_conn_mask,
                 init_conn_weights,
                 random_state=None):
        """
        Initializes the RBF network

        Args:
            X (np.matrix): dataset to work with
            m_min (int): minimum number of hidden neurons
            m_max (int): maximum number of hidden neurons
            Ib (float): maximum absolute value of input weights
            Ob (float): maximum absolute value of output weights
            init_conn_mask (np.array): initial input connections
            init_conn_weights (np.array): initial input weights
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        self.X = X
        self.m_min = m_min
        self.m_max = m_max
        self.Ib = Ib
        self.Ob = Ob
        self.init_conn_mask = init_conn_mask
        self.init_conn_weights = init_conn_weights

        self.set_random_state(random_state)

        self.neurons = []
        self.range_mins = np.min(X, axis=0)
        self.ranges = np.max(X, axis=0) - self.range_mins

        # adding initial neurons
        num_neurons = self.random_state.randint(m_min, m_max)
        for _ in range(num_neurons):
            self.neurons.append(self.create_new_node())

        self.beta_0 = (self.random_state.random_sample()-0.5)*Ob

    def clone(self):
        """
        Clones the entire network

        Returns:
            RBF: the cloned network
        """
        r = RBF(self.X,
                self.m_min,
                self.m_max,
                self.Ib,
                self.Ob,
                self.init_conn_mask,
                self.init_conn_weights,
                random_state=self._random_state_init)
        r.neurons = [n.clone() for n in self.neurons]
        r.range_mins = self.range_mins.copy()
        r.ranges = self.ranges.copy()
        r.beta_0 = self.beta_0

        return r

    def create_new_node(self):
        """
        Creates a new node.

        Returns:
            RBFNeuron: a new hidden neuron
        """
        return RBFNeuron(self.X[self.random_state.randint(len(self.X))],
                         self.Ib,
                         self.Ob,
                         self.ranges,
                         self.range_mins,
                         self.init_conn_mask,
                         self.init_conn_weights,
                         random_state=self._random_state_init)

    def update_data(self, X):
        """
        Updates the data to work with
        """
        self.X = X
        for n in self.neurons:
            n.X = X

    def improve_centers(self):
        """
        Improves the center locations by kmeans clustering
        """
        if len(np.unique(self.X, axis=0)) > len(self.neurons):
            cluster_init = np.vstack([n.c for n in self.neurons])
            kmeans = KMeans(n_clusters=len(self.neurons),
                            init=cluster_init,
                            n_init=1,
                            max_iter=30,
                            random_state=self._random_state_init)
            kmeans.fit(self.X)
            for i in range(len(self.neurons)):
                self.neurons[i].c = kmeans.cluster_centers_[i]

    def evaluate(self, X, y):
        """
        Evaluates the target function

        Returns:
            float: the target function value
        """
        evaluation = np.column_stack([n.evaluate(X) for n in self.neurons])
        f = self.beta_0 + np.sum(evaluation, axis=1)
        L_star = np.mean(abs(y[y == 1] - f[y == 1]))
        L_star += np.mean(abs(y[y == 0] - f[y == 0]))
        return L_star

    def mutation(self):
        """
        Mutates the neurons

        Returns:
            RBF: a new, mutated RBF network
        """
        rbf = self.clone()
        for n in rbf.neurons:
            n.mutate()
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
        r = self.random_state.random_sample()
        if r < 0.5:
            if len(rbf.neurons) < rbf.m_max:
                rbf.neurons.append(rbf.create_new_node())
            elif len(rbf.neurons) > rbf.m_min:
                del rbf.neurons[self.random_state.randint(len(rbf.neurons))]
        else:
            rbf.neurons[self.random_state.randint(
                len(rbf.neurons))].delete_connection()
            rbf.neurons[self.random_state.randint(
                len(rbf.neurons))].add_connection()

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
        new = self.clone()
        if self.random_state.random_sample() < 0.5:
            n_random = self.random_state.randint(1, len(new.neurons))
            new_neurons_0 = self.random_state.choice(new.neurons, n_random)
            n_random = self.random_state.randint(1, len(rbf.neurons))
            new_neurons_1 = self.random_state.choice(rbf.neurons, n_random)
            new.neurons = [n.clone() for n in new_neurons_0]
            new.neurons.extend([n.clone() for n in new_neurons_1])
            while len(new.neurons) > self.m_max:
                del new.neurons[self.random_state.randint(len(new.neurons))]
        else:
            for i in range(len(new.neurons)):
                if self.random_state.random_sample() < 0.2:
                    n_random = self.random_state.randint(len(rbf.neurons))
                    new.neurons[i] = rbf.neurons[n_random].clone()
        return new


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
                 nn_params={},
                 m_min=4,
                 m_max=10,
                 Ib=2,
                 Ob=2,
                 n_pop=500,
                 n_init_pop=5000,
                 n_iter=40,
                 n_sampling_epoch=5,
                 n_jobs=1,
                 random_state=None):
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
            m_min (int): minimum number of hidden units
            m_max (int): maximum number of hidden units
            Ib (float): input weight range
            Ob (float): output weight range
            n_pop (int): size of population
            n_init_pop (int): size of initial population
            n_iter (int): number of iterations
            n_sampling_epoch (int): resampling after this many iterations
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(m_min, "m_min", 1)
        self.check_greater_or_equal(m_max, "m_max", 1)
        self.check_greater(Ib, "Ib", 0)
        self.check_greater(Ob, "Ob", 0)
        self.check_greater_or_equal(n_pop, "n_pop", 2)
        self.check_greater_or_equal(n_init_pop, "n_pop", 2)
        self.check_greater_or_equal(n_iter, "n_iter", 0)
        self.check_greater_or_equal(n_sampling_epoch, "n_sampling_epoch", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.m_min = m_min
        self.m_max = m_max
        self.Ib = Ib
        self.Ob = Ob
        self.n_pop = n_pop
        self.n_init_pop = n_init_pop
        self.n_iter = n_iter
        self.n_sampling_epoch = n_sampling_epoch
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

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
                                  'm_min': [4],
                                  'm_max': [10],
                                  'Ib': [2.0],
                                  'Ob': [2.0],
                                  'n_pop': [100],
                                  'n_init_pop': [1000],
                                  'n_iter': [40],
                                  'n_sampling_epoch': [8]}
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

        # Standardizing the data to let the network work with comparable
        # attributes
        X_orig0, y_orig0 = X, y

        ss = StandardScaler()
        X = ss.fit_transform(X)

        X_orig, y_orig = X, y

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        X, y = SMOTE(proportion=self.proportion,
                     n_neighbors=self.n_neighbors,
                     nn_params=nn_params,
                     n_jobs=self.n_jobs,
                     random_state=self._random_state_init).sample(X, y)

        # generate initial connections and weights randomly
        domain = np.arange(len(X[0]))
        n_random = int(len(X[0])/2)

        # setting epoch lengths
        epoch_len = int(self.n_iter/self.n_sampling_epoch)

        if len(X_orig) < self.m_min + 1:
            return X_orig.copy(), y_orig.copy()
        m_max = min(len(X_orig), self.m_max)

        if self.m_min >= m_max:
            _logger.warning(self.__class__.__name__ + ": " +
                     "Range of the number of hidden units to small %s" % self.descriptor())
            return X_orig0.copy(), y_orig0.copy()

        # generating initial population
        def init_pop():
            init_conn_mask = self.random_state.choice(domain, n_random)
            init_conn_weights = self.random_state.random_sample(size=n_random)

            return RBF(X,
                       self.m_min,
                       m_max,
                       self.Ib,
                       self.Ob,
                       init_conn_mask,
                       init_conn_weights,
                       random_state=self._random_state_init)

        population = [init_pop() for _ in range(self.n_init_pop)]
        population = [[p, X, y, np.inf] for p in population]
        population = sorted([[p[0], p[1], p[2], p[0].evaluate(p[1], p[2])]
                             for p in population], key=lambda x: x[3])
        population = population[:self.n_pop]

        # executing center improval in the hidden units
        for p in population:
            p[0].improve_centers()

        # executing the optimization process
        for iteration in range(self.n_iter):
            message = "Iteration %d/%d, loss: %f, data size %d"
            message = message % (iteration, self.n_iter, population[0][3],
                                 len(population[0][1]))
            _logger.info(self.__class__.__name__ + ": " + message)
            # evaluating non-evaluated elements
            for p in population:
                if p[3] == np.inf:
                    p[3] = p[0].evaluate(p[1], p[2])

            # sorting the population by the loss values
            population = sorted([p for p in population], key=lambda x: x[3])
            population = population[:self.n_pop]

            # determining the number of elements to be changed
            p_best = population[0]
            p_parametric_mut = population[:int(0.1*self.n_pop)]
            p_structural_mut = population[:int(0.9*self.n_pop-1)]
            p_recombination = population[:int(0.1*self.n_pop)]

            # executing mutation
            for p in p_parametric_mut:
                population.append([p[0].mutation(), p[1], p[2], np.inf])

            # executing structural mutation
            for p in p_structural_mut:
                population.append(
                    [p[0].structural_mutation(), p[1], p[2], np.inf])

            # executing recombination
            for p in p_recombination:
                domain = range(len(p_recombination))
                p_rec_idx = self.random_state.choice(domain)
                p_rec = p_recombination[p_rec_idx][0]
                population.append([p[0].recombine(p_rec), p[1], p[2], np.inf])

            # do the sampling
            if iteration % epoch_len == 0:
                smote = SMOTE(proportion=self.proportion,
                              n_neighbors=self.n_neighbors,
                              nn_params=nn_params,
                              n_jobs=self.n_jobs,
                              random_state=self._random_state_init)
                X, y = smote.sample(X_orig, y_orig)
                for i in range(self.n_pop):
                    tmp = [population[i][0].clone(), X, y, np.inf]
                    tmp[0].update_data(X)
                    tmp[0].improve_centers()
                    population.append(tmp)

        # evaluate unevaluated elements of the population
        for p in population:
            if p[3] == np.inf:
                p[3] = p[0].evaluate(p[1], p[2])

        # sorting the population
        population = sorted([p for p in population],
                            key=lambda x: x[3])[:self.n_pop]

        return ss.inverse_transform(p_best[1]), p_best[2]

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'm_min': self.m_min,
                'm_max': self.m_max,
                'Ib': self.Ib,
                'Ob': self.Ob,
                'n_pop': self.n_pop,
                'n_init_pop': self.n_init_pop,
                'n_iter': self.n_iter,
                'n_sampling_epoch': self.n_sampling_epoch,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

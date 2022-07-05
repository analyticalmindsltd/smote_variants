import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
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
                 eps=0.05,
                 n_pop=10,
                 w=1.0,
                 c1=2.0,
                 c2=2.0,
                 num_it=10,
                 n_jobs=1,
                 random_state=None):
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
        super().__init__()
        self.check_greater_or_equal(k, "k", 1)
        self.check_greater(eps, "eps", 0)
        self.check_greater_or_equal(n_pop, "n_pop", 1)
        self.check_greater_or_equal(w, "w", 0)
        self.check_greater_or_equal(c1, "c1", 0)
        self.check_greater_or_equal(c2, "c2", 0)
        self.check_greater_or_equal(num_it, "num_it", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.k = k
        self.nn_params = nn_params
        self.eps = eps
        self.n_pop = n_pop
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.num_it = num_it
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        return cls.generate_parameter_combinations({'k': [3, 5, 7],
                                                    'eps': [0.05],
                                                    'n_pop': [5],
                                                    'w': [0.5, 1.0],
                                                    'c1': [1.0, 2.0],
                                                    'c2': [1.0, 2.0],
                                                    'num_it': [5]}, raw)

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

        # saving original dataset
        X_orig = X
        y_orig = y

        # scaling the records
        mms = MinMaxScaler()
        X_scaled = mms.fit_transform(X)

        # removing majority and minority samples far from the training data if
        # needed to increase performance
        performance_threshold = 500

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X_scaled, y)

        n_maj_to_remove = np.sum(
            y == self.maj_label) - performance_threshold
        if n_maj_to_remove > 0:
            # if majority samples are to be removed
            nn= NearestNeighborsWithMetricTensor(n_neighbors=1, 
                                                    n_jobs=self.n_jobs, 
                                                    **nn_params)
            nn.fit(X_scaled[y == self.min_label])
            dist, ind = nn.kneighbors(X_scaled)
            di = sorted([(dist[i][0], i)
                         for i in range(len(ind))], key=lambda x: x[0])
            to_remove = []
            # finding the proper number of samples farest from the minority
            # samples
            for i in reversed(range(len(di))):
                if y[di[i][1]] == self.maj_label:
                    to_remove.append(di[i][1])
                if len(to_remove) >= n_maj_to_remove:
                    break
            # removing the samples
            X_scaled = np.delete(X_scaled, to_remove, axis=0)
            y = np.delete(y, to_remove)

        n_min_to_remove = np.sum(
            y == self.min_label) - performance_threshold
        if n_min_to_remove > 0:
            # if majority samples are to be removed
            nn = NearestNeighborsWithMetricTensor(n_neighbors=1, 
                                                    n_jobs=self.n_jobs, 
                                                    **nn_params)
            nn.fit(X_scaled[y == self.maj_label])
            dist, ind = nn.kneighbors(X_scaled)
            di = sorted([(dist[i][0], i)
                         for i in range(len(ind))], key=lambda x: x[0])
            to_remove = []
            # finding the proper number of samples farest from the minority
            # samples
            for i in reversed(range(len(di))):
                if y[di[i][1]] == self.min_label:
                    to_remove.append(di[i][1])
                if len(to_remove) >= n_min_to_remove:
                    break
            # removing the samples
            X_scaled = np.delete(X_scaled, to_remove, axis=0)
            y = np.delete(y, to_remove)

        # fitting SVM to extract initial support vectors
        svc = SVC(kernel='rbf', probability=True,
                  gamma='auto', random_state=self._random_state_init)
        svc.fit(X_scaled, y)

        # extracting the support vectors
        SV_min = np.array(
            [i for i in svc.support_ if y[i] == self.min_label])
        SV_maj = np.array(
            [i for i in svc.support_ if y[i] == self.maj_label])

        X_SV_min = X_scaled[SV_min]
        X_SV_maj = X_scaled[SV_maj]

        # finding nearest majority support vectors
        n_neighbors = min([len(X_SV_maj), self.k])
        nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **nn_params)
        nn.fit(X_SV_maj)
        dist, ind = nn.kneighbors(X_SV_min)

        # finding the initial particle and specifying the search space
        X_min_gen = []
        search_space = []
        init_velocity = []
        for i in range(len(SV_min)):
            for j in range(min([len(X_SV_maj), self.k])):
                min_vector = X_SV_min[i]
                maj_vector = X_SV_maj[ind[i][j]]
                # the upper bound of the search space if specified by the
                # closest majority support vector
                upper_bound = X_SV_maj[ind[i][0]]
                # the third element of the search space specification is
                # the distance of the vector and the closest
                # majority support vector, which specifies the radius of
                # the search
                norms = np.linalg.norm(min_vector - upper_bound)
                search_space.append([min_vector, maj_vector, norms])
                # initial particles
                X_min_gen.append(min_vector + self.eps *
                                 (maj_vector - min_vector))
                # initial velocities
                init_velocity.append(self.eps*(maj_vector - min_vector))

        X_min_gen = np.vstack(X_min_gen)
        init_velocity = np.vstack(init_velocity)

        # evaluates a specific particle
        def evaluate(X_train, y_train, X_test, y_test):
            """
            Trains support vector classifier and evaluates it

            Args:
                X_train (np.matrix): training vectors
                y_train (np.array): target labels
                X_test (np.matrix): test vectors
                y_test (np.array): test labels
            """
            svc.fit(X_train, y_train)
            y_pred = svc.predict_proba(X_test)[:, np.where(
                svc.classes_ == self.min_label)[0][0]]
            return roc_auc_score(y_test, y_pred)

        # initializing the particle swarm and the particle and population level
        # memory
        particle_swarm = [X_min_gen.copy() for _ in range(self.n_pop)]
        velocities = [init_velocity.copy() for _ in range(self.n_pop)]
        local_best = [X_min_gen.copy() for _ in range(self.n_pop)]
        local_best_scores = [0.0]*self.n_pop
        global_best = X_min_gen.copy()
        global_best_score = 0.0

        def evaluate_particle(X_scaled, p, y):
            X_extended = np.vstack([X_scaled, p])
            y_extended = np.hstack([y, np.repeat(self.min_label, len(p))])
            return evaluate(X_extended, y_extended, X_scaled, y)

        for i in range(self.num_it):
            _logger.info(self.__class__.__name__ + ": " + "Iteration %d" % i)
            # evaluate population
            scores = [evaluate_particle(X_scaled, p, y)
                      for p in particle_swarm]

            # update best scores
            for i, s in enumerate(scores):
                if s > local_best_scores[i]:
                    local_best_scores[i] = s
                    local_best[i] = particle_swarm[i]
                if s > global_best_score:
                    global_best_score = s
                    global_best = particle_swarm[i]

            # update velocities
            for i, p in enumerate(particle_swarm):
                term_0 = self.w*velocities[i]
                random_1 = self.random_state.random_sample()
                random_2 = self.random_state.random_sample()
                term_1 = self.c1*random_1*(local_best[i] - p)
                term_2 = self.c2*random_2*(global_best - p)

                velocities[i] = term_0 + term_1 + term_2

            # bound velocities according to search space constraints
            for v in velocities:
                for i in range(len(v)):
                    v_i_norm = np.linalg.norm(v[i])
                    if v_i_norm > search_space[i][2]/2.0:
                        v[i] = v[i]/v_i_norm*search_space[i][2]/2.0

            # update positions
            for i, p in enumerate(particle_swarm):
                particle_swarm[i] = particle_swarm[i] + velocities[i]

            # bound positions according to search space constraints
            for p in particle_swarm:
                for i in range(len(p)):
                    ss = search_space[i]

                    trans_vector = p[i] - ss[0]
                    trans_norm = np.linalg.norm(trans_vector)
                    normed_trans = trans_vector/trans_norm

                    if trans_norm > ss[2]:
                        p[i] = ss[0] + normed_trans*ss[2]

        X_ret = np.vstack([X_orig, mms.inverse_transform(global_best)])
        y_ret = np.hstack(
            [y_orig, np.repeat(self.min_label, len(global_best))])

        return (X_ret, y_ret)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'k': self.k,
                'nn_params': self.nn_params,
                'eps': self.eps,
                'n_pop': self.n_pop,
                'w': self.w,
                'c1': self.c1,
                'c2': self.c2,
                'num_it': self.num_it,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

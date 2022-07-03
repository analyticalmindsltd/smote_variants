import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances

from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['CURE_SMOTE']

class CURE_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @Article{cure_smote,
                        author="Ma, Li
                        and Fan, Suohai",
                        title="CURE-SMOTE algorithm and hybrid algorithm for
                                feature selection and parameter optimization
                                based on random forests",
                        journal="BMC Bioinformatics",
                        year="2017",
                        month="Mar",
                        day="14",
                        volume="18",
                        number="1",
                        pages="169",
                        issn="1471-2105",
                        doi="10.1186/s12859-017-1578-z",
                        url="https://doi.org/10.1186/s12859-017-1578-z"
                        }

    Notes:
        * It is not specified how to determine the cluster with the
            "slowest growth rate"
        * All clusters can be removed as noise.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering]

    def __init__(self,
                 proportion=1.0,
                 *,
                 n_clusters=5,
                 noise_th=2,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_clusters (int): number of clusters to generate
            noise_th (int): below this number of elements the cluster is
                                considered as noise
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_clusters, "n_clusters", 1)
        self.check_greater_or_equal(noise_th, "noise_th", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_clusters = n_clusters
        self.noise_th = noise_th
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_clusters': [5, 10, 15],
                                  'noise_th': [1, 3]}

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

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # standardizing the data
        mms = MinMaxScaler()
        X_scaled = mms.fit_transform(X)

        X_min = X_scaled[y == self.min_label]

        # initiating clustering
        clusters = [np.array([i]) for i in range(len(X_min))]
        dm = pairwise_distances(X_min)

        # setting the diagonal of the distance matrix to infinity
        for i in range(len(dm)):
            dm[i, i] = np.inf

        # starting the clustering iteration
        iteration = 0
        while len(clusters) > self.n_clusters:
            iteration = iteration + 1

            # delete a cluster with slowest growth rate, determined by
            # the cluster size
            if iteration % self.n_clusters == 0:
                # extracting cluster sizes
                cluster_sizes = np.array([len(c) for c in clusters])
                # removing one of the clusters with the smallest size
                to_remove = np.where(cluster_sizes == np.min(cluster_sizes))[0]
                to_remove = self.random_state.choice(to_remove)
                del clusters[to_remove]
                # adjusting the distance matrix accordingly
                dm = np.delete(dm, to_remove, axis=0)
                dm = np.delete(dm, to_remove, axis=1)

            # finding the cluster pair with the smallest distance
            min_coord = np.where(dm == np.min(dm))
            merge_a = min_coord[0][0]
            merge_b = min_coord[1][0]

            # merging the clusters
            clusters[merge_a] = np.hstack(
                [clusters[merge_a], clusters[merge_b]])
            # removing one of them
            del clusters[merge_b]
            # adjusting the distances in the distance matrix
            dm[merge_a] = np.min(np.vstack([dm[merge_a], dm[merge_b]]), axis=0)
            dm[:, merge_a] = dm[merge_a]
            # removing the row and column corresponding to one of
            # the merged clusters
            dm = np.delete(dm, merge_b, axis=0)
            dm = np.delete(dm, merge_b, axis=1)
            # updating the diagonal
            for i in range(len(dm)):
                dm[i, i] = np.inf

        # removing clusters declared as noise
        to_remove = []
        for i in range(len(clusters)):
            if len(clusters[i]) < self.noise_th:
                to_remove.append(i)
        clusters = [clusters[i]
                    for i in range(len(clusters)) if i not in to_remove]

        # all clusters can be noise
        if len(clusters) == 0:
            _logger.warning(self.__class__.__name__ + ": " +
                            "all clusters removed as noise")
            return X.copy(), y.copy()

        # generating samples
        samples = []
        for _ in range(n_to_sample):
            cluster_idx = self.random_state.randint(len(clusters))
            center = np.mean(X_min[clusters[cluster_idx]], axis=0)
            representative = X_min[self.random_state.choice(
                clusters[cluster_idx])]
            samples.append(self.sample_between_points(center, representative))

        return (np.vstack([X, mms.inverse_transform(np.vstack(samples))]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_clusters': self.n_clusters,
                'noise_th': self.noise_th,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


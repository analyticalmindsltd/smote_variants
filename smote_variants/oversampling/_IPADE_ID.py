import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

from ._OverSampling import OverSampling
from .._base import mode

from .._logger import logger
_logger= logger

__all__= ['IPADE_ID']

class IPADE_ID(OverSampling):
    """
    References:
        * BibTex::

            @article{ipade_id,
                    title = "Addressing imbalanced classification with
                                instance generation techniques: IPADE-ID",
                    journal = "Neurocomputing",
                    volume = "126",
                    pages = "15 - 28",
                    year = "2014",
                    note = "Recent trends in Intelligent Data Analysis Online
                                Data Processing",
                    issn = "0925-2312",
                    doi = "https://doi.org/10.1016/j.neucom.2013.01.050",
                    author = "Victoria López and Isaac Triguero and Cristóbal
                                J. Carmona and Salvador García and
                                Francisco Herrera",
                    keywords = "Differential evolution, Instance generation,
                                Nearest neighbor, Decision tree, Imbalanced
                                datasets"
                    }

    Notes:
        * According to the algorithm, if the addition of a majority sample
            doesn't improve the AUC during the DE optimization process,
            the addition of no further majority points is tried.
        * In the differential evolution the multiplication by a random number
            seems have a deteriorating effect, new scaling parameter added to
            fix this.
        * It is not specified how to do the evaluation.
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_memetic,
                  OverSampling.cat_uses_classifier]

    def __init__(self,
                 *,
                 F=0.1,
                 G=0.1,
                 OT=20,
                 max_it=40,
                 dt_classifier=DecisionTreeClassifier(random_state=2),
                 base_classifier=DecisionTreeClassifier(random_state=2),
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            F (float): control parameter of differential evolution
            G (float): control parameter of the evolution
            OT (int): number of optimizations
            max_it (int): maximum number of iterations for DE_optimization
            dt_classifier (obj): decision tree classifier object
            base_classifier (obj): classifier object
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater(F, 'F', 0)
        self.check_greater(G, 'G', 0)
        self.check_greater(OT, 'OT', 0)
        self.check_greater(max_it, 'max_it', 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.F = F
        self.G = G
        self.OT = OT
        self.max_it = max_it
        self.dt_classifier = dt_classifier
        self.base_classifier = base_classifier
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        # as the OT and max_it parameters control the discovery of the feature
        # space it is enough to try sufficiently large numbers
        dt_classifiers = [DecisionTreeClassifier(random_state=2)]
        base_classifiers = [DecisionTreeClassifier(random_state=2)]
        parameter_combinations = {'F': [0.1, 0.2],
                                  'G': [0.1, 0.2],
                                  'OT': [30],
                                  'max_it': [40],
                                  'dt_classifier': dt_classifiers,
                                  'base_classifier': base_classifiers}
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

        if not self.check_enough_min_samples_for_sampling(3):
            return X.copy(), y.copy()

        mms = MinMaxScaler()
        X = mms.fit_transform(X)

        min_indices = np.where(y == self.min_label)[0]
        maj_indices = np.where(y == self.maj_label)[0]

        def DE_optimization(GS,
                            GS_y,
                            X,
                            y,
                            min_indices,
                            maj_indices,
                            classifier,
                            for_validation):
            """
            Implements the DE_optimization method of the paper.

            Args:
                GS (np.matrix): actual best training set
                GS_y (np.array): corresponding class labels
                X (np.matrix): complete training set
                y (np.array): all class labels
                min_indices (np.array): array of minority class labels in y
                maj_indices (np.array): array of majority class labels in y
                classifier (object): base classifier
                for_validation (np.array): array of indices for X used for
                                            validation

            Returns:
                np.matrix: optimized training set
            """
            # evaluate training set
            AUC_GS = evaluate_ID(
                GS, GS_y, X[for_validation], y[for_validation], classifier)

            # optimizing the training set
            for _ in range(self.max_it):
                GS_hat = []
                # doing the differential evolution
                for i in range(len(GS)):
                    if GS_y[i] == self.min_label:
                        r1, r2, r3 = self.random_state.choice(min_indices,
                                                              3,
                                                              replace=False)
                    else:
                        r1, r2, r3 = self.random_state.choice(maj_indices,
                                                              3,
                                                              replace=False)

                    random_value = self.random_state.random_sample()
                    force_G = X[r1] - X[i]
                    force_F = X[r2] - X[r3]
                    value = GS[i] + self.G*random_value * \
                        force_G + self.F*force_F
                    GS_hat.append(np.clip(value, 0.0, 1.0))

                # evaluating the current setting
                AUC_GS_hat = evaluate_ID(GS_hat,
                                         GS_y,
                                         X[for_validation],
                                         y[for_validation],
                                         classifier)

                if AUC_GS_hat > AUC_GS:
                    GS = GS_hat
                    AUC_GS = AUC_GS_hat

            return GS

        def evaluate_ID(GS, GS_y, TR, TR_y, base_classifier):
            """
            Implements the evaluate_ID function of the paper.

            Args:
                GS (np.matrix): actual training set
                GS_y (np.array): list of corresponding class labels
                TR (np.matrix): complete training set
                TR_y (np.array): all class labels
                base_classifier (object): classifier to be used

            Returns:
                float: ROC AUC score
            """
            base_classifier.fit(GS, GS_y)
            pred = base_classifier.predict_proba(TR)[:, np.where(
                base_classifier.classes_ == self.min_label)[0][0]]
            if len(np.unique(TR_y)) != 2:
                return 0.0
            return roc_auc_score(TR_y, pred)

        def evaluate_class(GS, GS_y, TR, TR_y, base_classifier):
            """
            Implements the evaluate_ID function of the paper.

            Args:
                GS (np.matrix): actual training set
                GS_y (np.array): list of corresponding class labels
                TR (np.matrix): complete training set
                TR_y (np.array): all class labels
                base_classifier (object): classifier to be used

            Returns:
                float: accuracy score
            """
            base_classifier.fit(GS, GS_y)
            pred = base_classifier.predict(TR)
            return accuracy_score(TR_y, pred)

        # Phase 1: Initialization
        _logger.info(self.__class__.__name__ + ": " + "Initialization")
        self.dt_classifier.fit(X, y)
        leafs = self.dt_classifier.apply(X)
        unique_leafs = np.unique(leafs)
        used_in_GS = np.repeat(False, len(X))
        for_validation = np.where(np.logical_not(used_in_GS))[0]

        # extracting mean elements of the leafs
        GS = []
        GS_y = []
        for u in unique_leafs:
            indices = np.where(leafs == u)[0]
            GS.append(np.mean(X[indices], axis=0))
            GS_y.append(mode(y[indices]))
            if len(indices) == 1:
                used_in_GS[indices[0]] = True

        # updating the indices of the validation set excluding those used in GS
        for_validation = np.where(np.logical_not(used_in_GS))[0]
        _logger.info(self.__class__.__name__ + ": " +
                     "Size of validation set %d" % len(for_validation))
        if len(np.unique(y[for_validation])) == 1:
            _logger.info(self.__class__.__name__ + ": " +
                         "No minority samples in validation set")
            return X.copy(), y.copy()
        if len(np.unique(GS_y)) == 1:
            _logger.info(self.__class__.__name__ + ": " +
                         "No minority samples in reduced dataset")
            return X.copy(), y.copy()

        # DE optimization takes place
        _logger.info(self.__class__.__name__ + ": " + "DE optimization")
        base_classifier = self.base_classifier.__class__(
            **(self.base_classifier.get_params()))
        GS = DE_optimization(GS, GS_y, X, y, min_indices,
                             maj_indices, base_classifier, for_validation)
        # evaluate results
        base_classifier = self.base_classifier.__class__(
            **(self.base_classifier.get_params()))
        AUC = evaluate_ID(GS, GS_y, X[for_validation],
                          y[for_validation], base_classifier)

        # Phase 2: Addition of new instances
        register_class = {self.min_label: 'optimizable',
                          self.maj_label: 'optimizable'}
        number_of_optimizations = {self.min_label: 0,
                                   self.maj_label: 0}
        accuracy_class = {self.min_label: 0, self.maj_label: 0}

        _logger.info(self.__class__.__name__ + ": " + "Starting optimization")
        while (AUC < 1.0
                and (register_class[self.min_label] == 'optimizable'
                     or register_class[self.maj_label] == 'optimizable')):
            less_accuracy = np.inf
            # loop in line 8
            for i in [self.min_label, self.maj_label]:
                # condition in line 9
                if register_class[i] == 'optimizable':
                    y_mask = y[for_validation] == i
                    class_for_validation = for_validation[y_mask]
                    bp = self.base_classifier.get_params()
                    base_classifier = self.base_classifier.__class__(**(bp))
                    accuracy_class[i] = evaluate_class(GS,
                                                       GS_y,
                                                       X[class_for_validation],
                                                       y[class_for_validation],
                                                       base_classifier)
                    if accuracy_class[i] < less_accuracy:
                        less_accuracy = accuracy_class[i]
                        target_class = i
            # conditional in line 17
            if (target_class == self.min_label
                    and number_of_optimizations[target_class] > 0):
                # it is not clear where does GS_trial coming from in line 18
                GS = DE_optimization(GS,
                                     GS_y,
                                     X,
                                     y,
                                     min_indices,
                                     maj_indices,
                                     base_classifier,
                                     for_validation)
            else:
                if target_class == self.min_label:
                    idx = self.random_state.choice(min_indices)
                else:
                    idx = self.random_state.choice(maj_indices)

                GS_trial = np.vstack([GS, X[idx]])
                GS_trial_y = np.hstack([GS_y, y[idx]])
                # removing idx from the validation set in order to keep
                # the validation fair
                for_validation_trial = for_validation.tolist()
                if idx in for_validation:
                    for_validation_trial.remove(idx)

                for_validation_trial = np.array(
                    for_validation_trial).astype(int)
                # doing optimization
                GS_trial = DE_optimization(GS_trial,
                                           GS_trial_y,
                                           X,
                                           y,
                                           min_indices,
                                           maj_indices,
                                           base_classifier,
                                           for_validation)

            # line 23
            bp = self.base_classifier.get_params()
            base_classifier = self.base_classifier.__class__(**(bp))

            AUC_trial = evaluate_ID(GS_trial,
                                    GS_trial_y,
                                    X[for_validation],
                                    y[for_validation],
                                    base_classifier)
            # conditional in line 24
            if AUC_trial > AUC:
                AUC = AUC_trial
                GS = GS_trial
                GS_y = GS_trial_y
                for_validation = for_validation_trial

                _logger.info(self.__class__.__name__ + ": " +
                             "Size of validation set %d" % len(for_validation))
                if len(np.unique(y[for_validation])) == 1:
                    _logger.info(self.__class__.__name__ + ": " +
                                 "No minority samples in validation set")
                    return X.copy(), y.copy()
                if len(np.unique(GS_y)) == 1:
                    _logger.info(self.__class__.__name__ + ": " +
                                 "No minority samples in reduced dataset")
                    return X.copy(), y.copy()

                number_of_optimizations[target_class] = 0
            else:
                # conditional in line 29
                if (target_class == self.min_label
                        and number_of_optimizations[target_class] < self.OT):
                    number_of_optimizations[target_class] += 1
                else:
                    register_class[target_class] = 'non-optimizable'

        return mms.inverse_transform(GS), GS_y

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'F': self.F,
                'G': self.G,
                'OT': self.OT,
                'max_it': self.max_it,
                'n_jobs': self.n_jobs,
                'dt_classifier': self.dt_classifier,
                'base_classifier': self.base_classifier,
                'random_state': self._random_state_init}

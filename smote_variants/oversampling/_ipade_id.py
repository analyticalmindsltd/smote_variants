"""
This module implements the IPADE_ID method.
"""
from dataclasses import dataclass

import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from ..base import OverSampling
from ..base import mode, instantiate_obj

from .._logger import logger
_logger= logger

__all__= ['IPADE_ID']

@dataclass
class TrainingSet:
    """
    Represents a trining set
    """
    X: np.array
    y: np.array

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
        * When the dt classifier is grown to full depth, each leaf contains
            only one element, and there are no independent elements left
            for validation.
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
                 dt_classifier=('sklearn.tree',
                                'DecisionTreeClassifier',
                                {'random_state': 2}),
                 base_classifier=('sklearn.tree',
                                'DecisionTreeClassifier',
                                {'random_state': 2}),
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
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
        super().__init__(random_state=random_state, checks={'min_n_min': 4})
        self.check_greater(F, 'F', 0)
        self.check_greater(G, 'G', 0)
        self.check_greater(OT, 'OT', 0)
        self.check_greater(max_it, 'max_it', 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.de_params = {'F': F,
                            'G': G,
                            'OT': OT,
                            'max_it': max_it,
                            'dt_classifier': dt_classifier,
                            'base_classifier': base_classifier}

        self.n_jobs = n_jobs

        self.dt_classifier_obj = instantiate_obj(dt_classifier)
        self.base_classifier_obj = instantiate_obj(base_classifier)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        # as the OT and max_it parameters control the discovery of the feature
        # space it is enough to try sufficiently large numbers
        dt_classifiers = [('sklearn.tree',
                            'DecisionTreeClassifier',
                            {'random_state': 2})]
        base_classifiers = [('sklearn.tree',
                            'DecisionTreeClassifier',
                            {'random_state': 2})]
        parameter_combinations = {'F': [0.1, 0.2],
                                  'G': [0.1, 0.2],
                                  'OT': [30],
                                  'max_it': [40],
                                  'dt_classifier': dt_classifiers,
                                  'base_classifier': base_classifiers}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def update_element(self, *, GS_y, idx, GS_x, indices, TR_X):
        """
        Updates the GS_idx vector

        Args:
            GS_y (np.array): the y labels in the optimization set
            idx (np.array): index of the vector to update
            GS_x (np.array): the vector to update
            indices (dict): indices of the various classes
            TR_X (np.array): the original vectors

        Returns:
            np.array: the updated GS_x vector
        """
        rand_idx = self.random_state.choice(indices[GS_y[idx]],
                                            3,
                                            replace=False)

        random = self.random_state.random_sample()
        value = GS_x + self.de_params['G'] * random\
                                        * (TR_X[rand_idx[0]] - GS_x)\
                            + self.de_params['F']\
                                        * (TR_X[rand_idx[1]] - TR_X[rand_idx[2]])

        return value

    def DE_optimization(self, # pylint: disable=invalid-name
                            *,
                            GS, # pylint: disable=invalid-name
                            TR,
                            indices,
                            for_validation):
        """
        Implements the DE_optimization method of the paper.
        Roughly, for each point of GS takes 3 random samples from the same
        class and moves the actual point towards them, then
        evaluates the new configuration.

        Args:
            GS (TrainingSet): the optimized training set
            TR (TrainingSet): the original training set
            indices (dict): indices of the various class labels
            for_validation (np.array): array of indices for X used for
                                        validation

        Returns:
            TrainingSet: optimized training set
        """

        # evaluate training set
        AUC_GS = self.evaluate_ID(GS=GS, # pylint: disable=invalid-name
                                    TR=TrainingSet(TR.X[for_validation],
                                                    TR.y[for_validation]))

        # optimizing the training set
        for _ in range(self.de_params['max_it']):
            GS_hat = [] # pylint: disable=invalid-name
            # doing the differential evolution
            for idx, GS_x in enumerate(GS.X): # pylint: disable=invalid-name
                value = self.update_element(GS_y=GS.y,
                                            idx=idx,
                                            GS_x=GS_x,
                                            indices=indices,
                                            TR_X=TR.X)

                GS_hat.append(np.clip(value, 0.0, 1.0))

            GS_hat = np.vstack(GS_hat) # pylint: disable=invalid-name

            # evaluating the current setting
            AUC_GS_hat = self.evaluate_ID(GS=TrainingSet(GS_hat, GS.y), # pylint: disable=invalid-name
                                        TR=TrainingSet(TR.X[for_validation],
                                                        TR.y[for_validation]))

            if AUC_GS_hat > AUC_GS:
                GS = TrainingSet(GS_hat, GS.y)
                AUC_GS = AUC_GS_hat # pylint: disable=invalid-name

        return GS

    def evaluate_ID(self, *, GS, TR): # pylint: disable=invalid-name
        """
        Implements the evaluate_ID function of the paper.

        Args:
            GS (TrainingSet): the optimized training set
            TR (TrainingSet): the original training set

        Returns:
            float: ROC AUC score
        """

        base_classifier = self.new_base_classifier()

        base_classifier.fit(GS.X, GS.y)
        min_index = np.where(base_classifier.classes_ == self.min_label)[0][0]
        pred = base_classifier.predict_proba(TR.X)[:, min_index]

        return roc_auc_score(TR.y, pred)

    def evaluate_class(self, *, GS, TR):
        """
        Implements the classification evaluation function of the paper.

        Args:
            GS (TrainingSet): the optimized training set
            TR (TrainingSet): the original training set

        Returns:
            float: accuracy score
        """
        base_classifier = self.new_base_classifier()

        base_classifier.fit(GS.X, GS.y)
        pred = base_classifier.predict(TR.X)

        return accuracy_score(TR.y, pred)

    def new_base_classifier(self):
        """
        Instantiates a new instance of the base classifier

        Returns:
            obj: a new instance of the base classifier
        """
        class_ = self.base_classifier_obj.__class__
        return class_(**(self.base_classifier_obj.get_params()))

    def configuration_checks(self,
                                y,
                                for_validation,
                                GS_y # pylint: disable=invalid-name
                                ):
        """
        Checks if the configuration is valid.

        Args:
            y (np.array): target labels
            for_validation (np.array): the indices of validation samples
            GS_y (np.array): the updated target labels
        """
        if len(np.unique(y[for_validation])) == 1:
            _logger.info("%s: No minority samples in validation set",
                        self.__class__.__name__)
            raise ValueError("No minority samples in validation set")

        if len(np.unique(GS_y)) == 1:
            _logger.info("%s: No minority samples in reduced dataset",
                            self.__class__.__name__)
            raise ValueError("No minority samples in reduced dataset")

    def phase_1(self,
                TR, # pylint: disable=invalid-name
                indices):
        """
        Initialization of the method.

        Args:
            TR (TrainingSet): the original training set
            indices (dict): the indices of the various classes

        Returns:
            float, np.array, TrainingSet: the AUC score, the validation
                                        indices and the initial optimized
                                        training set
        """
        _logger.info("%s: Initialization", self.__class__.__name__)

        self.dt_classifier_obj.fit(TR.X, TR.y)
        leafs = self.dt_classifier_obj.apply(TR.X)
        unique_leafs = np.unique(leafs)

        used_in_GS = np.repeat(False, len(TR.X)) # pylint: disable=invalid-name
        for_validation = np.where(np.logical_not(used_in_GS))[0]

        # extracting mean elements of the leafs
        GS_X = [] # pylint: disable=invalid-name
        GS_y = [] # pylint: disable=invalid-name
        for leaf in unique_leafs:
            leaf_indices = np.where(leafs == leaf)[0]
            GS_X.append(np.mean(TR.X[leaf_indices], axis=0))
            GS_y.append(mode(TR.y[leaf_indices]))
            if len(leaf_indices) == 1:
                used_in_GS[leaf_indices[0]] = True

        GS = TrainingSet(np.vstack(GS_X), np.array(GS_y)) # pylint: disable=invalid-name

        # updating the indices of the validation set excluding those used in GS
        for_validation = np.where(np.logical_not(used_in_GS))[0]

        _logger.info("%s: Size of validation set %d",
                    self.__class__.__name__, len(for_validation))
        self.configuration_checks(TR.y, for_validation, GS.y)


        # DE optimization takes place
        _logger.info("%s: DE optimization", self.__class__.__name__)

        GS = self.DE_optimization(GS=GS, # pylint: disable=invalid-name
                                TR=TR,
                                indices=indices,
                                for_validation=for_validation)

        # evaluate results
        AUC = self.evaluate_ID(GS=GS, # pylint: disable=invalid-name
                                TR=TrainingSet(TR.X[for_validation], TR.y[for_validation]))

        return AUC, for_validation, GS

    def determine_target_class(self, *, TR, GS,
                                for_validation, register_class):
        """
        Determine the target class according to the description in the paper

        Args:
            TR (TrainingSet): the original training set
            GS (TrainingSet): the optimized training set
            for_validation (np.array): labels used for validation
            register_class (dict): the status of the class

        Returns:
            int: the target class label
        """
        accuracy_class = {self.min_label: 0, self.maj_label: 0}

        less_accuracy = np.inf

        # loop in line 8
        for idx in [self.min_label, self.maj_label]:
            # condition in line 9
            if register_class[idx] == 'optimizable':
                y_mask = TR.y[for_validation] == idx
                class_for_validation = for_validation[y_mask]
                accuracy_class[idx] = self.evaluate_class(GS=GS,
                                                    TR=TrainingSet(TR.X[class_for_validation],
                                                                    TR.y[class_for_validation]))
                if accuracy_class[idx] < less_accuracy:
                    less_accuracy = accuracy_class[idx]
                    target_class = idx

        return target_class

    def generate_trial(self, *, indices, target_class, TR, GS, for_validation):
        """
        Generates a trial training set

        Args:
            indices (dict): the indices of the various classes
            target_class (int): the target class label
            TR (TrainingSet): the original training set
            GS (TrainingSet): the optimized training set
            for_validation (np.array): the indices used for validation

        Returns:
            TrainingSet, np.array: the trial training set and the updated
                                    validation indices
        """
        idx = self.random_state.choice(indices[target_class])

        GS_trial_X = np.vstack([GS.X, TR.X[idx]]) # pylint: disable=invalid-name
        GS_trial_y = np.hstack([GS.y, TR.y[idx]]) # pylint: disable=invalid-name

        # removing idx from the validation set in order to keep
        # the validation fair
        for_validation_trial = for_validation[for_validation != idx]

        return TrainingSet(GS_trial_X, GS_trial_y), for_validation_trial

    def phase_2(self, *, TR, indices, for_validation, GS, AUC):
        """
        Phase 2 of the algorithm

        Args:
            TR (TrainingSet): original training set
            indices (dict): indices of the various classes
            for_validation (np.array): the mask of the validation samples
            GS (TrainingSet): the optimized training set
            AUC (float): the AUC score

        Returns:
            TrainingSet: the optimized training set
        """
        register_class = {self.min_label: 'optimizable',
                          self.maj_label: 'optimizable'}
        number_of_optimizations = {self.min_label: 0,
                                   self.maj_label: 0}

        _logger.info("%s: Starting optimization", self.__class__.__name__)

        GS_trial = None # pylint: disable=invalid-name

        while (AUC < 1.0
                and (register_class[self.min_label] == 'optimizable'
                     or register_class[self.maj_label] == 'optimizable')):

            target_class = self.determine_target_class(TR=TR,
                                                        GS=GS,
                                                        for_validation=for_validation,
                                                        register_class=register_class)

            # conditional in line 17
            if (target_class == self.min_label
                    and number_of_optimizations[target_class] > 0):
                # this is a tricky part, because GS_trial is defined later only
                GS_trial = self.DE_optimization(GS=GS_trial, # pylint: disable=invalid-name
                                                TR=TR,
                                                indices=indices,
                                                for_validation=for_validation)
            else:
                GS_trial, for_validation_trial = self.generate_trial( # pylint: disable=invalid-name
                                            indices=indices,
                                            target_class=target_class,
                                            TR=TR, GS=GS,
                                            for_validation=for_validation)


                # doing optimization
                GS_trial = self.DE_optimization(GS=GS, # pylint: disable=invalid-name
                                           TR=TR,
                                           indices=indices,
                                           for_validation=for_validation_trial)

            # line 23
            AUC_trial = self.evaluate_ID(GS=GS_trial, # pylint: disable=invalid-name
                                            TR=TrainingSet(TR.X[for_validation_trial],
                                                            TR.y[for_validation_trial]))
            # conditional in line 24
            if AUC_trial > AUC:
                AUC = AUC_trial
                GS = GS_trial
                for_validation = for_validation_trial

                _logger.info("%s: Size of validation set %d",
                        self.__class__.__name__, len(for_validation))
                self.configuration_checks(TR.y, for_validation, GS.y)

                number_of_optimizations[target_class] = 0
            else:
                # conditional in line 29
                if (target_class == self.min_label
                        and number_of_optimizations[target_class] < self.de_params['OT']):
                    number_of_optimizations[target_class] += 1
                else:
                    register_class[target_class] = 'non-optimizable'

        return GS

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        mms = MinMaxScaler()
        X = mms.fit_transform(X)

        min_indices = np.where(y == self.min_label)[0]
        maj_indices = np.where(y == self.maj_label)[0]

        indices = {self.min_label: min_indices,
                    self.maj_label: maj_indices}

        try:
            # Phase 1: Initialization
            AUC, for_validation, GS = self.phase_1(TrainingSet(X, y), indices) # pylint: disable=invalid-name

            # Phase 2: Addition of new instances
            GS = self.phase_2(TR=TrainingSet(X, y), # pylint: disable=invalid-name
                            indices=indices,
                            for_validation=for_validation,
                            GS=GS,
                            AUC=AUC)
        except ValueError as value:
            return self.return_copies(X, y, value.args[0])

        return mms.inverse_transform(GS.X), GS.y

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'F': self.de_params['F'],
                'G': self.de_params['G'],
                'OT': self.de_params['OT'],
                'max_it': self.de_params['max_it'],
                'n_jobs': self.n_jobs,
                'dt_classifier': self.de_params['dt_classifier'],
                'base_classifier': self.de_params['base_classifier'],
                **OverSampling.get_params(self)}

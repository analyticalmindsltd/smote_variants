import numpy as np
import pandas as pd

import os
import pickle
import re
import time
import glob
from joblib import Parallel, delayed

from scipy.stats.mstats import gmean

from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from .oversampling._OverSampling import OverSampling
from ._MLPClassifierWrapper import MLPClassifierWrapper

from ._smote_variants import *

from ._logger import logger
_logger= logger

__all__= ['Folding',
          'Sampling',
          'Evaluation',
          'read_oversampling_results',
          'evaluate_oversamplers',
          'model_selection',
          'cross_validate']

class Folding():
    """
    Cache-able folding of dataset for cross-validation
    """

    def __init__(self, dataset, validator, cache_path=None, random_state=None):
        """
        Constructor of Folding object

        Args:
            dataset (dict): dataset dictionary with keys 'data', 'target'
                            and 'DESCR'
            validator (obj): cross-validator object
            cache_path (str): path to cache directory
            random_state (int/np.random.RandomState/None): initializer of
                                                            the random state
        """
        self.dataset = dataset
        self.db_name = self.dataset['name']
        self.validator = validator
        self.cache_path = cache_path
        self.filename = 'folding_' + self.db_name + '.pickle'
        self.db_size = len(dataset['data'])
        self.db_n_attr = len(dataset['data'][0])
        self.imbalanced_ratio = np.sum(
            self.dataset['target'] == 0)/np.sum(self.dataset['target'] == 1)
        self.random_state = random_state

    def do_folding(self):
        """
        Does the folding or reads it from file if already available

        Returns:
            list(tuple): list of tuples of X_train, y_train, X_test, y_test
                            objects
        """

        self.validator.random_state = self.random_state

        if not hasattr(self, 'folding'):
            cond_cache_none = self.cache_path is None
            if not cond_cache_none:
                filename = os.path.join(self.cache_path, self.filename)
                cond_file_not_exists = not os.path.isfile(filename)
            else:
                cond_file_not_exists = False

            if cond_cache_none or cond_file_not_exists:
                _logger.info(self.__class__.__name__ +
                             (" doing folding %s" % self.filename))

                self.folding = {}
                self.folding['folding'] = []
                self.folding['db_size'] = len(self.dataset['data'])
                self.folding['db_n_attr'] = len(self.dataset['data'][0])
                n_maj = np.sum(self.dataset['target'] == 0)
                n_min = np.sum(self.dataset['target'] == 1)
                self.folding['imbalanced_ratio'] = n_maj / n_min

                X = self.dataset['data']
                y = self.dataset['target']

                data = self.dataset['data']
                target = self.dataset['target']

                for train, test in self.validator.split(data, target, target):
                    folding = (X[train], y[train], X[test], y[test])
                    self.folding['folding'].append(folding)
                if self.cache_path is not None:
                    _logger.info(self.__class__.__name__ +
                                 (" dumping to file %s" % self.filename))
                    random_filename = np.random.randint(1000000)
                    random_filename = str(random_filename) + '.pickle'
                    random_filename = os.path.join(self.cache_path,
                                                   random_filename)
                    pickle.dump(self.folding, open(random_filename, "wb"))
                    os.rename(random_filename, os.path.join(
                        self.cache_path, self.filename))
            else:
                _logger.info(self.__class__.__name__ +
                             (" reading from file %s" % self.filename))
                self.folding = pickle.load(
                    open(os.path.join(self.cache_path, self.filename), "rb"))
        return self.folding

    def get_params(self, deep=False):
        return {'db_name': self.db_name}

    def descriptor(self):
        return str(self.get_params())


class Sampling():
    """
    Cache-able sampling of dataset folds
    """

    def __init__(self,
                 folding,
                 sampler,
                 sampler_parameters,
                 scaler,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            folding (obj): Folding object
            sampler (class): class of a sampler object
            sampler_parameters (dict): a parameter combination for the sampler
                                        object
            scaler (obj): scaler object
            random_state (int/np.random.RandomState/None): initializer of the
                                                            random state
        """
        self.folding = folding
        self.db_name = folding.db_name
        self.sampler = sampler
        self.sampler_parameters = sampler_parameters
        self.sampler_parameters['random_state'] = random_state
        self.scaler = scaler
        self.cache_path = folding.cache_path
        self.filename = self.standardized_filename('sampling')
        self.random_state = random_state

    def standardized_filename(self,
                              prefix,
                              db_name=None,
                              sampler=None,
                              sampler_parameters=None):
        """
        standardizes the filename

        Args:
            filename (str): filename

        Returns:
            str: standardized name
        """
        import hashlib

        db_name = (db_name or self.db_name)

        sampler = (sampler or self.sampler)
        sampler = sampler.__name__
        sampler_parameters = sampler_parameters or self.sampler_parameters
        _logger.info(str(sampler_parameters))
        from collections import OrderedDict
        sampler_parameters_ordered = OrderedDict()
        for k in sorted(list(sampler_parameters.keys())):
            sampler_parameters_ordered[k] = sampler_parameters[k]

        message = " sampler parameter string "
        message = message + str(sampler_parameters_ordered)
        _logger.info(self.__class__.__name__ + message)
        sampler_parameter_str = hashlib.md5(
            str(sampler_parameters_ordered).encode('utf-8')).hexdigest()

        filename = '_'.join(
            [prefix, db_name, sampler, sampler_parameter_str]) + '.pickle'
        filename = re.sub('["\\,:(){}]', '', filename)
        filename = filename.replace("'", '')
        filename = filename.replace(": ", "_")
        filename = filename.replace(" ", "_")
        filename = filename.replace("\n", "_")

        return filename

    def cache_sampling(self):
        try:
            import mkl
            mkl.set_num_threads(1)
            _logger.info(self.__class__.__name__ +
                         (" mkl thread number set to 1 successfully"))
        except Exception as e:
            _logger.info(self.__class__.__name__ +
                         (" setting mkl thread number didn't succeed"))
            _logger.info(str(e))

        if not os.path.isfile(os.path.join(self.cache_path, self.filename)):
            # if the sampled dataset does not exist
            sampler_categories = self.sampler.categories
            is_extensive = OverSampling.cat_extensive in sampler_categories
            has_proportion = 'proportion' in self.sampler_parameters
            higher_prop_sampling_avail = None

            if is_extensive and has_proportion:
                proportion = self.sampler_parameters['proportion']
                all_pc = self.sampler.parameter_combinations()
                all_proportions = np.unique([p['proportion'] for p in all_pc])
                all_proportions = all_proportions[all_proportions > proportion]

                for p in all_proportions:
                    tmp_par = self.sampler_parameters.copy()
                    tmp_par['proportion'] = p
                    tmp_filename = self.standardized_filename(
                        'sampling', self.db_name, self.sampler, tmp_par)

                    filename = os.path.join(self.cache_path, tmp_filename)
                    if os.path.isfile(filename):
                        higher_prop_sampling_avail = (p, tmp_filename)
                        break

            if (not is_extensive or not has_proportion or
                    (is_extensive and has_proportion and
                        higher_prop_sampling_avail is None)):
                _logger.info(self.__class__.__name__ + " doing sampling")
                begin = time.time()
                sampling = []
                folds = self.folding.do_folding()
                for X_train, y_train, X_test, y_test in folds['folding']:
                    s = self.sampler(**self.sampler_parameters)

                    if self.scaler is not None:
                        print(self.scaler.__class__.__name__)
                        X_train = self.scaler.fit_transform(X_train, y_train)
                    X_samp, y_samp = s.sample_with_timing(X_train, y_train)

                    if hasattr(s, 'transform'):
                        X_test_trans = s.preprocessing_transform(X_test)
                    else:
                        X_test_trans = X_test.copy()

                    if self.scaler is not None:
                        X_samp = self.scaler.inverse_transform(X_samp)

                    sampling.append((X_samp, y_samp, X_test_trans, y_test))
                runtime = time.time() - begin
            else:
                higher_prop, higher_prop_filename = higher_prop_sampling_avail
                message = " reading and resampling from file %s to %s"
                message = message % (higher_prop_filename, self.filename)
                _logger.info(self.__class__.__name__ + message)
                filename = os.path.join(self.cache_path, higher_prop_filename)
                tmp_results = pickle.load(open(filename, 'rb'))
                tmp_sampling = tmp_results['sampling']
                tmp_runtime = tmp_results['runtime']

                sampling = []
                folds = self.folding.do_folding()
                nums = [len(X_train) for X_train, _, _, _ in folds['folding']]
                i = 0
                for X_train, y_train, X_test, y_test in tmp_sampling:
                    new_num = (len(X_train) - nums[i])/higher_prop*proportion
                    new_num = int(new_num)
                    offset = nums[i] + new_num
                    X_offset = X_train[:offset]
                    y_offset = y_train[:offset]
                    sampling.append((X_offset, y_offset, X_test, y_test))
                    i = i + 1
                runtime = tmp_runtime/p*proportion

            results = {}
            results['sampling'] = sampling
            results['runtime'] = runtime
            results['db_size'] = folds['db_size']
            results['db_n_attr'] = folds['db_n_attr']
            results['imbalanced_ratio'] = folds['imbalanced_ratio']

            _logger.info(self.__class__.__name__ +
                         (" dumping to file %s" % self.filename))

            random_filename = np.random.randint(1000000)
            random_filename = str(random_filename) + '.pickle'
            random_filename = os.path.join(self.cache_path, random_filename)
            pickle.dump(results, open(random_filename, "wb"))
            os.rename(random_filename, os.path.join(
                self.cache_path, self.filename))

    def do_sampling(self):
        self.cache_sampling()
        results = pickle.load(
            open(os.path.join(self.cache_path, self.filename), 'rb'))
        return results

    def get_params(self, deep=False):
        return {'folding': self.folding.get_params(),
                'sampler_name': self.sampler.__name__,
                'sampler_parameters': self.sampler_parameters}

    def descriptor(self):
        return str(self.get_params())


class Evaluation():
    """
    Cache-able evaluation of classifier on sampling
    """

    def __init__(self,
                 sampling,
                 classifiers,
                 n_threads=None,
                 random_state=None):
        """
        Constructor of an Evaluation object

        Args:
            sampling (obj): Sampling object
            classifiers (list(obj)): classifier objects
            n_threads (int/None): number of threads
            random_state (int/np.random.RandomState/None): random state
                                                            initializer
        """
        self.sampling = sampling
        self.classifiers = classifiers
        self.n_threads = n_threads
        self.cache_path = sampling.cache_path
        self.filename = self.sampling.standardized_filename('eval')
        self.random_state = random_state

        self.labels = []
        for i in range(len(classifiers)):
            from collections import OrderedDict
            sampling_parameters = OrderedDict()
            sp = self.sampling.sampler_parameters
            for k in sorted(list(sp.keys())):
                sampling_parameters[k] = sp[k]
            cp = classifiers[i].get_params()
            classifier_parameters = OrderedDict()
            for k in sorted(list(cp.keys())):
                classifier_parameters[k] = cp[k]

            label = str((self.sampling.db_name, sampling_parameters,
                         classifiers[i].__class__.__name__,
                         classifier_parameters))
            self.labels.append(label)

        print(self.labels)

    def calculate_metrics(self, all_pred, all_test, all_folds):
        """
        Calculates metrics of binary classifiction

        Args:
            all_pred (np.matrix): predicted probabilities
            all_test (np.matrix): true labels

        Returns:
            dict: all metrics of binary classification
        """

        results = {}
        if all_pred is not None:
            all_pred_labels = np.apply_along_axis(
                lambda x: np.argmax(x), 1, all_pred)

            results['tp'] = np.sum(np.logical_and(
                np.equal(all_test, all_pred_labels), (all_test == 1)))
            results['tn'] = np.sum(np.logical_and(
                np.equal(all_test, all_pred_labels), (all_test == 0)))
            results['fp'] = np.sum(np.logical_and(np.logical_not(
                np.equal(all_test, all_pred_labels)), (all_test == 0)))
            results['fn'] = np.sum(np.logical_and(np.logical_not(
                np.equal(all_test, all_pred_labels)), (all_test == 1)))
            results['p'] = results['tp'] + results['fn']
            results['n'] = results['fp'] + results['tn']
            results['acc'] = (results['tp'] + results['tn']) / \
                (results['p'] + results['n'])
            results['sens'] = results['tp']/results['p']
            results['spec'] = results['tn']/results['n']
            results['ppv'] = results['tp']/(results['tp'] + results['fp'])
            results['npv'] = results['tn']/(results['tn'] + results['fn'])
            results['fpr'] = 1.0 - results['spec']
            results['fdr'] = 1.0 - results['ppv']
            results['fnr'] = 1.0 - results['sens']
            results['bacc'] = (results['tp']/results['p'] +
                               results['tn']/results['n'])/2.0
            results['gacc'] = np.sqrt(
                results['tp']/results['p']*results['tn']/results['n'])
            results['f1'] = 2*results['tp'] / \
                (2*results['tp'] + results['fp'] + results['fn'])
            mcc_num = results['tp']*results['tn'] - results['fp']*results['fn']
            mcc_denom_0 = (results['tp'] + results['fp'])
            mcc_denom_1 = (results['tp'] + results['fn'])
            mcc_denom_2 = (results['tn'] + results['fp'])
            mcc_denom_3 = (results['tn'] + results['fn'])
            mcc_denom = mcc_denom_0 * mcc_denom_1 * mcc_denom_2*mcc_denom_3
            results['mcc'] = mcc_num/np.sqrt(mcc_denom)
            results['l'] = (results['p'] + results['n']) * \
                np.log(results['p'] + results['n'])
            tp_fp = (results['tp'] + results['fp'])
            tp_fn = (results['tp'] + results['fn'])
            tn_fp = (results['fp'] + results['tn'])
            tn_fn = (results['fn'] + results['tn'])
            results['ltp'] = results['tp']*np.log(results['tp']/(tp_fp*tp_fn))
            results['lfp'] = results['fp']*np.log(results['fp']/(tp_fp*tn_fp))
            results['lfn'] = results['fn']*np.log(results['fn']/(tp_fn*tn_fn))
            results['ltn'] = results['tn']*np.log(results['tn']/(tn_fp*tn_fn))
            results['lp'] = results['p'] * \
                np.log(results['p']/(results['p'] + results['n']))
            results['ln'] = results['n'] * \
                np.log(results['n']/(results['p'] + results['n']))
            uc_num = (results['l'] + results['ltp'] + results['lfp'] +
                      results['lfn'] + results['ltn'])
            uc_denom = (results['l'] + results['lp'] + results['ln'])
            results['uc'] = uc_num/uc_denom
            results['informedness'] = results['sens'] + results['spec'] - 1.0
            results['markedness'] = results['ppv'] + results['npv'] - 1.0
            results['log_loss'] = log_loss(all_test, all_pred)
            results['auc'] = roc_auc_score(all_test, all_pred[:, 1])
            aucs = [roc_auc_score(all_test[all_folds == i],
                                  all_pred[all_folds == i, 1])
                    for i in range(np.max(all_folds)+1)]
            results['auc_mean'] = np.mean(aucs)
            results['auc_std'] = np.std(aucs)
            test_labels, preds = zip(
                *sorted(zip(all_test, all_pred[:, 1]), key=lambda x: -x[1]))
            test_labels = np.array(test_labels)
            th = int(0.2*len(test_labels))
            results['p_top20'] = np.sum(test_labels[:th] == 1)/th
            results['brier'] = np.mean((all_pred[:, 1] - all_test)**2)
        else:
            results['tp'] = 0
            results['tn'] = 0
            results['fp'] = 0
            results['fn'] = 0
            results['p'] = 0
            results['n'] = 0
            results['acc'] = 0
            results['sens'] = 0
            results['spec'] = 0
            results['ppv'] = 0
            results['npv'] = 0
            results['fpr'] = 1
            results['fdr'] = 1
            results['fnr'] = 1
            results['bacc'] = 0
            results['gacc'] = 0
            results['f1'] = 0
            results['mcc'] = np.nan
            results['l'] = np.nan
            results['ltp'] = np.nan
            results['lfp'] = np.nan
            results['lfn'] = np.nan
            results['ltn'] = np.nan
            results['lp'] = np.nan
            results['ln'] = np.nan
            results['uc'] = np.nan
            results['informedness'] = 0
            results['markedness'] = 0
            results['log_loss'] = np.nan
            results['auc'] = 0
            results['auc_mean'] = 0
            results['auc_std'] = 0
            results['p_top20'] = 0
            results['brier'] = 1

        return results

    def do_evaluation(self):
        """
        Does the evaluation or reads it from file

        Returns:
            dict: all metrics
        """

        if self.n_threads is not None:
            try:
                import mkl
                mkl.set_num_threads(self.n_threads)
                message = " mkl thread number set to %d successfully"
                message = message % self.n_threads
                _logger.info(self.__class__.__name__ + message)
            except Exception as e:
                message = " setting mkl thread number didn't succeed"
                _logger.info(self.__class__.__name__ + message)

        evaluations = {}
        if os.path.isfile(os.path.join(self.cache_path, self.filename)):
            evaluations = pickle.load(
                open(os.path.join(self.cache_path, self.filename), 'rb'))

        already_evaluated = np.array([li in evaluations for li in self.labels])

        if not np.all(already_evaluated):
            samp = self.sampling.do_sampling()
        else:
            return list(evaluations.values())

        # setting random states
        for i in range(len(self.classifiers)):
            clf_params = self.classifiers[i].get_params()
            if 'random_state' in clf_params:
                clf_params['random_state'] = self.random_state
                self.classifiers[i] = self.classifiers[i].__class__(
                    **clf_params)
            if isinstance(self.classifiers[i], CalibratedClassifierCV):
                clf_params = self.classifiers[i].base_estimator.get_params()
                clf_params['random_state'] = self.random_state
                class_inst = self.classifiers[i].base_estimator.__class__
                new_inst = class_inst(**clf_params)
                self.classifiers[i].base_estimator = new_inst

        for i in range(len(self.classifiers)):
            if not already_evaluated[i]:
                message = " do the evaluation %s %s %s"
                message = message % (self.sampling.db_name,
                                     self.sampling.sampler.__name__,
                                     self.classifiers[i].__class__.__name__)
                _logger.info(self.__class__.__name__ + message)
                all_preds, all_tests, all_folds = [], [], []
                minority_class_label = None
                majority_class_label = None
                fold_idx = -1
                for X_train, y_train, X_test, y_test in samp['sampling']:
                    fold_idx += 1

                    # X_train[X_train == np.inf]= 0
                    # X_train[X_train == -np.inf]= 0
                    # X_test[X_test == np.inf]= 0
                    # X_test[X_test == -np.inf]= 0

                    class_labels = np.unique(y_train)
                    min_class_size = np.min(
                        [np.sum(y_train == c) for c in class_labels])

                    ss = StandardScaler()
                    X_train_trans = ss.fit_transform(X_train)
                    nonzero_var_idx = np.where(ss.var_ > 1e-8)[0]
                    X_test_trans = ss.transform(X_test)

                    enough_minority_samples = min_class_size > 4
                    y_train_big_enough = len(y_train) > 4
                    two_classes = len(class_labels) > 1
                    at_least_one_feature = (len(nonzero_var_idx) > 0)

                    if not enough_minority_samples:
                        message = " not enough minority samples: %d"
                        message = message % min_class_size
                        _logger.warning(
                            self.__class__.__name__ + message)
                    elif not y_train_big_enough:
                        message = (" number of minority training samples is "
                                   "not enough: %d")
                        message = message % len(y_train)
                        _logger.warning(self.__class__.__name__ + message)
                    elif not two_classes:
                        message = " there is only 1 class in training data"
                        _logger.warning(self.__class__.__name__ + message)
                    elif not at_least_one_feature:
                        _logger.warning(self.__class__.__name__ +
                                        (" no information in features"))
                    else:
                        all_tests.append(y_test)
                        if (minority_class_label is None or
                                majority_class_label is None):
                            class_labels = np.unique(y_train)
                            n_0 = sum(class_labels[0] == y_test)
                            n_1 = sum(class_labels[1] == y_test)
                            if n_0 < n_1:
                                minority_class_label = int(class_labels[0])
                                majority_class_label = int(class_labels[1])
                            else:
                                minority_class_label = int(class_labels[1])
                                majority_class_label = int(class_labels[0])

                        X_fit = X_train_trans[:, nonzero_var_idx]
                        self.classifiers[i].fit(X_fit, y_train)
                        clf = self.classifiers[i]
                        X_pred = X_test_trans[:, nonzero_var_idx]
                        pred = clf.predict_proba(X_pred)
                        all_preds.append(pred)
                        all_folds.append(
                            np.repeat(fold_idx, len(all_preds[-1])))

                if len(all_tests) > 0:
                    all_preds = np.vstack(all_preds)
                    all_tests = np.hstack(all_tests)
                    all_folds = np.hstack(all_folds)

                    evaluations[self.labels[i]] = self.calculate_metrics(
                        all_preds, all_tests, all_folds)
                else:
                    evaluations[self.labels[i]] = self.calculate_metrics(
                        None, None, None)

                evaluations[self.labels[i]]['runtime'] = samp['runtime']
                sampler_name = self.sampling.sampler.__name__
                evaluations[self.labels[i]]['sampler'] = sampler_name
                clf_name = self.classifiers[i].__class__.__name__
                evaluations[self.labels[i]]['classifier'] = clf_name
                sampler_parameters = self.sampling.sampler_parameters.copy()

                evaluations[self.labels[i]]['sampler_parameters'] = str(
                    sampler_parameters)
                evaluations[self.labels[i]]['classifier_parameters'] = str(
                    self.classifiers[i].get_params())
                evaluations[self.labels[i]]['sampler_categories'] = str(
                    self.sampling.sampler.categories)
                evaluations[self.labels[i]
                            ]['db_name'] = self.sampling.folding.db_name
                evaluations[self.labels[i]]['db_size'] = samp['db_size']
                evaluations[self.labels[i]]['db_n_attr'] = samp['db_n_attr']
                evaluations[self.labels[i]
                            ]['imbalanced_ratio'] = samp['imbalanced_ratio']

        if not np.all(already_evaluated):
            _logger.info(self.__class__.__name__ +
                         (" dumping to file %s" % self.filename))
            random_filename = os.path.join(self.cache_path, str(
                np.random.randint(1000000)) + '.pickle')
            pickle.dump(evaluations, open(random_filename, "wb"))
            os.rename(random_filename, os.path.join(
                self.cache_path, self.filename))

        return list(evaluations.values())


def trans(X):
    """
    Transformation function used to aggregate the evaluation results.

    Args:
        X (pd.DataFrame): a grouping of a data frame containing evaluation
                            results
    """
    auc_std = X.iloc[np.argmax(X['auc_mean'].values)]['auc_std']
    cp_auc = X.sort_values('auc')['classifier_parameters'].iloc[-1]
    cp_acc = X.sort_values('acc')['classifier_parameters'].iloc[-1]
    cp_gacc = X.sort_values('gacc')['classifier_parameters'].iloc[-1]
    cp_f1 = X.sort_values('f1')['classifier_parameters'].iloc[-1]
    cp_p_top20 = X.sort_values('p_top20')['classifier_parameters'].iloc[-1]
    cp_brier = X.sort_values('brier')['classifier_parameters'].iloc[-1]
    sp_auc = X.sort_values('auc')['sampler_parameters'].iloc[-1]
    sp_acc = X.sort_values('acc')['sampler_parameters'].iloc[-1]
    sp_gacc = X.sort_values('gacc')['sampler_parameters'].iloc[-1]
    sp_f1 = X.sort_values('f1')['sampler_parameters'].iloc[-1]
    sp_p_top20 = X.sort_values('p_top20')['sampler_parameters'].iloc[-1]
    sp_brier = X.sort_values('p_top20')['sampler_parameters'].iloc[0]

    return pd.DataFrame({'auc': np.max(X['auc']),
                         'auc_mean': np.max(X['auc_mean']),
                         'auc_std': auc_std,
                         'brier': np.min(X['brier']),
                         'acc': np.max(X['acc']),
                         'f1': np.max(X['f1']),
                         'p_top20': np.max(X['p_top20']),
                         'gacc': np.max(X['gacc']),
                         'runtime': np.mean(X['runtime']),
                         'db_size': X['db_size'].iloc[0],
                         'db_n_attr': X['db_n_attr'].iloc[0],
                         'imbalanced_ratio': X['imbalanced_ratio'].iloc[0],
                         'sampler_categories': X['sampler_categories'].iloc[0],
                         'classifier_parameters_auc': cp_auc,
                         'classifier_parameters_acc': cp_acc,
                         'classifier_parameters_gacc': cp_gacc,
                         'classifier_parameters_f1': cp_f1,
                         'classifier_parameters_p_top20': cp_p_top20,
                         'classifier_parameters_brier': cp_brier,
                         'sampler_parameters_auc': sp_auc,
                         'sampler_parameters_acc': sp_acc,
                         'sampler_parameters_gacc': sp_gacc,
                         'sampler_parameters_f1': sp_f1,
                         'sampler_parameters_p_top20': sp_p_top20,
                         'sampler_parameters_brier': sp_brier,
                         }, index=[0])


def _clone_classifiers(classifiers):
    """
    Clones a set of classifiers

    Args:
        classifiers (list): a list of classifier objects
    """
    results = []
    for c in classifiers:
        if isinstance(c, MLPClassifierWrapper):
            results.append(c.copy())
        else:
            results.append(clone(c))

    return results


def _cache_samplings(folding,
                     samplers,
                     scaler,
                     max_n_sampler_par_comb=35,
                     n_jobs=1,
                     random_state=None):
    """

    """
    _logger.info("create sampling objects, random_state: %s" %
                 str(random_state or ""))
    sampling_objs = []

    random_state_init = random_state
    random_state = np.random.RandomState(random_state_init)

    _logger.info("samplers: %s" % str(samplers))
    for s in samplers:
        sampling_par_comb = s.parameter_combinations()
        _logger.info(sampling_par_comb)
        domain = np.array(list(range(len(sampling_par_comb))))
        n_random = min([len(sampling_par_comb), max_n_sampler_par_comb])
        random_indices = random_state.choice(domain, n_random, replace=False)
        _logger.info("random_indices: %s" % random_indices)
        sampling_par_comb = [sampling_par_comb[i] for i in random_indices]
        _logger.info(sampling_par_comb)

        for spc in sampling_par_comb:
            sampling_objs.append(Sampling(folding,
                                          s,
                                          spc,
                                          scaler,
                                          random_state_init))

    # sorting sampling objects to optimize execution
    def key(x):
        if (isinstance(x.sampler, ADG) or isinstance(x.sampler, AMSCO) or
                isinstance(x.sampler, DSRBF)):
            if 'proportion' in x.sampler_parameters:
                return 30 + x.sampler_parameters['proportion']
            else:
                return 30
        elif 'proportion' in x.sampler_parameters:
            return x.sampler_parameters['proportion']
        elif OverSampling.cat_memetic in x.sampler.categories:
            return 20
        else:
            return 10

    sampling_objs = list(reversed(sorted(sampling_objs, key=key)))

    # executing sampling in parallel
    _logger.info("executing %d sampling in parallel" % len(sampling_objs))
    Parallel(n_jobs=n_jobs, batch_size=1)(delayed(s.cache_sampling)()
                                          for s in sampling_objs)

    return sampling_objs


def _cache_evaluations(sampling_objs,
                       classifiers,
                       n_jobs=1,
                       random_state=None):
    # create evaluation objects
    _logger.info("create classifier jobs")
    evaluation_objs = []

    num_threads = None if n_jobs is None or n_jobs == 1 else 1

    for s in sampling_objs:
        evaluation_objs.append(Evaluation(s, _clone_classifiers(
            classifiers), num_threads, random_state))

    _logger.info("executing %d evaluation jobs in parallel" %
                 (len(evaluation_objs)))
    # execute evaluation in parallel
    evals = Parallel(n_jobs=n_jobs, batch_size=1)(
        delayed(e.do_evaluation)() for e in evaluation_objs)

    return evals


def _read_db_results(cache_path_db):
    results = []
    evaluation_files = glob.glob(os.path.join(cache_path_db, 'eval*.pickle'))

    for f in evaluation_files:
        eval_results = pickle.load(open(f, 'rb'))
        results.append(list(eval_results.values()))

    return results


def read_oversampling_results(datasets, cache_path=None, all_results=False):
    """
    Reads the results of the evaluation

    Args:
        datasets (list): list of datasets and/or dataset loaders - a dataset
                            is a dict with 'data', 'target' and 'name' keys
        cache_path (str): path to a cache directory
        all_results (bool): True to return all results, False to return an
                                aggregation

    Returns:
        pd.DataFrame: all results or the aggregated results if all_results is
                        False
    """

    results = []
    for dataset_spec in datasets:

        # loading dataset if needed and determining dataset name
        if not isinstance(dataset_spec, dict):
            dataset = dataset_spec()
        else:
            dataset = dataset_spec

        if 'name' in dataset:
            dataset_name = dataset['name']
        else:
            dataset_name = dataset_spec.__name__

        dataset['name'] = dataset_name

        # determining dataset specific cache path
        cache_path_db = os.path.join(cache_path, dataset_name)

        # reading the results
        res = _read_db_results(cache_path_db)

        # concatenating the results
        _logger.info("concatenating results")
        db_res = [pd.DataFrame(r) for r in res]
        db_res = pd.concat(db_res).reset_index(drop=True)

        _logger.info("aggregating the results")
        if all_results is False:
            db_res = db_res.groupby(by=['db_name', 'classifier', 'sampler'])
            db_res.apply(trans).reset_index().drop('level_3', axis=1)

        results.append(db_res)

    return pd.concat(results).reset_index(drop=True)


def evaluate_oversamplers(datasets,
                          samplers,
                          classifiers,
                          cache_path,
                          validator=RepeatedStratifiedKFold(
                              n_splits=5, n_repeats=3),
                          scaler=None,
                          all_results=False,
                          remove_cache=False,
                          max_samp_par_comb=35,
                          n_jobs=1,
                          random_state=None):
    """
    Evaluates oversampling techniques using various classifiers on various
        datasets

    Args:
        datasets (list): list of datasets and/or dataset loaders - a dataset
                            is a dict with 'data', 'target' and 'name' keys
        samplers (list): list of oversampling classes/objects
        classifiers (list): list of classifier objects
        cache_path (str): path to a cache directory
        validator (obj): validator object
        scaler (obj): scaler object
        all_results (bool): True to return all results, False to return an
                                aggregation
        remove_cache (bool): True to remove sampling objects after
                                        evaluation
        max_samp_par_comb (int): maximum number of sampler parameter
                                    combinations to be tested
        n_jobs (int): number of parallel jobs
        random_state (int/np.random.RandomState/None): initializer of the
                                                        random state

    Returns:
        pd.DataFrame: all results or the aggregated results if all_results is
                        False

    Example::

        import smote_variants as sv
        import imbalanced_datasets as imbd

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier

        datasets= [imbd.load_glass2, imbd.load_ecoli4]
        oversamplers= [sv.SMOTE_ENN, sv.NEATER, sv.Lee]
        classifiers= [KNeighborsClassifier(n_neighbors= 3),
                      KNeighborsClassifier(n_neighbors= 5),
                      DecisionTreeClassifier()]

        cache_path= '/home/<user>/smote_validation/'

        results= evaluate_oversamplers(datasets,
                                       oversamplers,
                                       classifiers,
                                       cache_path)
    """

    if cache_path is None:
        raise ValueError('cache_path is not specified')

    results = []
    for dataset_spec in datasets:
        # loading dataset if needed and determining dataset name
        if not isinstance(dataset_spec, dict):
            dataset = dataset_spec()
        else:
            dataset = dataset_spec

        if 'name' in dataset:
            dataset_name = dataset['name']
        else:
            dataset_name = dataset_spec.__name__

        dataset['name'] = dataset_name

        dataset_original_target = dataset['target'].copy()
        class_labels = np.unique(dataset['target'])
        n_0 = sum(dataset['target'] == class_labels[0])
        n_1 = sum(dataset['target'] == class_labels[1])
        if n_0 < n_1:
            min_label = class_labels[0]
            maj_label = class_labels[1]
        else:
            min_label = class_labels[1]
            maj_label = class_labels[0]
        min_ind = np.where(dataset['target'] == min_label)[0]
        maj_ind = np.where(dataset['target'] == maj_label)[0]
        np.put(dataset['target'], min_ind, 1)
        np.put(dataset['target'], maj_ind, 0)

        cache_path_db = os.path.join(cache_path, dataset_name)
        if not os.path.isdir(cache_path_db):
            _logger.info("creating cache directory")
            os.makedirs(cache_path_db)

        # checking of samplings and evaluations are available
        samplings_available = False
        evaluations_available = False

        samplings = glob.glob(os.path.join(cache_path_db, 'sampling*.pickle'))
        if len(samplings) > 0:
            samplings_available = True

        evaluations = glob.glob(os.path.join(cache_path_db, 'eval*.pickle'))
        if len(evaluations) > 0:
            evaluations_available = True

        message = ("dataset: %s, samplings_available: %s, "
                   "evaluations_available: %s")
        message = message % (dataset_name, str(samplings_available),
                             str(evaluations_available))
        _logger.info(message)

        if (remove_cache and evaluations_available and
                not samplings_available):
            # remove_cache is enabled and evaluations are available,
            # they are being read
            message = ("reading result from cache, sampling and evaluation is"
                       " not executed")
            _logger.info(message)
            res = _read_db_results(cache_path_db)
        else:
            _logger.info("doing the folding")
            folding = Folding(dataset, validator, cache_path_db, random_state)
            folding.do_folding()

            _logger.info("do the samplings")
            sampling_objs = _cache_samplings(folding,
                                             samplers,
                                             scaler,
                                             max_samp_par_comb,
                                             n_jobs,
                                             random_state)

            _logger.info("do the evaluations")
            res = _cache_evaluations(
                sampling_objs, classifiers, n_jobs, random_state)

        dataset['target'] = dataset_original_target

        # removing samplings once everything is done
        if remove_cache:
            filenames = glob.glob(os.path.join(cache_path_db, 'sampling*'))
            _logger.info("removing unnecessary sampling files")
            if len(filenames) > 0:
                for f in filenames:
                    os.remove(f)

        _logger.info("concatenating the results")
        db_res = [pd.DataFrame(r) for r in res]
        db_res = pd.concat(db_res).reset_index(drop=True)

        random_filename = os.path.join(cache_path_db, str(
            np.random.randint(1000000)) + '.pickle')
        pickle.dump(db_res, open(random_filename, "wb"))
        os.rename(random_filename, os.path.join(
            cache_path_db, 'results.pickle'))

        _logger.info("aggregating the results")
        if all_results is False:
            db_res = db_res.groupby(by=['db_name', 'classifier', 'sampler'])
            db_res = db_res.apply(trans).reset_index().drop('level_3', axis=1)

        results.append(db_res)

    return pd.concat(results).reset_index(drop=True)


def model_selection(dataset,
                    samplers,
                    classifiers,
                    cache_path,
                    score='auc',
                    validator=RepeatedStratifiedKFold(n_splits=5, n_repeats=3),
                    remove_cache=False,
                    max_samp_par_comb=35,
                    n_jobs=1,
                    random_state=None):
    """
    Evaluates oversampling techniques on various classifiers and a dataset
    and returns the oversampling and classifier objects giving the best
    performance

    Args:
        dataset (dict): a dataset is a dict with 'data', 'target' and 'name'
                        keys
        samplers (list): list of oversampling classes/objects
        classifiers (list): list of classifier objects
        cache_path (str): path to a cache directory
        score (str): 'auc'/'acc'/'gacc'/'f1'/'brier'/'p_top20'
        validator (obj): validator object
        all_results (bool): True to return all results, False to return an
                            aggregation
        remove_cache (bool): True to remove sampling objects after
                                        evaluation
        max_samp_par_comb (int): maximum number of sampler parameter
                                    combinations to be tested
        n_jobs (int): number of parallel jobs
        random_state (int/np.random.RandomState/None): initializer of the
                                                        random state

    Returns:
        obj, obj: the best performing sampler object and the best performing
                    classifier object

    Example::

        import smote_variants as sv
        import imbalanced_datasets as imbd

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier

        datasets= imbd.load_glass2()
        oversamplers= [sv.SMOTE_ENN, sv.NEATER, sv.Lee]
        classifiers= [KNeighborsClassifier(n_neighbors= 3),
                      KNeighborsClassifier(n_neighbors= 5),
                      DecisionTreeClassifier()]

        cache_path= '/home/<user>/smote_validation/'

        sampler, classifier= model_selection(dataset,
                                             oversamplers,
                                             classifiers,
                                             cache_path,
                                             'auc')
    """

    if score not in ['auc', 'acc', 'gacc', 'f1', 'brier', 'p_top20']:
        raise ValueError("score %s not supported" % score)

    results = evaluate_oversamplers(datasets=[dataset],
                                    samplers=samplers,
                                    classifiers=classifiers,
                                    cache_path=cache_path,
                                    validator=validator,
                                    remove_cache=remove_cache,
                                    max_samp_par_comb=max_samp_par_comb,
                                    n_jobs=n_jobs,
                                    random_state=random_state)

    # extracting the best performing classifier and oversampler parameters
    # regarding AUC
    highest_score = results[score].idxmax()
    cl_par_name = 'classifier_parameters_' + score
    samp_par_name = 'sampler_parameters_' + score
    cl, cl_par, samp, samp_par = results.loc[highest_score][['classifier',
                                                             cl_par_name,
                                                             'sampler',
                                                             samp_par_name]]

    # instantiating the best performing oversampler and classifier objects
    samp_obj = eval(samp)(**eval(samp_par))
    cl_obj = eval(cl)(**eval(cl_par))

    return samp_obj, cl_obj


def cross_validate(dataset,
                   sampler,
                   classifier,
                   validator=RepeatedStratifiedKFold(n_splits=5, n_repeats=3),
                   scaler=StandardScaler(),
                   random_state=None):
    """
    Evaluates oversampling techniques on various classifiers and a dataset
    and returns the oversampling and classifier objects giving the best
    performance

    Args:
        dataset (dict): a dataset is a dict with 'data', 'target' and 'name'
                        keys
        samplers (list): list of oversampling classes/objects
        classifiers (list): list of classifier objects
        validator (obj): validator object
        scaler (obj): scaler object
        random_state (int/np.random.RandomState/None): initializer of the
                                                        random state

    Returns:
        pd.DataFrame: the cross-validation scores

    Example::

        import smote_variants as sv
        import imbalanced_datasets as imbd

        from sklearn.neighbors import KNeighborsClassifier

        dataset= imbd.load_glass2()
        sampler= sv.SMOTE_ENN
        classifier= KNeighborsClassifier(n_neighbors= 3)

        sampler, classifier= model_selection(dataset,
                                             oversampler,
                                             classifier)
    """

    class_labels = np.unique(dataset['target'])
    binary_problem = (len(class_labels) == 2)

    dataset_orig_target = dataset['target'].copy()
    if binary_problem:
        _logger.info("The problem is binary")
        n_0 = sum(dataset['target'] == class_labels[0])
        n_1 = sum(dataset['target'] == class_labels[1])
        if n_0 < n_1:
            min_label = class_labels[0]
            maj_label = class_labels[1]
        else:
            min_label = class_labels[0]
            maj_label = class_labels[1]

        min_ind = np.where(dataset['target'] == min_label)[0]
        maj_ind = np.where(dataset['target'] == maj_label)[0]
        np.put(dataset['target'], min_ind, 1)
        np.put(dataset['target'], maj_ind, 0)
    else:
        _logger.info("The problem is not binary")
        label_indices = {}
        for c in class_labels:
            label_indices[c] = np.where(dataset['target'] == c)[0]
        mapping = {}
        for i, c in enumerate(class_labels):
            np.put(dataset['target'], label_indices[c], i)
            mapping[i] = c

    runtimes = []
    all_preds, all_tests = [], []

    for train, test in validator.split(dataset['data'], dataset['target']):
        _logger.info("Executing fold")
        X_train, y_train = dataset['data'][train], dataset['target'][train]
        X_test, y_test = dataset['data'][test], dataset['target'][test]

        begin = time.time()
        X_samp, y_samp = sampler.sample(X_train, y_train)
        runtimes.append(time.time() - begin)

        X_samp_trans = scaler.fit_transform(X_samp)
        nonzero_var_idx = np.where(scaler.var_ > 1e-8)[0]
        X_test_trans = scaler.transform(X_test)

        all_tests.append(y_test)

        classifier.fit(X_samp_trans[:, nonzero_var_idx], y_samp)
        all_preds.append(classifier.predict_proba(
            X_test_trans[:, nonzero_var_idx]))

    if len(all_tests) > 0:
        all_preds = np.vstack(all_preds)
        all_tests = np.hstack(all_tests)

    dataset['target'] = dataset_orig_target

    _logger.info("Computing the results")

    results = {}
    results['runtime'] = np.mean(runtimes)
    results['sampler'] = sampler.__class__.__name__
    results['classifier'] = classifier.__class__.__name__
    results['sampler_parameters'] = str(sampler.get_params())
    results['classifier_parameters'] = str(classifier.get_params())
    results['db_size'] = len(dataset['data'])
    results['db_n_attr'] = len(dataset['data'][0])
    results['db_n_classes'] = len(class_labels)

    if binary_problem:
        results['imbalance_ratio'] = sum(
            dataset['target'] == maj_label)/sum(dataset['target'] == min_label)
        all_pred_labels = np.apply_along_axis(
            lambda x: np.argmax(x), 1, all_preds)

        results['tp'] = np.sum(np.logical_and(
            np.equal(all_tests, all_pred_labels), (all_tests == 1)))
        results['tn'] = np.sum(np.logical_and(
            np.equal(all_tests, all_pred_labels), (all_tests == 0)))
        results['fp'] = np.sum(np.logical_and(np.logical_not(
            np.equal(all_tests, all_pred_labels)), (all_tests == 0)))
        results['fn'] = np.sum(np.logical_and(np.logical_not(
            np.equal(all_tests, all_pred_labels)), (all_tests == 1)))
        results['p'] = results['tp'] + results['fn']
        results['n'] = results['fp'] + results['tn']
        results['acc'] = (results['tp'] + results['tn']) / \
            (results['p'] + results['n'])
        results['sens'] = results['tp']/results['p']
        results['spec'] = results['tn']/results['n']
        results['ppv'] = results['tp']/(results['tp'] + results['fp'])
        results['npv'] = results['tn']/(results['tn'] + results['fn'])
        results['fpr'] = 1.0 - results['spec']
        results['fdr'] = 1.0 - results['ppv']
        results['fnr'] = 1.0 - results['sens']
        results['bacc'] = (results['tp']/results['p'] +
                           results['tn']/results['n'])/2.0
        results['gacc'] = np.sqrt(
            results['tp']/results['p']*results['tn']/results['n'])
        results['f1'] = 2*results['tp'] / \
            (2*results['tp'] + results['fp'] + results['fn'])
        mcc_num = (results['tp']*results['tn'] - results['fp']*results['fn'])
        tp_fp = (results['tp'] + results['fp'])
        tp_fn = (results['tp'] + results['fn'])
        tn_fp = (results['tn'] + results['fp'])
        tn_fn = (results['tn'] + results['fn'])
        mcc_denom = np.sqrt(tp_fp * tp_fn * tn_fp * tn_fn)
        results['mcc'] = mcc_num/mcc_denom
        results['l'] = (results['p'] + results['n']) * \
            np.log(results['p'] + results['n'])
        results['ltp'] = results['tp']*np.log(results['tp']/(
            (results['tp'] + results['fp'])*(results['tp'] + results['fn'])))
        results['lfp'] = results['fp']*np.log(results['fp']/(
            (results['fp'] + results['tp'])*(results['fp'] + results['tn'])))
        results['lfn'] = results['fn']*np.log(results['fn']/(
            (results['fn'] + results['tp'])*(results['fn'] + results['tn'])))
        results['ltn'] = results['tn']*np.log(results['tn']/(
            (results['tn'] + results['fp'])*(results['tn'] + results['fn'])))
        results['lp'] = results['p'] * \
            np.log(results['p']/(results['p'] + results['n']))
        results['ln'] = results['n'] * \
            np.log(results['n']/(results['p'] + results['n']))
        ucc_num = (results['l'] + results['ltp'] + results['lfp'] +
                   results['lfn'] + results['ltn'])
        results['uc'] = ucc_num/(results['l'] + results['lp'] + results['ln'])
        results['informedness'] = results['sens'] + results['spec'] - 1.0
        results['markedness'] = results['ppv'] + results['npv'] - 1.0
        results['log_loss'] = log_loss(all_tests, all_preds)
        results['auc'] = roc_auc_score(all_tests, all_preds[:, 1])
        test_labels, preds = zip(
            *sorted(zip(all_tests, all_preds[:, 1]), key=lambda x: -x[1]))
        test_labels = np.array(test_labels)
        th = int(0.2*len(test_labels))
        results['p_top20'] = np.sum(test_labels[:th] == 1)/th
        results['brier'] = np.mean((all_preds[:, 1] - all_tests)**2)
    else:
        all_pred_labels = np.apply_along_axis(
            lambda x: np.argmax(x), 1, all_preds)

        results['acc'] = accuracy_score(all_tests, all_pred_labels)
        results['confusion_matrix'] = confusion_matrix(
            all_tests, all_pred_labels)
        sum_confusion = np.sum(results['confusion_matrix'], axis=0)
        results['gacc'] = gmean(np.diagonal(
            results['confusion_matrix'])/sum_confusion)
        results['class_label_mapping'] = mapping

    return pd.DataFrame({'value': list(results.values())},
                        index=results.keys())

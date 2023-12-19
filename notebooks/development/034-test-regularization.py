# %%
import datetime

from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import pickle

import tqdm

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from smote_variants.oversampling import SMOTE, NoSMOTE, ADASYN, Borderline_SMOTE1, ProWSyn, SMOTE_IPF, Lee, SMOBD
from common_datasets.binary_classification import get_filtered_data_loaders
import common_datasets.binary_classification as binclas

# %%
import logging
logger = logging.getLogger('smote_variants')
logger.setLevel(logging.ERROR)

# %%
classifiers = {
DecisionTreeClassifier: [{'max_depth': md, 'random_state': 5} for md in [1, 2] + list(range(3, 18, 2))],
RandomForestClassifier: [{'max_depth': md, 'random_state': 5, 'n_jobs': 1} for md in [1, 2] + list(range(3, 18, 2))],
KNeighborsClassifier: [{'n_neighbors': nn, 'n_jobs': 1} for nn in range(1, 70, 4)],
SVC: [{'C': c, 'probability': True, 'random_state': 5} for c in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]]\
            + [{'C': c, 'probability': True, 'kernel': 'poly', 'degree': 2, 'random_state': 5} for c in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]]\
            + [{'C': c, 'probability': True, 'kernel': 'poly', 'degree': 3, 'random_state': 5} for c in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]]\
}

# %%
datasets = get_filtered_data_loaders(n_col_bounds=(2, 50),
                                        n_bounds=(10, 700),
                                        n_minority_bounds=(10, 500),
                                        n_from_phenotypes=1,
                                        n_smallest=40)

# %%
datasets = [loader for loader in datasets if loader not in [binclas.load_iris0, binclas.load_dermatology_6, binclas.load_shuttle_6_vs_2_3, binclas.load_monk_2, binclas.load_new_thyroid1]]

# %%
len(datasets)

# %%
oversampler_classes = [SMOTE, Borderline_SMOTE1, ADASYN, ProWSyn, SMOTE_IPF, Lee, SMOBD, NoSMOTE]

# %%
oversamplers = {}
for oversampler in oversampler_classes:
    random_state = np.random.RandomState(5)
    params = oversampler.parameter_combinations()
    params = [comb for comb in params if comb.get('proportion', 1.0) == 1.0]
    n_params = min(10, len(params))
    oversamplers[oversampler] = random_state.choice(params, n_params, replace=False)

# %%
def job_generator(data_loader):

    dataset = data_loader()

    X = dataset['data']
    y = dataset['target']

    validator = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=5)

    for fidx, (train, test) in enumerate(validator.split(X, y, y)):
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]

        ss = StandardScaler()
        ss.fit(X_train)
        X_train = ss.transform(X_train)
        X_test = ss.transform(X_test)

        for oversampler, oparam in oversamplers.items():
            for sparam in oparam:
                oversampling = oversampler(**sparam)
                X_samp, y_samp = oversampling.sample(X_train, y_train)

                job = {
                    'X_samp': X_samp,
                    'y_samp': y_samp,
                    'X_test': X_test,
                    'y_test': y_test,
                }

                description = {
                    'name': dataset['name'],
                    'fold': fidx,
                    'oversampler': oversampler.__name__,
                    'sparam': sparam,
                }

                yield job, description

# %%
def do_job(job, description):
    results = []
    for classifier, cparams in classifiers.items():
        for cparam in cparams:
            tmp = description.copy()
            classifier_obj = classifier(**cparam)
            classifier_obj.fit(job['X_samp'], job['y_samp'])
            y_pred = classifier_obj.predict_proba(job['X_test'])
            auc = roc_auc_score(job['y_test'], y_pred[:, 1])

            tmp['classifier'] = classifier.__name__
            tmp['cparam'] = cparam
            tmp['auc'] = auc
            results.append(tmp)

    return results

# %%
for data_loader in datasets:
    if data_loader != binclas.load_appendicitis:
        continue
    dataset = data_loader()

    print(datetime.datetime.now(), dataset['name'])

    results = Parallel(n_jobs=3)(delayed(do_job)(*x) for x in tqdm.tqdm(job_generator(data_loader)))

    with open(f'{dataset["name"]}-reg.pickle', 'wb') as file:
        pickle.dump(results, file)


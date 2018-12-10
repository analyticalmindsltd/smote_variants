# import classifiers
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from smote_variants import MLPClassifierWrapper

# import SMOTE variants
import smote_variants as sv

# imbalanced databases
import imbalanced_databases as imbd

# to derive parameter combinations
import itertools

# global variables
cache_path= '/home/gykovacs/workspaces/smote_results/'
max_sampler_parameter_combinations= 35
n_jobs= 5

# instantiate classifiers
sv_classifiers= [CalibratedClassifierCV(LinearSVC(C=1.0, penalty='l1', loss= 'squared_hinge', dual= False)),
                 CalibratedClassifierCV(LinearSVC(C=1.0, penalty='l2', loss= 'hinge', dual= True)),
                 CalibratedClassifierCV(LinearSVC(C=1.0, penalty='l2', loss= 'squared_hinge', dual= False)),
                 CalibratedClassifierCV(LinearSVC(C=10.0, penalty='l1', loss= 'squared_hinge', dual= False)),
                 CalibratedClassifierCV(LinearSVC(C=10.0, penalty='l2', loss= 'hinge', dual= True)),
                 CalibratedClassifierCV(LinearSVC(C=10.0, penalty='l2', loss= 'squared_hinge', dual= False))]

mlp_classifiers= []
for x in itertools.product(['relu', 'logistic'], [1.0, 0.5, 0.1]):
#for x in itertools.product(['relu'], [1.0, 0.1]):
    mlp_classifiers.append(MLPClassifierWrapper(activation= x[0], hidden_layer_fraction= x[1]))

nn_classifiers= []
for x in itertools.product([3, 5, 7], ['uniform', 'distance'], [1, 2, 3]):
#for x in itertools.product([3, 7], ['uniform', 'distance'], [2]):
    nn_classifiers.append(KNeighborsClassifier(n_neighbors= x[0], weights= x[1], p= x[2]))

dt_classifiers= []
for x in itertools.product(['gini', 'entropy'], [None, 3, 5]):
#for x in itertools.product(['gini', 'entropy'], [None, 5]):
    dt_classifiers.append(DecisionTreeClassifier(criterion= x[0], max_depth= x[1]))

classifiers= []
classifiers.extend(sv_classifiers)
classifiers.extend(mlp_classifiers)
classifiers.extend(nn_classifiers)
classifiers.extend(dt_classifiers)

datasets= imbd.get_filtered_data_loaders(len_upper_bound= 5000,
                                         len_lower_bound= 1,
                                         num_features_upper_bound= 100,
                                         num_features_lower_bound= 0)

print(len(datasets))

#datasets= [imbd.load_pc1, imbd.load_kc1, imbd.load_hypothyroid, imbd.load_abalone_20_vs_8_9_10]
#           imbd.load_abalone_17_vs_7_8_9_10, imbd.load_abalone_19_vs_10_11_12_13]

# instantiate the validation object
cv= sv.CacheAndValidate(samplers= sv.get_all_oversamplers(),
                       classifiers= classifiers,
                       datasets= datasets,
                       cache_path= cache_path,
                       n_jobs= 6,
                       max_n_sampler_par_comb= 35)

#cv= sv.CacheAndValidate(samplers= [sv.RWO_sampling,
#                                   sv.cluster_SMOTE,
#                                   sv.NoSMOTE],
#                       classifiers= classifiers,
#                       datasets= datasets,
#                       cache_path= cache_path,
#                       n_jobs= 6,
#                       max_n_sampler_par_comb= 35)


#cv= sv.CacheAndValidate(samplers= sv.get_all_oversamplers(),
#                       classifiers= classifiers,
#                       datasets= [imbd.load_ecoli4],
#                       cache_path= cache_path,
#                       n_jobs= 1,
#                       max_n_sampler_par_comb= 35)

#cv= sv.CacheAndValidate(samplers= sv.get_all_oversamplers(),
#                       classifiers= classifiers,
#                       datasets= [imbd.load_glass, imbd.load_iris0],
#                       cache_path= cache_path,
#                       n_jobs= 5,
#                       max_n_sampler_par_comb= 35)

# execute the validation
results= cv.cache_and_evaluate()

import pickle
import os
pickle.dump(results, open(os.path.join(cache_path, 'results.pickle'), 'wb'))

import numpy as np

results[results['classifier'] == 'CalibratedClassifierCV'].groupby(by=['sampler']).agg({'auc': np.mean}).sort_values('auc')
results[results['classifier'] == 'DecisionTreeClassifier'].groupby(by=['sampler']).agg({'auc': np.mean}).sort_values('auc')
results[results['classifier'] == 'KNeighborsClassifier'].groupby(by=['sampler']).agg({'auc': np.mean}).sort_values('auc')
results[results['classifier'] == 'MLPClassifierWrapper'].groupby(by=['sampler']).agg({'auc': np.mean}).sort_values('auc')

results[results['classifier'] == 'CalibratedClassifierCV'].groupby(by=['sampler']).agg({'auc': np.mean}).sort_values('auc')
results[results['classifier'] == 'DecisionTreeClassifier'].groupby(by=['sampler']).agg({'auc': np.mean}).sort_values('auc')
results[results['classifier'] == 'KNeighborsClassifier'].groupby(by=['sampler']).agg({'auc': np.mean}).sort_values('auc')
results[results['classifier'] == 'MLPClassifierWrapper'].groupby(by=['sampler']).agg({'auc': np.mean}).sort_values('auc')

results.groupby(by=['sampler']).agg({'auc': np.mean}).sort_values('auc')

results.groupby(by=['sampler']).agg({'brier': np.mean}).sort_values('brier')

results.groupby(by=['sampler']).agg({'gacc': np.mean}).sort_values('gacc')

results.groupby(by=['sampler']).agg({'f1': np.mean}).sort_values('f1')

results.groupby(by=['sampler']).agg({'p_top20': np.mean}).sort_values('p_top20')
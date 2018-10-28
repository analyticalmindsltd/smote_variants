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
cache_path= '/home/gykovacs/workspaces/sampling_cache'
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

datasets= imbd.get_filtered_data_loaders(len_upper_bound= 1000,
                                         num_features_upper_bound= 50)

# instantiate the validation object
cv= sv.CacheAndValidate(samplers= sv.get_all_oversamplers(),
                       classifiers= classifiers,
                       datasets= datasets[:20],
                       cache_path= cache_path,
                       n_jobs= 5,
                       max_n_sampler_par_comb= 35)

#cv= sv.CacheAndValidate(samplers= sv.get_all_oversamplers(),
#                       classifiers= classifiers,
#                       datasets= [imbd.load_glass, imbd.load_iris0],
#                       cache_path= cache_path,
#                       n_jobs= 5,
#                       max_n_sampler_par_comb= 35)

# execute the validation
results= cv.cache_and_evaluate()

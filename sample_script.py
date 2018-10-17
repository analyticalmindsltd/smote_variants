from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import smote_variants as sv
from smote_variants import MLPClassifierWrapper, CacheAndValidate

# imbalanced databases
import imbalanced_databases as imbd

sv_classifiers= [CalibratedClassifierCV(LinearSVC(C=1.0, penalty='l1', loss= 'squared_hinge', dual= False)),
                 CalibratedClassifierCV(LinearSVC(C=1.0, penalty='l2', loss= 'hinge', dual= True)),
                 CalibratedClassifierCV(LinearSVC(C=1.0, penalty='l2', loss= 'squared_hinge', dual= False)),
                 CalibratedClassifierCV(LinearSVC(C=10.0, penalty='l1', loss= 'squared_hinge', dual= False)),
                 CalibratedClassifierCV(LinearSVC(C=10.0, penalty='l2', loss= 'hinge', dual= True)),
                 CalibratedClassifierCV(LinearSVC(C=10.0, penalty='l2', loss= 'squared_hinge', dual= False))]

mlp_classifiers= [MLPClassifierWrapper(activation='relu', hidden_layer_fraction= 1.0),
                  MLPClassifierWrapper(activation='relu', hidden_layer_fraction= 0.5),
                  MLPClassifierWrapper(activation='relu', hidden_layer_fraction= 0.1),
                  MLPClassifierWrapper(activation='logistic', hidden_layer_fraction= 1.0),
                  MLPClassifierWrapper(activation='logistic', hidden_layer_fraction= 0.5),
                  MLPClassifierWrapper(activation='logistic', hidden_layer_fraction= 0.1)]

nn_classifiers= [KNeighborsClassifier(n_neighbors= 3, weights='uniform', p=1),
                  KNeighborsClassifier(n_neighbors= 3, weights='uniform', p=2),
                  KNeighborsClassifier(n_neighbors= 3, weights='uniform', p=3),
                  KNeighborsClassifier(n_neighbors= 5, weights='uniform', p=1),
                  KNeighborsClassifier(n_neighbors= 5, weights='uniform', p=2),
                  KNeighborsClassifier(n_neighbors= 5, weights='uniform', p=3),
                  KNeighborsClassifier(n_neighbors= 7, weights='uniform', p=1),
                  KNeighborsClassifier(n_neighbors= 7, weights='uniform', p=2),
                  KNeighborsClassifier(n_neighbors= 7, weights='uniform', p=3),
                  KNeighborsClassifier(n_neighbors= 3, weights='distance', p=1),
                  KNeighborsClassifier(n_neighbors= 3, weights='distance', p=2),
                  KNeighborsClassifier(n_neighbors= 3, weights='distance', p=3),
                  KNeighborsClassifier(n_neighbors= 5, weights='distance', p=1),
                  KNeighborsClassifier(n_neighbors= 5, weights='distance', p=2),
                  KNeighborsClassifier(n_neighbors= 5, weights='distance', p=3),
                  KNeighborsClassifier(n_neighbors= 7, weights='distance', p=1),
                  KNeighborsClassifier(n_neighbors= 7, weights='distance', p=2),
                  KNeighborsClassifier(n_neighbors= 7, weights='distance', p=3)]

dt_classifiers= [DecisionTreeClassifier(criterion='gini', max_depth= None),
                 DecisionTreeClassifier(criterion='entropy', max_depth= None),
                 DecisionTreeClassifier(criterion='gini', max_depth= 3),
                 DecisionTreeClassifier(criterion='entropy', max_depth= 3),
                 DecisionTreeClassifier(criterion='gini', max_depth= 5),
                 DecisionTreeClassifier(criterion='entropy', max_depth= 5)]

classifiers= []
classifiers.extend(sv_classifiers)
classifiers.extend(mlp_classifiers)
classifiers.extend(nn_classifiers)
classifiers.extend(dt_classifiers)

tv= CacheAndValidate(samplers= sv.get_all_oversamplers(),
                       classifiers= classifiers,
                       datasets= [imbd.load_glass],
                       cache_path= '/home/gykovacs/workspaces/sampling_cache')

results= tv.cache_and_validate(max_n_sampler_par_comb= 35, n_jobs= 5)

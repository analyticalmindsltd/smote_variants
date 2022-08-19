
# coding: utf-8

# # Evaluation of the new oversampler on the standard database foldings
# 
# In this notebook we give an example evaluating a new oversampler on the standard 104 imbalanced datasets. The evaluation is highly similar to that illustrated in the notebook ```002_evaluation_multiple_datasets``` with the difference that in this case some predefined dataset foldings are used to make the results comparable to those reported in the ranking page of the documentation. The database foldings need to be downloaded from the github repository and placed in the 'smote_foldings' directory.

# In[1]:


import os, pickle, itertools

# import classifiers
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from smote_variants import MLPClassifierWrapper

# import SMOTE variants
import smote_variants as sv

# itertools to derive imbalanced databases
import imbalanced_databases as imbd


# In[2]:


# setting global parameters

folding_path= os.path.join(os.path.expanduser('~'), 'smote_foldings')

if not os.path.exists(folding_path):
    os.makedirs(folding_path)
    
max_sampler_parameter_combinations= 35
n_jobs= 5


# In[3]:


# instantiate classifiers

# Support Vector Classifiers with 6 parameter combinations
sv_classifiers= [CalibratedClassifierCV(LinearSVC(C=1.0, penalty='l1', loss= 'squared_hinge', dual= False)),
                 CalibratedClassifierCV(LinearSVC(C=1.0, penalty='l2', loss= 'hinge', dual= True)),
                 CalibratedClassifierCV(LinearSVC(C=1.0, penalty='l2', loss= 'squared_hinge', dual= False)),
                 CalibratedClassifierCV(LinearSVC(C=10.0, penalty='l1', loss= 'squared_hinge', dual= False)),
                 CalibratedClassifierCV(LinearSVC(C=10.0, penalty='l2', loss= 'hinge', dual= True)),
                 CalibratedClassifierCV(LinearSVC(C=10.0, penalty='l2', loss= 'squared_hinge', dual= False))]

# Multilayer Perceptron Classifiers with 6 parameter combinations
mlp_classifiers= []
for x in itertools.product(['relu', 'logistic'], [1.0, 0.5, 0.1]):
    mlp_classifiers.append(MLPClassifierWrapper(activation= x[0], hidden_layer_fraction= x[1]))

# Nearest Neighbor Classifiers with 18 parameter combinations
nn_classifiers= []
for x in itertools.product([3, 5, 7], ['uniform', 'distance'], [1, 2, 3]):
    nn_classifiers.append(KNeighborsClassifier(n_neighbors= x[0], weights= x[1], p= x[2]))

# Decision Tree Classifiers with 6 parameter combinations
dt_classifiers= []
for x in itertools.product(['gini', 'entropy'], [None, 3, 5]):
    dt_classifiers.append(DecisionTreeClassifier(criterion= x[0], max_depth= x[1]))

classifiers= []
classifiers.extend(sv_classifiers)
classifiers.extend(mlp_classifiers)
classifiers.extend(nn_classifiers)
classifiers.extend(dt_classifiers)


# In[4]:


# querying datasets for the evaluation

datasets= imbd.get_data_loaders('study')


# In[ ]:


# executing the evaluation

results= sv.evaluate_oversamplers(datasets,
                                  samplers= sv.get_all_oversamplers(),
                                  classifiers= classifiers,
                                  cache_path= folding_path,
                                  n_jobs= n_jobs,
                                  remove_sampling_cache= True,
                                  max_samp_par_comb= max_sampler_parameter_combinations)


# In[ ]:


# The evaluation results are available in the results dataframe for further analysis.

print(results)


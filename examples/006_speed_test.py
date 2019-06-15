
# coding: utf-8

# # Speed test
# 
# In this notebook, we compare the performance of the ```smote_variants``` package with that of the ```imblearn``` package through the three oversamplers implemented in common. Note that the implementations contain different logic to determine the number of samples to be generated. Generally, ```imblearn``` implementations are more flexible, ```smote_variants``` implementations are more simple to use.

# In[1]:


import smote_variants as sv
import imbalanced_databases as imbd
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE

import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


# In[2]:


def measure(sv, imb, datasets):
    """
    The function measuring the runtimes of oversamplers on a set of datasets.
    
    Args:
        sv (list(smote_variants.Oversampling)): the list of oversampling objects from smote_variants
        imb (list(imblearn.Oversampling)): the list of oversampling objects from imblearn, imb[i] is the
                                            implementation corresponding to sv[i]
        datasets (list(function)): dataset loading functions
    Returns:
        pd.DataFrame: mean oversampling runtimes for the various oversamplers over all datasets
    """
    
    results= {}
    # iterating through all datasets
    for d in datasets:
        data= d()
        print('processing: %s' % data['name'])
        
        X= data['data']
        y= data['target']
        for i, s in enumerate(sv):
            # imblearn seems to fail on some edge cases
            try:
                # measuring oversampling runtime using smote_variants
                t0= time.time()
                X_samp, y_samp= sv[i].sample(X, y)
                res_sv= time.time() - t0
                
                # measuring oversampling runtime using imblearn
                t0= time.time()
                X_samp, y_samp= imb[i].fit_resample(X, y)
                res_imb= time.time() - t0
                
                if not s.__class__.__name__ in results:
                    results[s.__class__.__name__]= ([], [])
                
                # appending the results
                results[s.__class__.__name__][0].append(res_sv)
                results[s.__class__.__name__][1].append(res_imb)
            except:
                pass
    
    # preparing the final dataframe
    for k in results:
        results[k]= [np.mean(results[k][0]), np.mean(results[k][1])]
    
    results= pd.DataFrame(results).T
    results.columns= ['smote_variants', 'imblearn']
    
    return results


# In[3]:


# Executing the evaluation for the techniques implemented by both smote_variants and imblearn, using the
# same parameters, involving 104 datasets

sv_techniques= [sv.SMOTE(), sv.Borderline_SMOTE2(k_neighbors=10), sv.ADASYN()]
imb_techniques= [SMOTE(), BorderlineSMOTE(), ADASYN()]

results= measure(sv_techniques,
                 imb_techniques,
                 imbd.get_data_loaders())


# In[4]:


# Printing the results, the unit is 'seconds'

print(results)


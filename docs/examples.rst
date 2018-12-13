Examples
********

Simple oversampling
===================

Oversampling can be carried out by importing any oversampler from the ``smote_variants`` package, instantiating and calling its ``sample`` function:

.. code-block:: Python

    import smote_variants as sv
      
    oversampler= sv.SMOTE_ENN()
    
    # supposing that X and y contain some the feature and target data of some dataset
    X_samp, y_samp= oversampler.sample(X, y)
    
Using the ``datasets`` package of ``sklearn`` to import some data:

.. code-block:: Python

    import smote_variants as sv
    import sklearn.datasets as datasets
    
    dataset= datasets.load_breast_cancer()
    
    oversampler= sv.KernelADASYN()
    
    X_samp, y_samp= oversampler.sample(dataset['data'], dataset['target'])
    
Using the imbalanced datasets available in the ``imbalanced_datasets`` package:

.. code-block:: Python

    import smote_variants as sv
    import imbalanced_datasets as imbd
    
    dataset= imbd.load_iris0()
    
    oversamplers= sv.SMOTE_OUT()
    
    X_samp, y_samp= oversampler.sample(dataset['data'], dataset['target'])

Oversampling with random, reasonable parameters
===============================================

In order to facilitate model selection, each oversampler class is able to generate a set of reasonable parameter combinations. Running an oversampler using a reasonable parameter combination:

.. code-block:: Python

    import numpy as np

    import smote_variants as sv
    import imbalanced datasets as imbd
    
    dataset= imbd.load_yeast1()
    
    par_combs= SMOTE_Cosine.parameter_combinations()
    
    oversampler= SMOTE_Cosine(**np.random.choice(par_combs))
    
    X_samp, y_samp= oversampler.sample(dataset['data'], dataset['target'])

Multiclass oversampling
=======================

Multiclass oversampling is highly ambiguous task, as balancing various classes might be optimal with various oversampling techniques. Currently, we have support for multiclass oversampling with one specific oversampler, and only those oversamplers can be used which do not change the majority class and have a ``proportion`` parameter to explicitly specify the number of samples to be generated. Suitable oversampling techniques can be queried by the ``get_all_oversamplers_multiclass`` function. In the below example the ``wine`` dataset is balanced by multiclass oversampling:

.. code-block:: Python

    import smote_variants as sv
    import sklearn.datasets as datasets
    
    dataset= datasets.load_wine()
    
    oversampler= sv.MulticlassOversampling(sv.distance_SMOTE())
    
    X_samp, y_samp= oversampler.sample(dataset['data'], dataset['target'])

Model selection
===============

When facing an imbalanced dataset, model selection is crucial to find the right oversampling approach and the right classifier. It is obvious that the best performing oversampling technique depends on the subsequent classification, thus, the model selection of oversampler and classifier needs to be carried out hand in hand. This is facilitated by the ``model_selection`` function of the package. One must specify a set of oversamplers and a set of classifiers, a score function (in this case 'AUC') to optimize in cross validation and the ``model_selection`` function does all the job:

.. code-block:: Python
    
    import smote_variants as sv
    import imbalanced_datasets as imbd
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    
    datasets= [imbd.load_glass2]
    oversamplers= [sv.SMOTE_ENN, sv.NEATER, sv.Lee]
    classifiers= [KNeighborsClassifier(n_neighbors= 3),
                    KNeighborsClassifier(n_neighbors= 5),
                    DecisionTreeClassifier()]
    
    cache_path= '/home/<user>/smote_validation/'
    
    sampler, classifier= model_selection(datasets,
                                            oversamplers,
                                            classifiers,
                                            cache_path,
                                            'auc',
                                            n_jobs= 10,
                                            max_n_sampler_parameters= 15)

Note, that we have also supplied a cache path, it is used to store partial results, samplings and cross validation scores. The ``n_jobs`` parameter specifies the number of oversampling and classification jobs to be executed in parallel, and ``max_n_sampler_parameters` specifies the maximum number of reasonable parameter combinations tested for each oversampler. The function call returns the best performing oversampling object and the corresponding, best performing classifier object, respecting the 'glass2' dataset.
                                             
Thorough evaluation involving multiple datasets
===============================================

Another scenario is the comparison and evaluation of a new oversampler to conventional ones involving a set of imbalance datasets. This scenario is facilitated by the ``evaluate_oversamplers`` function, which is parameterized similarly to ``model_selection``, but returns all the raw results of the numerous cross-validation scenarios (all datasets times (all oversamplers with ``max_n_sampler_parameters`` parameter combinations) times (all supplied classifiers)):

.. code-block:: Python

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
                                    cache_path,
                                    n_jobs= 10,
                                    max_n_sampler_parameters= 10)

Again, the function uses 10 parallel jobs to execute oversampling and classification. In the example above, 2 datasets, 3 classifiers and maximum 10 oversampler parameter combinations are specified for 3 oversampling objects, which requires 2x3x10x3 180 cross-validations altogether. In the resulting pandas DataFrame, for each classifier type (KNeighborsClassifier and DecisionTreeClassifier), and for each oversampler the highest performance measures and the corresponding classifier and oversampler parameters are returned. The structure of the DataFrame is self-explaining.

Reproducing the results in the comparative study
================================================

Although a 5-fold 3 times repeated stratified k-fold cross validation was executed, one might expect that the results still depend slightly on the foldings being used. In order to fully reproduce the results of the comparative study, download the foldings we use, and execute the following script by setting the cache_path to the path containing the downloaded foldings. The folding generator will pick-up and use the foldings supplied:

.. code-block:: Python

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

    # global variables
    folding_path= '/home/<user>/smote_foldings/'
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
        mlp_classifiers.append(MLPClassifierWrapper(activation= x[0], hidden_layer_fraction= x[1]))

    nn_classifiers= []
    for x in itertools.product([3, 5, 7], ['uniform', 'distance'], [1, 2, 3]):
        nn_classifiers.append(KNeighborsClassifier(n_neighbors= x[0], weights= x[1], p= x[2]))

    dt_classifiers= []
    for x in itertools.product(['gini', 'entropy'], [None, 3, 5]):
        dt_classifiers.append(DecisionTreeClassifier(criterion= x[0], max_depth= x[1]))

    classifiers= []
    classifiers.extend(sv_classifiers)
    classifiers.extend(mlp_classifiers)
    classifiers.extend(nn_classifiers)
    classifiers.extend(dt_classifiers)

    datasets= imbd.get_data_loaders('study')

    # instantiate the validation object
    results= sv.evaluate_oversamplers(datasets,
                                    samplers= sv.get_all_oversamplers(),
                                    classifiers= classifiers,
                                    cache_path= folding_path,
                                    n_jobs= n_jobs,
                                    remove_sampling_cache= True,
                                    max_n_sampler_parameters= max_sampler_parameter_combinations)


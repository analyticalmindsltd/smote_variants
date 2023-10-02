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

    oversampler= sv.MulticlassOversampling(oversampler='distance_SMOTE', oversampler_params={})

    X_samp, y_samp= oversampler.sample(dataset['data'], dataset['target'])

Model selection
===============

When facing an imbalanced dataset, model selection is crucial to find the right oversampling approach and the right classifier. It is obvious that the best performing oversampling technique depends on the subsequent classification, thus, the model selection of oversampler and classifier needs to be carried out hand in hand. This is facilitated by the ``model_selection`` function of the package. One must specify a set of oversamplers and a set of classifiers, a score function (in this case 'AUC') to optimize in cross validation and the ``model_selection`` function does all the job:

.. code-block:: Python

    import smote_variants as sv
    import imbalanced_datasets as imbd

    datasets = [imbd.load_glass2]
    oversamplers = sv.get_all_oversamplers(n_quickest=5)
    oversamplers = sv.generate_parameter_combinations(oversamplers,
                                                      n_max_comb=5)
    classifiers = [('sklearn.neighbors', 'KNeighborsClassifier', {'n_neighbors': 3}),
                  ('sklearn.neighbors', 'KNeighborsClassifier', {'n_neighbors': 5}),
                  ('sklearn.tree', 'DecisionTreeClassifier', {})]

    sampler, classifier= model_selection(datasets=datasets,
                                         oversamplers=oversamplers,
                                         classifiers=classifiers)

The function call returns the best performing oversampling object and the corresponding, best performing classifier object, respecting the 'glass2' dataset.

Thorough evaluation involving multiple datasets
===============================================

Another scenario is the comparison and evaluation of a new oversampler to conventional ones involving a set of imbalance datasets. This scenario is facilitated by the ``evaluate_oversamplers`` function, which is parameterized similarly to ``model_selection``, but returns all the raw results of the numerous cross-validation scenarios (all datasets times (all oversamplers with ``max_n_sampler_parameters`` parameter combinations) times (all supplied classifiers)):

.. code-block:: Python

    import smote_variants as sv
    import imbalanced_datasets as imbd

    datasets= [imbd.load_glass2, imbd.load_ecoli4]

    oversamplers = sv.get_all_oversamplers(n_quickest=5)

    oversamplers = sv.generate_parameter_combinations(oversamplers,
                                                      n_max_comb=5)

    classifiers = [('sklearn.neighbors', 'KNeighborsClassifier', {'n_neighbors': 3}),
                  ('sklearn.neighbors', 'KNeighborsClassifier', {'n_neighbors': 5}),
                  ('sklearn.tree', 'DecisionTreeClassifier', {})]

    results= evaluate_oversamplers(datasets=datasets,
                                   oversamplers=oversamplers,
                                   classifiers=classifiers,
                                   n_jobs= 10)

The function uses 10 parallel jobs to execute oversampling and classification. In the example above, 2 datasets, 3 classifiers and maximum 5 oversampler parameter combinations are specified for 3 oversampling objects, which requires 2x3x5x3 90 cross-validations altogether. In the resulting pandas DataFrame, for each classifier type (KNeighborsClassifier and DecisionTreeClassifier), and for each oversampler the highest performance measures and the corresponding classifier and oversampler parameters are returned. The structure of the DataFrame is self-explaining.


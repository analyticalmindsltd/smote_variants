Model selection, evaluation and validation
***************

Besides the oversampler implementation, we have prepared some codes for model selection compatible with ``sklearn`` classifier interfaces.

Having a dataset, a bunch of candidate oversamplers and classifiers, the tools below enable customizable model selection.

Caching
=======

The evaluation and comparison of oversampling techniques on many datasets might take enormous time. In order to increase the reliability
of an evaluation process, make it stoppable and restartable and let the oversampling techniques utilize results already computed, we
have implemented some model selection and evaluation scripts, both using some hard-disk cache directory to store partial and final results.
These functions cannot be used without specifying some cache directory.

Parallelization
===============

The evaluation and model selection scripts are executing oversampling and classification jobs in parallel. If the number of jobs specified
is 1, they will call the sklearn algorithms to run in parallel, otherwise the sklearn implementations run in sequential, and the oversampling
and classification jobs will be executed in parallel, using ``n_jobs`` processes.

Querying and filtering oversamplers
===================================

.. autofunction:: smote_variants.get_all_oversamplers

.. autofunction:: smote_variants.get_n_quickest_oversamplers

Evaluation and validation
=========================

.. autofunction:: smote_variants.evaluate_oversamplers

Model selection
===============

.. autofunction:: smote_variants.model_selection

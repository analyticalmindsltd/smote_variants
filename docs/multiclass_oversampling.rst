Multiclass oversampling
***********************

Multiclass oversampling is highly ambiguous task, as balancing various classes might be optimal with various oversampling techniques. The multiclass oversampling goes on by selecting minority classes one-by-one and oversampling them to the same cardinality as the original majority class, using the union of the original majority class and all already oversampled classes as the majority class in the binary oversampling process. This technique works only with those binary oversampling techniques which do not change the majority class and have a ``proportion`` parameter to explicitly specify the number of samples to be generated. Suitable oversampling techniques can be queried by the ``get_all_oversamplers_multiclass`` function:

.. autofunction:: smote_variants.get_all_oversamplers_multiclass

.. autofunction:: smote_variants.get_n_quickest_oversamplers_multiclass

.. autoclass:: smote_variants.MulticlassOversampling
   :members:
   
   .. automethod:: __init__


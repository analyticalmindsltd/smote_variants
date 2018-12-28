Installation
************

Prerequisites
=============

The following packages are requirements:

    * ``joblib``
    * ``numpy``
    * ``pandas``
    * ``sklearn``
    * ``scipy``
    * ``minisom``
    * ``keras`` (with any backend)
    
Optionally, consider installing the package ``imbalanced_databases`` for evaluation.

Installation
============

Install from PyPi
^^^^^^^^^^^^^^^^^

.. code-block:: bash

    > pip install smote_variants

For testing purposes, it is recommended to install the ``imbalanced_databases`` package:

.. code-block:: bash

    > pip install imbalanced_databases

Clone from GitHub
^^^^^^^^^^^^^^^^^

.. code-block:: bash
    
    > git clone git@github.com:gykovacs/smote_variants.git
    > cd smote_variants
    > pip install .
    
For out of box imbalanced databases consider installing the ``imbalanced_databases`` package:

.. code-block:: bash

    > git clone git@github.com:gykovacs/imbalanced_databases.git
    > cd imbalanced_databases
    > pip install .

Install directly from GitHub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    > pip install git+https://github.com:gykovacs/smote_variants.git

For out of box imbalanced databases consider installing the ``imbalanced_databases`` package, as well:

.. code-block:: bash

    > pip install git+https://github.com:gykovacs/imbalanced_databases.git


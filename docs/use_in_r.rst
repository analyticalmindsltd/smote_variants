Using ``smote_variants`` in R
*************************

All the implemented oversampling techniques can be called from R using the ``reticulate`` package. It needs a distinct, working Python installation, which then takes care about the conversion of data back and forth. Supposing that an Anaconda installation is available in the home directory of the user, with ``smote_variants`` and ``imbalanced_databases`` (to load imbalanced datasets easily) installed, the following R code works flawlessly.

.. code-block:: R

    library(reticulate)

    python_path <- file.path(file.expand('~'), 'anaconda3', 'bin', 'python')
    virtualenv_name <- 'base'

    use_python(python_path)
    use_virtualenv(virtualenv_name)

    imbd <- import("imbalanced_databases")
    sv <- import("smote_variants")

    data <- imbalanced_databases$load_iris0()
    sv$SMOTE()$sample(data$data, data$target)


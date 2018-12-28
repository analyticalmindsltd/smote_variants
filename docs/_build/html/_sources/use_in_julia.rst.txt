Using ``smote_variants`` in Julia
*****************************

Similarly to R using ``reticulate``, Python packages can be called from Julia using the package ``PyCall`` given that some python installation with ``smote_variants`` is available.

Suppose, there is an Anaconda3 install available at '/home/<user>/anaconda3' and ``smote_variants`` and ``imbalanced_databases`` are installed on the base conda environment.

The following steps are needed to run oversampler codes from the ``smote_variants`` package:

    * Start the Julia interpreter:
    
    .. code-block:: Julia
    
        julia

    * In the Julia prompt, set the Python path to that of the Anaconda install:
    
    .. code-block:: Julia
    
        ENV["PYTHON"]= "/home/<user>/anaconda3/bin/python3"
    
    * Add the ``PyCall`` package:

    .. code-block:: Julia
    
        import Pkg
        
        Pkg.add("PyCall")
        
        Pkg.build("PyCall")
        
    * Restart the Julia interpreter.
    * The following code should work:
    
    .. code-block:: Julia

        using PyCall
        
        @pyimport imbalanced_databases as imbd
        @pyimport smote_variants as sv
        
        dataset= imbd.load_iris0()
        oversampler= sv.SMOTE_ENN()
        
        X_samp, y_samp= oversampler[:sample](dataset["data"], dataset["target"])

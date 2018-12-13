.. _new_oversampler:

Adding a new oversampler
************************

Implementing a new oversampling logic which can be used in the model selection framework is easy:

    * It should inherit from the ``smote_variants.OverSampling`` class
    * Implement the class-level method ``parameter_combinations``, which returns a list of reasonable parameter combinations compatible with the constructor. A parameter combination in the list needs to be a dictionary which can be passed to the constructor of the object using the asterisk-operator.
    * It needs to implement the ``sample`` function, which takes a feature array and a target array.
    * Finally, it needs to implement the ``get_params`` function to return the parameters of an oversampling instance as a dictionary.
    
Below can be found a template for adding new oversamplers::
    
    class New_SMOTE_Variant(smote_variants.OverSampling):
        def __init__(self, param1, param2):
            super().__init__()
        
            self.param1= param1
            self.param2= param2
        
        @classmethod
        def parameter_combinations(cls):
            return [{'param1': 1, 'param2': 'a'}, 
                    {'param1': 2, 'param2': 'b'},
                    {'param1': 3, 'param2': 'c'}]
        
        def sample(self, X, y):
            # implement sampling logic here
            return X_samp, y_samp
            
        def get_params(self):
            return {'param1': self.param1, 'param2': self.param2}

An oversampler like this should work flawlessly with the model selection and evaluation scripts provided.

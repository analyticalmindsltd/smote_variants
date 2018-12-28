Noise filters and prototype selection
*************************************

TomekLinkRemoval
================


API
^^^

.. autoclass:: smote_variants.TomekLinkRemoval
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> noise_filter= smote_variants.TomekLinkRemoval()
    >>> X_samp, y_samp= noise_filter.remove_noise(X, y)


.. image:: figures/base.png
.. image:: figures/TomekLinkRemoval.png


Tomek link removal

References:
    * BibTex::
        
        @article{smoteNoise0,
                 author = {Batista, Gustavo E. A. P. A. and Prati, Ronaldo C. and Monard, Maria Carolina},
                 title = {A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data},
                 journal = {SIGKDD Explor. Newsl.},
                 issue_date = {June 2004},
                 volume = {6},
                 number = {1},
                 month = jun,
                 year = {2004},
                 issn = {1931-0145},
                 pages = {20--29},
                 numpages = {10},
                 url = {http://doi.acm.org/10.1145/1007730.1007735},
                 doi = {10.1145/1007730.1007735},
                 acmid = {1007735},
                 publisher = {ACM},
                 address = {New York, NY, USA}
                } 

    * URL: https://drive.google.com/open?id=1-AckPO4e4R3e3P3Zrsh6dVoFwRhL5Obx
CondensedNearestNeighbors
=========================


API
^^^

.. autoclass:: smote_variants.CondensedNearestNeighbors
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> noise_filter= smote_variants.CondensedNearestNeighbors()
    >>> X_samp, y_samp= noise_filter.remove_noise(X, y)


.. image:: figures/base.png
.. image:: figures/CondensedNearestNeighbors.png


Condensed nearest neighbors

References:
    * BibTex::
        
        @ARTICLE{condensed_nn, 
                    author={P. Hart}, 
                    journal={IEEE Transactions on Information Theory}, 
                    title={The condensed nearest neighbor rule (Corresp.)}, 
                    year={1968}, 
                    volume={14}, 
                    number={3}, 
                    pages={515-516}, 
                    keywords={Pattern classification}, 
                    doi={10.1109/TIT.1968.1054155}, 
                    ISSN={0018-9448}, 
                    month={May}}
OneSidedSelection
=================


API
^^^

.. autoclass:: smote_variants.OneSidedSelection
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> noise_filter= smote_variants.OneSidedSelection()
    >>> X_samp, y_samp= noise_filter.remove_noise(X, y)


.. image:: figures/base.png
.. image:: figures/OneSidedSelection.png


References:
    * BibTex::
        
        @article{smoteNoise0,
                 author = {Batista, Gustavo E. A. P. A. and Prati, Ronaldo C. and Monard, Maria Carolina},
                 title = {A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data},
                 journal = {SIGKDD Explor. Newsl.},
                 issue_date = {June 2004},
                 volume = {6},
                 number = {1},
                 month = jun,
                 year = {2004},
                 issn = {1931-0145},
                 pages = {20--29},
                 numpages = {10},
                 url = {http://doi.acm.org/10.1145/1007730.1007735},
                 doi = {10.1145/1007730.1007735},
                 acmid = {1007735},
                 publisher = {ACM},
                 address = {New York, NY, USA}
                } 
        
    * URL: https://drive.google.com/open?id=1-AckPO4e4R3e3P3Zrsh6dVoFwRhL5Obx
CNNTomekLinks
=============


API
^^^

.. autoclass:: smote_variants.CNNTomekLinks
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> noise_filter= smote_variants.CNNTomekLinks()
    >>> X_samp, y_samp= noise_filter.remove_noise(X, y)


.. image:: figures/base.png
.. image:: figures/CNNTomekLinks.png


References:
    * BibTex::
        
        @article{smoteNoise0,
                 author = {Batista, Gustavo E. A. P. A. and Prati, Ronaldo C. and Monard, Maria Carolina},
                 title = {A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data},
                 journal = {SIGKDD Explor. Newsl.},
                 issue_date = {June 2004},
                 volume = {6},
                 number = {1},
                 month = jun,
                 year = {2004},
                 issn = {1931-0145},
                 pages = {20--29},
                 numpages = {10},
                 url = {http://doi.acm.org/10.1145/1007730.1007735},
                 doi = {10.1145/1007730.1007735},
                 acmid = {1007735},
                 publisher = {ACM},
                 address = {New York, NY, USA}
                } 
        
    * URL: https://drive.google.com/open?id=1-AckPO4e4R3e3P3Zrsh6dVoFwRhL5Obx
NeighborhoodCleaningRule
========================


API
^^^

.. autoclass:: smote_variants.NeighborhoodCleaningRule
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> noise_filter= smote_variants.NeighborhoodCleaningRule()
    >>> X_samp, y_samp= noise_filter.remove_noise(X, y)


.. image:: figures/base.png
.. image:: figures/NeighborhoodCleaningRule.png


References:
    * BibTex::
        
        @article{smoteNoise0,
                 author = {Batista, Gustavo E. A. P. A. and Prati, Ronaldo C. and Monard, Maria Carolina},
                 title = {A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data},
                 journal = {SIGKDD Explor. Newsl.},
                 issue_date = {June 2004},
                 volume = {6},
                 number = {1},
                 month = jun,
                 year = {2004},
                 issn = {1931-0145},
                 pages = {20--29},
                 numpages = {10},
                 url = {http://doi.acm.org/10.1145/1007730.1007735},
                 doi = {10.1145/1007730.1007735},
                 acmid = {1007735},
                 publisher = {ACM},
                 address = {New York, NY, USA}
                } 
        
    * URL: https://drive.google.com/open?id=1-AckPO4e4R3e3P3Zrsh6dVoFwRhL5Obx
EditedNearestNeighbors
======================


API
^^^

.. autoclass:: smote_variants.EditedNearestNeighbors
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> noise_filter= smote_variants.EditedNearestNeighbors()
    >>> X_samp, y_samp= noise_filter.remove_noise(X, y)


.. image:: figures/base.png
.. image:: figures/EditedNearestNeighbors.png


References:
    * BibTex::
        
        @article{smoteNoise0,
                 author = {Batista, Gustavo E. A. P. A. and Prati, Ronaldo C. and Monard, Maria Carolina},
                 title = {A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data},
                 journal = {SIGKDD Explor. Newsl.},
                 issue_date = {June 2004},
                 volume = {6},
                 number = {1},
                 month = jun,
                 year = {2004},
                 issn = {1931-0145},
                 pages = {20--29},
                 numpages = {10},
                 url = {http://doi.acm.org/10.1145/1007730.1007735},
                 doi = {10.1145/1007730.1007735},
                 acmid = {1007735},
                 publisher = {ACM},
                 address = {New York, NY, USA}
                } 
        
    * URL: https://drive.google.com/open?id=1-AckPO4e4R3e3P3Zrsh6dVoFwRhL5Obx

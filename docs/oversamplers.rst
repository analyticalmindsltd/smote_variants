Oversamplers
************

SMOTE
-----


API
^^^

.. autoclass:: smote_variants.SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SMOTE.png


References:
    * BibTex::
        
        @article{smote,
            author={N. V. Chawla and K. W. Bowyer and L. O. Hall and W. P. Kegelmeyer},
            title={{SMOTE}: synthetic minority over-sampling technique},
            journal={Journal of Artificial Intelligence Research},
            volume={16},
            year={2002},
            pages={321--357}
          }
    
    * URL: https://drive.google.com/open?id=1DSPXx8aaVkoNASNPue-O2_4OTu5HzZ2Z
SMOTE_TomekLinks
----------------


API
^^^

.. autoclass:: smote_variants.SMOTE_TomekLinks
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SMOTE_TomekLinks()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SMOTE_TomekLinks.png


References:
    * BibTex::
        
        @article{smote_tomeklinks_enn,
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
                 address = {New York, NY, USA},
                } 
        
    * URL: https://drive.google.com/open?id=1-AckPO4e4R3e3P3Zrsh6dVoFwRhL5Obx
SMOTE_ENN
---------


API
^^^

.. autoclass:: smote_variants.SMOTE_ENN
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SMOTE_ENN()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SMOTE_ENN.png


References:
    * BibTex::
        
        @article{smote_tomeklinks_enn,
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
                 address = {New York, NY, USA},
                } 
        
    * URL: https://drive.google.com/open?id=1-AckPO4e4R3e3P3Zrsh6dVoFwRhL5Obx
Notes:
    * Can remove too many of minority samples.
Borderline_SMOTE1
-----------------


API
^^^

.. autoclass:: smote_variants.Borderline_SMOTE1
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.Borderline_SMOTE1()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/Borderline_SMOTE1.png


References:
    * BibTex::
        
        @InProceedings{borderlineSMOTE,
                        author="Han, Hui
                        and Wang, Wen-Yuan
                        and Mao, Bing-Huan",
                        editor="Huang, De-Shuang
                        and Zhang, Xiao-Ping
                        and Huang, Guang-Bin",
                        title="Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning",
                        booktitle="Advances in Intelligent Computing",
                        year="2005",
                        publisher="Springer Berlin Heidelberg",
                        address="Berlin, Heidelberg",
                        pages="878--887",
                        isbn="978-3-540-31902-3"
                        }
        
    * URL: https://drive.google.com/open?id=1dlG3wtxMIuiWgmd08nP9KN-Wq7dXTReb
Borderline_SMOTE2
-----------------


API
^^^

.. autoclass:: smote_variants.Borderline_SMOTE2
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.Borderline_SMOTE2()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/Borderline_SMOTE2.png


References:
    * BibTex::
        
        @InProceedings{borderlineSMOTE,
                        author="Han, Hui
                        and Wang, Wen-Yuan
                        and Mao, Bing-Huan",
                        editor="Huang, De-Shuang
                        and Zhang, Xiao-Ping
                        and Huang, Guang-Bin",
                        title="Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning",
                        booktitle="Advances in Intelligent Computing",
                        year="2005",
                        publisher="Springer Berlin Heidelberg",
                        address="Berlin, Heidelberg",
                        pages="878--887",
                        isbn="978-3-540-31902-3"
                        }
        
    * URL: https://drive.google.com/open?id=1dlG3wtxMIuiWgmd08nP9KN-Wq7dXTReb
ADASYN
------


API
^^^

.. autoclass:: smote_variants.ADASYN
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.ADASYN()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/ADASYN.png


References:
    * BibTex::
        
        @inproceedings{adasyn,
                      author={H. He and Y. Bai and E. A. Garcia and S. Li},
                      title={{ADASYN}: adaptive synthetic sampling approach for imbalanced learning},
                      booktitle={Proceedings of IJCNN},
                      year={2008},
                      pages={1322--1328}
                    }
    
    * URL: https://drive.google.com/open?id=1CiybjtmNVe4wo3t36VG82lB10IBmjv17
AHC
---


API
^^^

.. autoclass:: smote_variants.AHC
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.AHC()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/AHC.png


References:
    * BibTex::
        
        @article{AHC,
                title = "Learning from imbalanced data in surveillance of nosocomial infection",
                journal = "Artificial Intelligence in Medicine",
                volume = "37",
                number = "1",
                pages = "7 - 18",
                year = "2006",
                note = "Intelligent Data Analysis in Medicine",
                issn = "0933-3657",
                doi = "https://doi.org/10.1016/j.artmed.2005.03.002",
                url = "http://www.sciencedirect.com/science/article/pii/S0933365705000850",
                author = "Gilles Cohen and Mélanie Hilario and Hugo Sax and Stéphane Hugonnet and Antoine Geissbuhler",
                keywords = "Nosocomial infection, Machine learning, Support vector machines, Data imbalance"
                }

    * URL: https://drive.google.com/open?id=1APnBwng3-AZofx3FxMaKnR-Su-6ItDUM
LLE_SMOTE
---------


API
^^^

.. autoclass:: smote_variants.LLE_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.LLE_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/LLE_SMOTE.png


References:
    * BibTex::
        
        @INPROCEEDINGS{lle_smote, 
                        author={J. Wang and M. Xu and H. Wang and J. Zhang}, 
                        booktitle={2006 8th international Conference on Signal Processing}, 
                        title={Classification of Imbalanced Data by Using the SMOTE Algorithm and Locally Linear Embedding}, 
                        year={2006}, 
                        volume={3}, 
                        number={}, 
                        pages={}, 
                        keywords={artificial intelligence;biomedical imaging;medical computing;imbalanced data classification;SMOTE algorithm;locally linear embedding;medical imaging intelligence;synthetic minority oversampling technique;high-dimensional data;low-dimensional space;Biomedical imaging;Back;Training data;Data mining;Biomedical engineering;Research and development;Electronic mail;Pattern recognition;Performance analysis;Classification algorithms}, 
                        doi={10.1109/ICOSP.2006.345752}, 
                        ISSN={2164-5221}, 
                        month={Nov}}
        
    * URL: https://drive.google.com/open?id=1gCPLdTq_5mhF5cKGSmJdkPzhw2GY2SWs

Notes:
    * There might be numerical issues if the nearest neighbors contain some element multiple times.
distance_SMOTE
--------------


API
^^^

.. autoclass:: smote_variants.distance_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.distance_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/distance_SMOTE.png


References:
    * BibTex::
        
        @INPROCEEDINGS{distance_smote, 
                        author={de la Calleja, J. and Fuentes, O.}, 
                        booktitle={Proceedings of the Twentieth International Florida Artificial Intelligence}, 
                        title={A distance-based over-sampling method for learning from imbalanced data sets}, 
                        year={2007}, 
                        volume={3}, 
                        pages={634--635}
                        }
        
    * URL: https://drive.google.com/open?id=1O7tGVLXdZwC8N1TxGblw0J8n70FYspDc
    
Notes:
    * It is not clear what the authors mean by "weighted distance".
SMMO
----


API
^^^

.. autoclass:: smote_variants.SMMO
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SMMO()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SMMO.png


References:
    * BibTex::
        
        @InProceedings{smmo,
                        author = {de la Calleja, Jorge and Fuentes, Olac and González, Jesús},
                        booktitle= {Proceedings of the Twenty-First International Florida Artificial Intelligence Research Society Conference},
                        year = {2008},
                        month = {01},
                        pages = {276-281},
                        title = {Selecting Minority Examples from Misclassified Data for Over-Sampling.}
                        }
        
    * URL: https://drive.google.com/open?id=1hPEez2lVZ9wVV4dZjZQgK0lcl_jNt59g

Notes:
    * In this implementation the ensemble is not specified. I have selected some very fast, basic classifiers.
    * Also, it is not clear what the authors mean by "weighted distance".
    * The original technique is not prepared for the case when no minority samples are classified correctly be the ensemble.
polynom_fit_SMOTE
-----------------


API
^^^

.. autoclass:: smote_variants.polynom_fit_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.polynom_fit_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/polynom_fit_SMOTE.png


References:
    * BibTex::
        
        @INPROCEEDINGS{polynomial_fit_smote, 
                        author={S. Gazzah and N. E. B. Amara}, 
                        booktitle={2008 The Eighth IAPR International Workshop on Document Analysis Systems}, 
                        title={New Oversampling Approaches Based on Polynomial Fitting for Imbalanced Data Sets}, 
                        year={2008}, 
                        volume={}, 
                        number={}, 
                        pages={677-684}, 
                        keywords={curve fitting;learning (artificial intelligence);mesh generation;pattern classification;polynomials;sampling methods;support vector machines;oversampling approach;polynomial fitting function;imbalanced data set;pattern classification task;class-modular strategy;support vector machine;true negative rate;true positive rate;star topology;bus topology;polynomial curve topology;mesh topology;Polynomials;Topology;Support vector machines;Support vector machine classification;Pattern classification;Performance evaluation;Training data;Text analysis;Data engineering;Convergence;writer identification system;majority class;minority class;imbalanced data sets;polynomial fitting functions;class-modular strategy}, 
                        doi={10.1109/DAS.2008.74}, 
                        ISSN={}, 
                        month={Sept},}
        
    * URL: https://drive.google.com/open?id=1WkGbFBqCV8vnUh7yM97kO6EPlEgkC51P
Stefanowski
-----------


API
^^^

.. autoclass:: smote_variants.Stefanowski
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.Stefanowski()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/Stefanowski.png


References:
    * BibTex::
        
        @inproceedings{stefanowski,
             author = {Stefanowski, Jerzy and Wilk, Szymon},
             title = {Selective Pre-processing of Imbalanced Data for Improving Classification Performance},
             booktitle = {Proceedings of the 10th International Conference on Data Warehousing and Knowledge Discovery},
             series = {DaWaK '08},
             year = {2008},
             isbn = {978-3-540-85835-5},
             location = {Turin, Italy},
             pages = {283--292},
             numpages = {10},
             url = {http://dx.doi.org/10.1007/978-3-540-85836-2_27},
             doi = {10.1007/978-3-540-85836-2_27},
             acmid = {1430591},
             publisher = {Springer-Verlag},
             address = {Berlin, Heidelberg},
            } 

    * URL: https://drive.google.com/open?id=1MMrk-QnEfr0SUgptQkRl7Abbmh9ZncpD
ADOMS
-----


API
^^^

.. autoclass:: smote_variants.ADOMS
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.ADOMS()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/ADOMS.png


References:
    * BibTex::
        
        @INPROCEEDINGS{adoms, 
                        author={S. Tang and S. Chen}, 
                        booktitle={2008 International Conference on Information Technology and Applications in Biomedicine}, 
                        title={The generation mechanism of synthetic minority class examples}, 
                        year={2008}, 
                        volume={}, 
                        number={}, 
                        pages={444-447}, 
                        keywords={medical image processing;generation mechanism;synthetic minority class examples;class imbalance problem;medical image analysis;oversampling algorithm;Principal component analysis;Biomedical imaging;Medical diagnostic imaging;Information technology;Biomedical engineering;Noise generators;Concrete;Nearest neighbor searches;Data analysis;Image analysis}, 
                        doi={10.1109/ITAB.2008.4570642}, 
                        ISSN={2168-2194}, 
                        month={May}}

    * URL: https://drive.google.com/open?id=1NHrfqf9tPYwdOMTd7gQAl49Z8Mppwv_n
Safe_Level_SMOTE
----------------


API
^^^

.. autoclass:: smote_variants.Safe_Level_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.Safe_Level_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/Safe_Level_SMOTE.png


References:
    * BibTex::
        
        @inproceedings{safe_level_smote,
                     author = {Bunkhumpornpat, Chumphol and Sinapiromsaran, Krung and Lursinsap, Chidchanok},
                     title = {Safe-Level-SMOTE: Safe-Level-Synthetic Minority Over-Sampling TEchnique for Handling the Class Imbalanced Problem},
                     booktitle = {Proceedings of the 13th Pacific-Asia Conference on Advances in Knowledge Discovery and Data Mining},
                     series = {PAKDD '09},
                     year = {2009},
                     isbn = {978-3-642-01306-5},
                     location = {Bangkok, Thailand},
                     pages = {475--482},
                     numpages = {8},
                     url = {http://dx.doi.org/10.1007/978-3-642-01307-2_43},
                     doi = {10.1007/978-3-642-01307-2_43},
                     acmid = {1533904},
                     publisher = {Springer-Verlag},
                     address = {Berlin, Heidelberg},
                     keywords = {Class Imbalanced Problem, Over-sampling, SMOTE, Safe Level},
                    } 
        
    * URL: https://drive.google.com/open?id=18XNDTxIYeQ9GMocEXU_-zyj3W_5ovplR
    
Notes:
    * The original method was not prepared for the case when no minority sample has minority neighbors.
MSMOTE
------


API
^^^

.. autoclass:: smote_variants.MSMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.MSMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/MSMOTE.png


References:
    * BibTex::
        
        @inproceedings{msmote,
                         author = {Hu, Shengguo and Liang, Yanfeng and Ma, Lintao and He, Ying},
                         title = {MSMOTE: Improving Classification Performance When Training Data is Imbalanced},
                         booktitle = {Proceedings of the 2009 Second International Workshop on Computer Science and Engineering - Volume 02},
                         series = {IWCSE '09},
                         year = {2009},
                         isbn = {978-0-7695-3881-5},
                         pages = {13--17},
                         numpages = {5},
                         url = {https://doi.org/10.1109/WCSE.2009.756},
                         doi = {10.1109/WCSE.2009.756},
                         acmid = {1682710},
                         publisher = {IEEE Computer Society},
                         address = {Washington, DC, USA},
                         keywords = {imbalanced data, over-sampling, SMOTE, AdaBoost, samples groups, SMOTEBoost},
                        } 

    * URL: https://drive.google.com/open?id=1tFtNJWUSIYDKnhBAdb6QqIhYqy-khIxa

Notes:
    * The original method was not prepared for the case when all minority samples are noise.
DE_oversampling
---------------


API
^^^

.. autoclass:: smote_variants.DE_oversampling
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.DE_oversampling()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/DE_oversampling.png


References:
    * BibTex::
        
        @INPROCEEDINGS{de_oversampling, 
                        author={L. Chen and Z. Cai and L. Chen and Q. Gu}, 
                        booktitle={2010 Third International Conference on Knowledge Discovery and Data Mining}, 
                        title={A Novel Differential Evolution-Clustering Hybrid Resampling Algorithm on Imbalanced Datasets}, 
                        year={2010}, 
                        volume={}, 
                        number={}, 
                        pages={81-85}, 
                        keywords={pattern clustering;sampling methods;support vector machines;differential evolution;clustering algorithm;hybrid resampling algorithm;imbalanced datasets;support vector machine;minority class;mutation operators;crossover operators;data cleaning method;F-measure criterion;ROC area criterion;Support vector machines;Intrusion detection;Support vector machine classification;Cleaning;Electronic mail;Clustering algorithms;Signal to noise ratio;Learning systems;Data mining;Geology;imbalanced datasets;hybrid resampling;clustering;differential evolution;support vector machine}, 
                        doi={10.1109/WKDD.2010.48}, 
                        ISSN={}, 
                        month={Jan},}

    * URL: https://drive.google.com/open?id=1LyfMvSdFqscupz4AADXV4GW-K8T3olwK
SMOBD
-----


API
^^^

.. autoclass:: smote_variants.SMOBD
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SMOBD()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SMOBD.png


References:
    * BibTex::
        
        @INPROCEEDINGS{smobd, 
                        author={Q. Cao and S. Wang}, 
                        booktitle={2011 International Conference on Information Management, Innovation Management and Industrial Engineering}, 
                        title={Applying Over-sampling Technique Based on Data Density and Cost-sensitive SVM to Imbalanced Learning}, 
                        year={2011}, 
                        volume={2}, 
                        number={}, 
                        pages={543-548}, 
                        keywords={data handling;learning (artificial intelligence);support vector machines;oversampling technique application;data density;cost sensitive SVM;imbalanced learning;SMOTE algorithm;data distribution;density information;Support vector machines;Classification algorithms;Noise measurement;Arrays;Noise;Algorithm design and analysis;Training;imbalanced learning;cost-sensitive SVM;SMOTE;data density;SMOBD}, 
                        doi={10.1109/ICIII.2011.276}, 
                        ISSN={2155-1456}, 
                        month={Nov},}

    * URL: https://drive.google.com/open?id=1jQGTZli3D2RB3y2oe50hwFzo0q6y73FI
SUNDO
-----


API
^^^

.. autoclass:: smote_variants.SUNDO
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SUNDO()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SUNDO.png


References:
    * BibTex::
        
        @INPROCEEDINGS{sundo, 
                        author={S. Cateni and V. Colla and M. Vannucci}, 
                        booktitle={2011 11th International Conference on Intelligent Systems Design and Applications}, 
                        title={Novel resampling method for the classification of imbalanced datasets for industrial and other real-world problems}, 
                        year={2011}, 
                        volume={}, 
                        number={}, 
                        pages={402-407}, 
                        keywords={decision trees;pattern classification;sampling methods;support vector machines;resampling method;imbalanced dataset classification;industrial problem;real world problem;oversampling technique;undersampling technique;support vector machine;decision tree;binary classification;synthetic dataset;public dataset;industrial dataset;Support vector machines;Training;Accuracy;Databases;Intelligent systems;Breast cancer;Decision trees;oversampling;undersampling;imbalanced dataset}, 
                        doi={10.1109/ISDA.2011.6121689}, 
                        ISSN={2164-7151}, 
                        month={Nov}}
        
    * URL: https://drive.google.com/open?id=1lVwDDE-wTx3bsA7HbwyQ2ifRX5BmO8rq
MSYN
----


API
^^^

.. autoclass:: smote_variants.MSYN
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.MSYN()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/MSYN.png


References:
    * BibTex::
        
        @InProceedings{msyn,
                        author="Fan, Xiannian
                        and Tang, Ke
                        and Weise, Thomas",
                        editor="Huang, Joshua Zhexue
                        and Cao, Longbing
                        and Srivastava, Jaideep",
                        title="Margin-Based Over-Sampling Method for Learning from Imbalanced Datasets",
                        booktitle="Advances in Knowledge Discovery and Data Mining",
                        year="2011",
                        publisher="Springer Berlin Heidelberg",
                        address="Berlin, Heidelberg",
                        pages="309--320",
                        abstract="Learning from imbalanced datasets has drawn more and more attentions from both theoretical and practical aspects. Over- sampling is a popular and simple method for imbalanced learning. In this paper, we show that there is an inherently potential risk associated with the over-sampling algorithms in terms of the large margin principle. Then we propose a new synthetic over sampling method, named Margin-guided Synthetic Over-sampling (MSYN), to reduce this risk. The MSYN improves learning with respect to the data distributions guided by the margin-based rule. Empirical study verities the efficacy of MSYN.",
                        isbn="978-3-642-20847-8"
                        }
        
    * URL: https://drive.google.com/open?id=1i1ah7i4JfSoD8j5AJiP9Lx3-DIniKeYN
SVM_balance
-----------


API
^^^

.. autoclass:: smote_variants.SVM_balance
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SVM_balance()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SVM_balance.png


References:
    * BibTex::
        
        @article{svm_balance,
                 author = {Farquad, M.A.H. and Bose, Indranil},
                 title = {Preprocessing Unbalanced Data Using Support Vector Machine},
                 journal = {Decis. Support Syst.},
                 issue_date = {April, 2012},
                 volume = {53},
                 number = {1},
                 month = apr,
                 year = {2012},
                 issn = {0167-9236},
                 pages = {226--233},
                 numpages = {8},
                 url = {http://dx.doi.org/10.1016/j.dss.2012.01.016},
                 doi = {10.1016/j.dss.2012.01.016},
                 acmid = {2181554},
                 publisher = {Elsevier Science Publishers B. V.},
                 address = {Amsterdam, The Netherlands, The Netherlands},
                 keywords = {COIL data, Hybrid method, Preprocessor, SVM, Unbalanced data},
                } 

    * URL: https://drive.google.com/open?id=1DWDPQhJfzvUFgGAeAej-Xtlz5zX7trPz
TRIM_SMOTE
----------


API
^^^

.. autoclass:: smote_variants.TRIM_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.TRIM_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/TRIM_SMOTE.png


References:
    * BibTex::
        
        @InProceedings{trim_smote,
                        author="Puntumapon, Kamthorn
                        and Waiyamai, Kitsana",
                        editor="Tan, Pang-Ning
                        and Chawla, Sanjay
                        and Ho, Chin Kuan
                        and Bailey, James",
                        title="A Pruning-Based Approach for Searching Precise and Generalized Region for Synthetic Minority Over-Sampling",
                        booktitle="Advances in Knowledge Discovery and Data Mining",
                        year="2012",
                        publisher="Springer Berlin Heidelberg",
                        address="Berlin, Heidelberg",
                        pages="371--382",
                        abstract="One solution to deal with class imbalance is to modify its class distribution. Synthetic over-sampling is a well-known method to modify class distribution by generating new synthetic minority data. Synthetic Minority Over-sampling TEchnique (SMOTE) is a state-of-the-art synthetic over-sampling algorithm that generates new synthetic data along the line between the minority data and their selected nearest neighbors. Advantages of SMOTE is to have decision regions larger and less specific to original data. However, its drawback is the over-generalization problem where synthetic data is generated into majority class region. Over-generalization leads to misclassify non-minority class region into minority class. To overcome the over-generalization problem, we propose an algorithm, called TRIM, to search for precise minority region while maintaining its generalization. TRIM iteratively filters out irrelevant majority data from the precise minority region. Output of the algorithm is the multiple set of seed minority data, and each individual set will be used for generating new synthetic data. Compared with state-of-the-art over-sampling algorithms, experimental results show significant performance improvement in terms of F-measure and AUC. This suggests over-generalization has a significant impact on the performance of the synthetic over-sampling method.",
                        isbn="978-3-642-30220-6"
                        }

    * URL: https://drive.google.com/open?id=1VIUmLYe29YeJeXHDAyD5zHf7Ban8RTZf

Notes:
    * It is not described precisely how the filtered data is used for sample generation. The method is proposed to be a preprocessing step, and it states that it applies sample generation to each group extracted. 
SMOTE_RSB
---------


API
^^^

.. autoclass:: smote_variants.SMOTE_RSB
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SMOTE_RSB()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SMOTE_RSB.png


References:
    * BibTex::
        
        @Article{smote_rsb,
                author="Ramentol, Enislay
                and Caballero, Yail{'e}
                and Bello, Rafael
                and Herrera, Francisco",
                title="SMOTE-RSB*: a hybrid preprocessing approach based on oversampling and undersampling for high imbalanced data-sets using SMOTE and rough sets theory",
                journal="Knowledge and Information Systems",
                year="2012",
                month="Nov",
                day="01",
                volume="33",
                number="2",
                pages="245--265",
                abstract="Imbalanced data is a common problem in classification. This phenomenon is growing in importance since it appears in most real domains. It has special relevance to highly imbalanced data-sets (when the ratio between classes is high). Many techniques have been developed to tackle the problem of imbalanced training sets in supervised learning. Such techniques have been divided into two large groups: those at the algorithm level and those at the data level. Data level groups that have been emphasized are those that try to balance the training sets by reducing the larger class through the elimination of samples or increasing the smaller one by constructing new samples, known as undersampling and oversampling, respectively. This paper proposes a new hybrid method for preprocessing imbalanced data-sets through the construction of new samples, using the Synthetic Minority Oversampling Technique together with the application of an editing technique based on the Rough Set Theory and the lower approximation of a subset. The proposed method has been validated by an experimental study showing good results using C4.5 as the learning algorithm.",
                issn="0219-3116",
                doi="10.1007/s10115-011-0465-6",
                url="https://doi.org/10.1007/s10115-011-0465-6"
                }
    
    * URL: https://drive.google.com/open?id=1erSr3NWqBNXO1dCK1h2DMfEzdtaHDTJB

Notes:
    * I think the description of the algorithm in Fig 5 of the paper is not correct. The set "resultSet" is initialized with the original instances, and then the While loop in the Algorithm run until resultSet is empty, which never holds. Also, the resultSet is only extended in the loop. Our implementation is changed in the following way: we generate twice as many instances are required to balance the dataset, and repeat the loop until the number of new samples added to the training set is enough to balance the dataset.
ProWSyn
-------


API
^^^

.. autoclass:: smote_variants.ProWSyn
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.ProWSyn()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/ProWSyn.png


References:
    * BibTex::
        
        @InProceedings{prowsyn,
                    author="Barua, Sukarna
                    and Islam, Md. Monirul
                    and Murase, Kazuyuki",
                    editor="Pei, Jian
                    and Tseng, Vincent S.
                    and Cao, Longbing
                    and Motoda, Hiroshi
                    and Xu, Guandong",
                    title="ProWSyn: Proximity Weighted Synthetic Oversampling Technique for Imbalanced Data Set Learning",
                    booktitle="Advances in Knowledge Discovery and Data Mining",
                    year="2013",
                    publisher="Springer Berlin Heidelberg",
                    address="Berlin, Heidelberg",
                    pages="317--328",
                    abstract="An imbalanced data set creates severe problems for the classifier as number of samples of one class (majority) is much higher than the other class (minority). Synthetic oversampling methods address this problem by generating new synthetic minority class samples. To distribute the synthetic samples effectively, recent approaches create weight values for original minority samples based on their importance and distribute synthetic samples according to weight values. However, most of the existing algorithms create inappropriate weights and in many cases, they cannot generate the required weight values for the minority samples. This results in a poor distribution of generated synthetic samples. In this respect, this paper presents a new synthetic oversampling algorithm, Proximity Weighted Synthetic Oversampling Technique (ProWSyn). Our proposed algorithm generate effective weight values for the minority data samples based on sample's proximity information, i.e., distance from boundary which results in a proper distribution of generated synthetic samples across the minority data set. Simulation results on some real world datasets shows the effectiveness of the proposed method showing improvements in various assessment metrics such as AUC, F-measure, and G-mean.",
                    isbn="978-3-642-37456-2"
                    }
    
    * URL: https://drive.google.com/open?id=12yiLJc0XOT6tjVN7yIPBKJa-eIC229s3
SL_graph_SMOTE
--------------


API
^^^

.. autoclass:: smote_variants.SL_graph_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SL_graph_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SL_graph_SMOTE.png


References:
    * BibTex::
        
        @inproceedings{sl_graph_smote,
                author = {Bunkhumpornpat, Chumpol and Subpaiboonkit, Sitthichoke},
                booktitle= {13th International Symposium on Communications and Information Technologies},
                year = {2013},
                month = {09},
                pages = {570-575},
                title = {Safe level graph for synthetic minority over-sampling techniques},
                isbn = {978-1-4673-5578-0}
                }
    
    * URL: https://drive.google.com/open?id=1UxPfhGjM9KA7eXA4yhPvfYBjbPp8fZeT
NRSBoundary_SMOTE
-----------------


API
^^^

.. autoclass:: smote_variants.NRSBoundary_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.NRSBoundary_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/NRSBoundary_SMOTE.png


References:
    * BibTex::
        
        @Article{nrsboundary_smote,
                author= {Feng, Hu and Hang, Li},
                title= {A Novel Boundary Oversampling Algorithm Based on Neighborhood Rough Set Model: NRSBoundary-SMOTE},
                journal= {Mathematical Problems in Engineering},
                year= {2013},
                pages= {10},
                doi= {10.1155/2013/694809},
                url= {http://dx.doi.org/10.1155/694809}
                }
    
    * URL: https://drive.google.com/open?id=1CdBzRHdcKmvGB6bkRSv6_ZviMeuwCZDT
LVQ_SMOTE
---------


API
^^^

.. autoclass:: smote_variants.LVQ_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.LVQ_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/LVQ_SMOTE.png


References:
    * BibTex::
        
        @inproceedings{lvq_smote,
                          title={LVQ-SMOTE – Learning Vector Quantization based Synthetic Minority Over–sampling Technique for biomedical data},
                          author={Munehiro Nakamura and Yusuke Kajiwara and Atsushi Otsuka and Haruhiko Kimura},
                          booktitle={BioData Mining},
                          year={2013}
                        }
    
    * URL: https://drive.google.com/open?id=18ecI_0tYQG-1nRm8GhFRXZxFVPw7CzfD

Notes:
    * This implementation is only a rough estimation of the method described in the paper. The main problem is that the paper uses many datasets to find similar patterns in the codebooks and replicate patterns appearing in other datasets to the imbalanced datasets based on their relative position compared to the codebook elements. What we do is clustering the minority class to extract a codebook as kmeans cluster means, then, find pairs of codebook elements which have the most similar relative position to a randomly selected pair of codebook elements, and translate nearby minority samples from the neighborhood one pair of codebook elements to the neighborood of another pair of codebook elements.
SOI_CJ
------


API
^^^

.. autoclass:: smote_variants.SOI_CJ
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SOI_CJ()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SOI_CJ.png


References:
    * BibTex::
        
        @article{soi_cj,
                author = {I. Sánchez, Atlántida and Morales, Eduardo and Gonzalez, Jesus},
                year = {2013},
                month = {01},
                pages = {},
                title = {Synthetic Oversampling of Instances Using Clustering},
                volume = {22},
                booktitle = {International Journal of Artificial Intelligence Tools}
                }
    
    * URL: https://drive.google.com/open?id=13GzuVxVZtelG6VNW_Ic8hlxge3XhCYKq
ROSE
----


API
^^^

.. autoclass:: smote_variants.ROSE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.ROSE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/ROSE.png


References:
    * BibTex::
        
        @Article{rose,
                author="Menardi, Giovanna
                and Torelli, Nicola",
                title="Training and assessing classification rules with imbalanced data",
                journal="Data Mining and Knowledge Discovery",
                year="2014",
                month="Jan",
                day="01",
                volume="28",
                number="1",
                pages="92--122",
                abstract="The problem of modeling binary responses by using cross-sectional data has been addressed with a number of satisfying solutions that draw on both parametric and nonparametric methods. However, there exist many real situations where one of the two responses (usually the most interesting for the analysis) is rare. It has been largely reported that this class imbalance heavily compromises the process of learning, because the model tends to focus on the prevalent class and to ignore the rare events. However, not only the estimation of the classification model is affected by a skewed distribution of the classes, but also the evaluation of its accuracy is jeopardized, because the scarcity of data leads to poor estimates of the model's accuracy. In this work, the effects of class imbalance on model training and model assessing are discussed. Moreover, a unified and systematic framework for dealing with the problem of imbalanced classification is proposed, based on a smoothed bootstrap re-sampling technique. The proposed technique is founded on a sound theoretical basis and an extensive empirical study shows that it outperforms the main other remedies to face imbalanced learning problems.",
                issn="1573-756X",
                doi="10.1007/s10618-012-0295-5",
                url="https://doi.org/10.1007/s10618-012-0295-5"
                }

    * URL: https://drive.google.com/open?id=1eLOTCWtXCcqti9NpgXujw7BNY38ToyTc

Notes:
    * It is not entirely clear if the authors propose kernel density estimation         or the fitting of simple multivariate Gaussians on the minority samples.         The latter seems to be more likely, I implement that approach.
SMOTE_OUT
---------


API
^^^

.. autoclass:: smote_variants.SMOTE_OUT
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SMOTE_OUT()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SMOTE_OUT.png


References:
    * BibTex::
        
        @article{smote_out_smote_cosine_selected_smote,
                  title={SMOTE-Out, SMOTE-Cosine, and Selected-SMOTE: An enhancement strategy to handle imbalance in data level},
                  author={Fajri Koto},
                  journal={2014 International Conference on Advanced Computer Science and Information System},
                  year={2014},
                  pages={280-284}
                }

    * URL: https://drive.google.com/open?id=1XyyaphBLWJU_nQFppy1f1v5MYkSDqrJ2
SMOTE_Cosine
------------


API
^^^

.. autoclass:: smote_variants.SMOTE_Cosine
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SMOTE_Cosine()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SMOTE_Cosine.png


References:
    * BibTex::
        
        @article{smote_out_smote_cosine_selected_smote,
                  title={SMOTE-Out, SMOTE-Cosine, and Selected-SMOTE: An enhancement strategy to handle imbalance in data level},
                  author={Fajri Koto},
                  journal={2014 International Conference on Advanced Computer Science and Information System},
                  year={2014},
                  pages={280-284}
                }

    * URL: https://drive.google.com/open?id=1XyyaphBLWJU_nQFppy1f1v5MYkSDqrJ2
Selected_SMOTE
--------------


API
^^^

.. autoclass:: smote_variants.Selected_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.Selected_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/Selected_SMOTE.png


References:
    * BibTex::
        
    @article{smote_out_smote_cosine_selected_smote,
              title={SMOTE-Out, SMOTE-Cosine, and Selected-SMOTE: An enhancement strategy to handle imbalance in data level},
              author={Fajri Koto},
              journal={2014 International Conference on Advanced Computer Science and Information System},
              year={2014},
              pages={280-284}
            }

    * URL: https://drive.google.com/open?id=1XyyaphBLWJU_nQFppy1f1v5MYkSDqrJ2

Notes:
    * Significant attribute selection was not described in the paper, therefore we have implemented something meaningful.
LN_SMOTE
--------


API
^^^

.. autoclass:: smote_variants.LN_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.LN_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/LN_SMOTE.png


References:
    * BibTex::
        
        @INPROCEEDINGS{ln_smote, 
                        author={T. Maciejewski and J. Stefanowski}, 
                        booktitle={2011 IEEE Symposium on Computational Intelligence and Data Mining (CIDM)}, 
                        title={Local neighbourhood extension of SMOTE for mining imbalanced data}, 
                        year={2011}, 
                        volume={}, 
                        number={}, 
                        pages={104-111}, 
                        keywords={Bayes methods;data mining;pattern classification;local neighbourhood extension;imbalanced data mining;focused resampling technique;SMOTE over-sampling method;naive Bayes classifiers;Noise measurement;Noise;Decision trees;Breast cancer;Sensitivity;Data mining;Training}, 
                        doi={10.1109/CIDM.2011.5949434}, 
                        ISSN={}, 
                        month={April}}

    * URL: https://drive.google.com/open?id=1VXfwlXcfFrrL_DYa6lpgTLn-bCxxTKAM
MWMOTE
------


API
^^^

.. autoclass:: smote_variants.MWMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.MWMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/MWMOTE.png


References:
    * BibTex::
        
        @ARTICLE{mwmote, 
                    author={S. Barua and M. M. Islam and X. Yao and K. Murase}, 
                    journal={IEEE Transactions on Knowledge and Data Engineering}, 
                    title={MWMOTE--Majority Weighted Minority Oversampling Technique for Imbalanced Data Set Learning}, 
                    year={2014}, 
                    volume={26}, 
                    number={2}, 
                    pages={405-425}, 
                    keywords={learning (artificial intelligence);pattern clustering;sampling methods;AUC;area under curve;ROC;receiver operating curve;G-mean;geometric mean;minority class cluster;clustering approach;weighted informative minority class samples;Euclidean distance;hard-to-learn informative minority class samples;majority class;synthetic minority class samples;synthetic oversampling methods;imbalanced learning problems;imbalanced data set learning;MWMOTE-majority weighted minority oversampling technique;Sampling methods;Noise measurement;Boosting;Simulation;Complexity theory;Interpolation;Abstracts;Imbalanced learning;undersampling;oversampling;synthetic sample generation;clustering}, 
                    doi={10.1109/TKDE.2012.232}, 
                    ISSN={1041-4347}, 
                    month={Feb}}

    * URL: https://drive.google.com/open?id=1PiOSHhJJMZZniuiYvMPyD2BWmR9Q9L1q

Notes:
    * The original method was not prepared for the case of having clusters of 1 elements.
PDFOS
-----


API
^^^

.. autoclass:: smote_variants.PDFOS
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.PDFOS()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/PDFOS.png


References:
    * BibTex::
        
        @article{pdfos,
                title = "PDFOS: PDF estimation based over-sampling for imbalanced two-class problems",
                journal = "Neurocomputing",
                volume = "138",
                pages = "248 - 259",
                year = "2014",
                issn = "0925-2312",
                doi = "https://doi.org/10.1016/j.neucom.2014.02.006",
                url = "http://www.sciencedirect.com/science/article/pii/S0925231214002501",
                author = "Ming Gao and Xia Hong and Sheng Chen and Chris J. Harris and Emad Khalaf",
                keywords = "Imbalanced classification, Probability density function based over-sampling, Radial basis function classifier, Orthogonal forward selection, Particle swarm optimisation"
                }

    * URL: https://drive.google.com/open?id=1sBz9pFeHoGJ0XBwNwQ24r-gQA6-KhLSw

Notes:
    * Not prepared for low-rank data.
IPADE_ID
--------


API
^^^

.. autoclass:: smote_variants.IPADE_ID
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.IPADE_ID()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/IPADE_ID.png


References:
    * BibTex::
        
        @article{ipade_id,
                title = "Addressing imbalanced classification with instance generation techniques: IPADE-ID",
                journal = "Neurocomputing",
                volume = "126",
                pages = "15 - 28",
                year = "2014",
                note = "Recent trends in Intelligent Data Analysis Online Data Processing",
                issn = "0925-2312",
                doi = "https://doi.org/10.1016/j.neucom.2013.01.050",
                url = "http://www.sciencedirect.com/science/article/pii/S0925231213006887",
                author = "Victoria López and Isaac Triguero and Cristóbal J. Carmona and Salvador García and Francisco Herrera",
                keywords = "Differential evolution, Instance generation, Nearest neighbor, Decision tree, Imbalanced datasets"
                }

    * URL: https://drive.google.com/open?id=1G6MS_K0uBgIWlwWMyTciR8_6O7fwPTTg

Notes:
    * According to the algorithm, if the addition of a majority sample doesn't improve the AUC during the DE optimization process, the addition of no further majority points is tried.
    * In the differential evolution the multiplication by a random number seems have a deteriorating effect, new scaling parameter added to fix this.
    * It is not specified how to do the evaluation.
RWO_sampling
------------


API
^^^

.. autoclass:: smote_variants.RWO_sampling
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.RWO_sampling()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/RWO_sampling.png


References:
    * BibTex::
        
        @article{rwo_sampling,
                author = {Zhang, Huaxzhang and Li, Mingfang},
                year = {2014},
                month = {11},
                pages = {},
                title = {RWO-Sampling: A Random Walk Over-Sampling Approach to Imbalanced Data Classification},
                volume = {20},
                booktitle = {Information Fusion}
                }

    * URL: https://drive.google.com/open?id=1zewg606Wpm1yDyuTOagmAiFwmwFyTzQv
NEATER
------


API
^^^

.. autoclass:: smote_variants.NEATER
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.NEATER()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/NEATER.png


References:
    * BibTex::
        
        @INPROCEEDINGS{neater, 
                        author={B. A. Almogahed and I. A. Kakadiaris}, 
                        booktitle={2014 22nd International Conference on Pattern Recognition}, 
                        title={NEATER: Filtering of Over-sampled Data Using Non-cooperative Game Theory}, 
                        year={2014}, 
                        volume={}, 
                        number={}, 
                        pages={1371-1376}, 
                        keywords={data handling;game theory;information filtering;NEATER;imbalanced data problem;synthetic data;filtering of over-sampled data using non-cooperative game theory;Games;Game theory;Vectors;Sociology;Statistics;Silicon;Mathematical model}, 
                        doi={10.1109/ICPR.2014.245}, 
                        ISSN={1051-4651}, 
                        month={Aug}}

    * URL: https://drive.google.com/open?id=1GfMmurmyG-B5jfEVhyFrQAx8lA1JxbXl

Notes:
    * Evolving both majority and minority probabilities as nothing ensures that the probabilities remain in the range [0,1], and they need to be normalized.
    * The inversely weighted function needs to be cut at some value (like the alpha level), otherwise it will overemphasize the utility of having differing neighbors next to each other.
DEAGO
-----


API
^^^

.. autoclass:: smote_variants.DEAGO
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.DEAGO()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/DEAGO.png


References:
    * BibTex::
        
        @INPROCEEDINGS{deago, 
                        author={C. Bellinger and N. Japkowicz and C. Drummond}, 
                        booktitle={2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA)}, 
                        title={Synthetic Oversampling for Advanced Radioactive Threat Detection}, 
                        year={2015}, 
                        volume={}, 
                        number={}, 
                        pages={948-953}, 
                        keywords={radioactive waste;advanced radioactive threat detection;gamma-ray spectral classification;industrial nuclear facilities;Health Canadas national monitoring networks;Vancouver 2010;Isotopes;Training;Monitoring;Gamma-rays;Machine learning algorithms;Security;Neural networks;machine learning;classification;class imbalance;synthetic oversampling;artificial neural networks;autoencoders;gamma-ray spectra}, 
                        doi={10.1109/ICMLA.2015.58}, 
                        ISSN={}, 
                        month={Dec}}

    * URL: https://drive.google.com/open?id=1cnCltSny-X_Dl8s3BqcNCbqVn_eDPWlB

Notes:
    * There is no hint on the activation functions and amounts of noise.
Gazzah
------


API
^^^

.. autoclass:: smote_variants.Gazzah
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.Gazzah()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/Gazzah.png


References:
    * BibTex::
        
        @INPROCEEDINGS{gazzah, 
                        author={S. Gazzah and A. Hechkel and N. Essoukri Ben Amara}, 
                        booktitle={2015 IEEE 12th International Multi-Conference on Systems, Signals Devices (SSD15)}, 
                        title={A hybrid sampling method for imbalanced data}, 
                        year={2015}, 
                        volume={}, 
                        number={}, 
                        pages={1-6}, 
                        keywords={computer vision;image classification;learning (artificial intelligence);sampling methods;hybrid sampling method;imbalanced data;diversification;computer vision domain;classical machine learning systems;intraclass variations;system performances;classification accuracy;imbalanced training data;training data set;over-sampling;minority class;SMOTE star topology;feature vector deletion;intra-class variations;distribution criterion;biometric data;true positive rate;Training data;Principal component analysis;Databases;Support vector machines;Training;Feature extraction;Correlation;Imbalanced data sets;Intra-class variations;Data analysis;Principal component analysis;One-against-all SVM}, 
                        doi={10.1109/SSD.2015.7348093}, 
                        ISSN={}, 
                        month={March}}

    * URL: https://drive.google.com/open?id=1oh_FRi1e0NElX3wzWxTg0ecfqe1VWshN
MCT
---


API
^^^

.. autoclass:: smote_variants.MCT
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.MCT()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/MCT.png


References:
    * BibTex::
        
        @article{mct,
                author = {Jiang, Liangxiao and Qiu, Chen and Li, Chaoqun},
                year = {2015},
                month = {03},
                pages = {1551004},
                title = {A Novel Minority Cloning Technique for Cost-Sensitive Learning},
                volume = {29},
                booktitle = {International Journal of Pattern Recognition and Artificial Intelligence}
                }

    * URL: https://drive.google.com/open?id=1yyy-DmCFGWEPa8mdOpzHOx4l8xVVlCgA

Notes:
    * Mode is changed to median, distance is changed to Euclidean to support continuous features, and normalized.
ADG
---


API
^^^

.. autoclass:: smote_variants.ADG
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.ADG()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/ADG.png


References:
    * BibTex::
        
        @article{adg,
                author = {Pourhabib, A. and Mallick, Bani K. and Ding, Yu},
                year = {2015},
                month = {16},
                pages = {2695--2724},
                title = {A Novel Minority Cloning Technique for Cost-Sensitive Learning},
                volume = {16},
                journal = {Journal of Machine Learning Research}
                }

    * URL: https://drive.google.com/open?id=16QNZOJA77rS-AEmNlj-xH2uogAAhp3b3

Notes:
    * This method has a lot of parameters, it becomes fairly hard to cross-validate thoroughly.
    * Fails if matrix is singular when computing alpha_star, fixed by PCA.
    * Singularity might be caused by repeating samples.
    * Maintaining the kernel matrix becomes unfeasible above a couple of thousand vectors.
SMOTE_IPF
---------


API
^^^

.. autoclass:: smote_variants.SMOTE_IPF
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SMOTE_IPF()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SMOTE_IPF.png


References:
    * BibTex::
        
        @article{smote_ipf,
                    title = "SMOTE–IPF: Addressing the noisy and borderline examples problem in imbalanced classification by a re-sampling method with filtering",
                    journal = "Information Sciences",
                    volume = "291",
                    pages = "184 - 203",
                    year = "2015",
                    issn = "0020-0255",
                    doi = "https://doi.org/10.1016/j.ins.2014.08.051",
                    url = "http://www.sciencedirect.com/science/article/pii/S0020025514008561",
                    author = "José A. Sáez and Julián Luengo and Jerzy Stefanowski and Francisco Herrera",
                    keywords = "Imbalanced classification, Borderline examples, Noisy data, Noise filters, SMOTE"
                    }

    * URL: https://drive.google.com/open?id=1j2SKWvovYczOkg2MaMBep2i8XaZ0r_rS
KernelADASYN
------------


API
^^^

.. autoclass:: smote_variants.KernelADASYN
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.KernelADASYN()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/KernelADASYN.png


References:
    * BibTex::
        
        @INPROCEEDINGS{kernel_adasyn, 
                        author={B. Tang and H. He}, 
                        booktitle={2015 IEEE Congress on Evolutionary Computation (CEC)}, 
                        title={KernelADASYN: Kernel based adaptive synthetic data generation for imbalanced learning}, 
                        year={2015}, 
                        volume={}, 
                        number={}, 
                        pages={664-671}, 
                        keywords={learning (artificial intelligence);pattern classification;sampling methods;KernelADASYN;kernel based adaptive synthetic data generation;imbalanced learning;standard classification algorithms;data distribution;minority class decision rule;expensive minority class data misclassification;kernel based adaptive synthetic over-sampling approach;imbalanced data classification problems;kernel density estimation methods;Kernel;Estimation;Accuracy;Measurement;Standards;Training data;Sampling methods;Imbalanced learning;adaptive over-sampling;kernel density estimation;pattern recognition;medical and healthcare data learning}, 
                        doi={10.1109/CEC.2015.7256954}, 
                        ISSN={1089-778X}, 
                        month={May}}

    * URL: https://drive.google.com/open?id=1RXURKKH7BLOzC0N7J-btZBhBK9OCyB4K

Notes:
    * The method of sampling was not specified, Markov Chain Monte Carlo has been implemented.
    * Not prepared for improperly conditioned covariance matrix.
MOT2LD
------


API
^^^

.. autoclass:: smote_variants.MOT2LD
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.MOT2LD()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/MOT2LD.png


References:
    * BibTex::
        
        @InProceedings{mot2ld,
                        author="Xie, Zhipeng
                        and Jiang, Liyang
                        and Ye, Tengju
                        and Li, Xiaoli",
                        editor="Renz, Matthias
                        and Shahabi, Cyrus
                        and Zhou, Xiaofang
                        and Cheema, Muhammad Aamir",
                        title="A Synthetic Minority Oversampling Method Based on Local Densities in Low-Dimensional Space for Imbalanced Learning",
                        booktitle="Database Systems for Advanced Applications",
                        year="2015",
                        publisher="Springer International Publishing",
                        address="Cham",
                        pages="3--18",
                        abstract="Imbalanced class distribution is a challenging problem in many real-life classification problems. Existing synthetic oversampling do suffer from the curse of dimensionality because they rely heavily on Euclidean distance. This paper proposed a new method, called Minority Oversampling Technique based on Local Densities in Low-Dimensional Space (or MOT2LD in short). MOT2LD first maps each training sample into a low-dimensional space, and makes clustering of their low-dimensional representations. It then assigns weight to each minority sample as the product of two quantities: local minority density and local majority count, indicating its importance of sampling. The synthetic minority class samples are generated inside some minority cluster. MOT2LD has been evaluated on 15 real-world data sets. The experimental results have shown that our method outperforms some other existing methods including SMOTE, Borderline-SMOTE, ADASYN, and MWMOTE, in terms of G-mean and F-measure.",
                        isbn="978-3-319-18123-3"
                        }

    * URL: https://drive.google.com/open?id=191-gIFEmY1EmOT7iq0mK8fNr3btKovQ6

Notes:
    * Clusters might contain 1 elements, and all points can be filtered as noise.
    * Clusters might contain 0 elements as well, if all points are filtered as noise.
    * The entire clustering can become empty.
    * TSNE is very slow when the number of instances is over a couple of 1000
V_SYNTH
-------


API
^^^

.. autoclass:: smote_variants.V_SYNTH
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.V_SYNTH()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/V_SYNTH.png


References:
    * BibTex::
        
        @article{v_synth,
                 author = {Young,Ii, William A. and Nykl, Scott L. and Weckman, Gary R. and Chelberg, David M.},
                 title = {Using Voronoi Diagrams to Improve Classification Performances when Modeling Imbalanced Datasets},
                 journal = {Neural Comput. Appl.},
                 issue_date = {July      2015},
                 volume = {26},
                 number = {5},
                 month = jul,
                 year = {2015},
                 issn = {0941-0643},
                 pages = {1041--1054},
                 numpages = {14},
                 url = {http://dx.doi.org/10.1007/s00521-014-1780-0},
                 doi = {10.1007/s00521-014-1780-0},
                 acmid = {2790665},
                 publisher = {Springer-Verlag},
                 address = {London, UK, UK},
                 keywords = {Data engineering, Data mining, Imbalanced datasets, Knowledge extraction, Numerical algorithms, Synthetic over-sampling},
                }

    * URL: https://drive.google.com/open?id=1mbp816SazOpTOL22eMHfDkmJEKRaUKez

Notes:
    * The proposed encompassing bounding box generation is incorrect.
    * Voronoi diagram generation in high dimensional spaces is instable
OUPS
----


API
^^^

.. autoclass:: smote_variants.OUPS
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.OUPS()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/OUPS.png


References:
    * BibTex::
        
        @article{oups,
                    title = "A priori synthetic over-sampling methods for increasing classification sensitivity in imbalanced data sets",
                    journal = "Expert Systems with Applications",
                    volume = "66",
                    pages = "124 - 135",
                    year = "2016",
                    issn = "0957-4174",
                    doi = "https://doi.org/10.1016/j.eswa.2016.09.010",
                    url = "http://www.sciencedirect.com/science/article/pii/S0957417416304882",
                    author = "William A. Rivera and Petros Xanthopoulos",
                    keywords = "SMOTE, OUPS, Class imbalance, Classification"
                    }

    * URL: https://drive.google.com/open?id=1Q9X9Ye7F3igLrIV9GRqrAoyNmp592TGn

Notes:
    * In the description of the algorithm a fractional number p (j) is used to index a vector.
SMOTE_D
-------


API
^^^

.. autoclass:: smote_variants.SMOTE_D
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SMOTE_D()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SMOTE_D.png


References:
    * BibTex::
        
        @InProceedings{smote_d,
                        author="Torres, Fredy Rodr{'i}guez
                        and Carrasco-Ochoa, Jes{'u}s A.
                        and Mart{'i}nez-Trinidad, Jos{'e} Fco.",
                        editor="Mart{'i}nez-Trinidad, Jos{'e} Francisco
                        and Carrasco-Ochoa, Jes{'u}s Ariel
                        and Ayala Ramirez, Victor
                        and Olvera-L{'o}pez, Jos{'e} Arturo
                        and Jiang, Xiaoyi",
                        title="SMOTE-D a Deterministic Version of SMOTE",
                        booktitle="Pattern Recognition",
                        year="2016",
                        publisher="Springer International Publishing",
                        address="Cham",
                        pages="177--188",
                        abstract="Imbalanced data is a problem of current research interest. This problem arises when the number of objects in a class is much lower than in other classes. In order to address this problem several methods for oversampling the minority class have been proposed. Oversampling methods generate synthetic objects for the minority class in order to balance the amount of objects between classes, among them, SMOTE is one of the most successful and well-known methods. In this paper, we introduce a modification of SMOTE which deterministically generates synthetic objects for the minority class. Our proposed method eliminates the random component of SMOTE and generates different amount of synthetic objects for each object of the minority class. An experimental comparison of the proposed method against SMOTE in standard imbalanced datasets is provided. The experimental results show an improvement of our proposed method regarding SMOTE, in terms of F-measure.",
                        isbn="978-3-319-39393-3"
                        }

    * URL: https://drive.google.com/open?id=1x_9IYnDvVBXeYjBcwgV9FOlL7pj-Yi2f

Notes:
    * Copying happens if two points are the neighbors of each other.
SMOTE_PSO
---------


API
^^^

.. autoclass:: smote_variants.SMOTE_PSO
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SMOTE_PSO()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SMOTE_PSO.png


References:
    * BibTex::
        
        @article{smote_pso,
                    title = "PSO-based method for SVM classification on skewed data sets",
                    journal = "Neurocomputing",
                    volume = "228",
                    pages = "187 - 197",
                    year = "2017",
                    note = "Advanced Intelligent Computing: Theory and Applications",
                    issn = "0925-2312",
                    doi = "https://doi.org/10.1016/j.neucom.2016.10.041",
                    url = "http://www.sciencedirect.com/science/article/pii/S0925231216312668",
                    author = "Jair Cervantes and Farid Garcia-Lamont and Lisbeth Rodriguez and Asdrúbal López and José Ruiz Castilla and Adrian Trueba",
                    keywords = "Skew data sets, SVM, Hybrid algorithms"
                    }

    * URL: https://drive.google.com/open?id=1rJu-2aLrosz_NGlcoRdz3qyCGlExQsvZ

Notes:
    * I find the description of the technique a bit confusing, especially on the bounds of the search space of velocities and positions. Equations 15 and 16 specify the lower and upper bounds, the lower bound is in fact a vector while the upper bound is a distance. I tried to implement something meaningful.
    * I also find the setting of accelerating constant 2.0 strange, most of the time the velocity will be bounded due to this choice. 
    * Also, training and predicting probabilities with a non-linear SVM as the evaluation function becomes fairly expensive when the number of training vectors reaches a couple of thousands. To reduce computational burden, minority and majority vectors far from the other class are removed to reduce the size of both classes to a maximum of 500 samples. Generally, this shouldn't really affect the results as the technique focuses on the samples near the class boundaries.
CURE_SMOTE
----------


API
^^^

.. autoclass:: smote_variants.CURE_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.CURE_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/CURE_SMOTE.png


References:
    * BibTex::
        
        @Article{cure_smote,
                    author="Ma, Li
                    and Fan, Suohai",
                    title="CURE-SMOTE algorithm and hybrid algorithm for feature selection and parameter optimization based on random forests",
                    journal="BMC Bioinformatics",
                    year="2017",
                    month="Mar",
                    day="14",
                    volume="18",
                    number="1",
                    pages="169",
                    abstract="The random forests algorithm is a type of classifier with prominent universality, a wide application range, and robustness for avoiding overfitting. But there are still some drawbacks to random forests. Therefore, to improve the performance of random forests, this paper seeks to improve imbalanced data processing, feature selection and parameter optimization.",
                    issn="1471-2105",
                    doi="10.1186/s12859-017-1578-z",
                    url="https://doi.org/10.1186/s12859-017-1578-z"
                    }

    * URL: https://drive.google.com/open?id=1XJua_4oAcffDxt_seCu-eQHhfG9ig1Og

Notes:
    * It is not specified how to determine the cluster with the "slowest growth rate"
    * All clusters can be removed as noise.
SOMO
----


API
^^^

.. autoclass:: smote_variants.SOMO
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SOMO()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SOMO.png


References:
    * BibTex::
        
        @article{somo,
                    title = "Self-Organizing Map Oversampling (SOMO) for imbalanced data set learning",
                    journal = "Expert Systems with Applications",
                    volume = "82",
                    pages = "40 - 52",
                    year = "2017",
                    issn = "0957-4174",
                    doi = "https://doi.org/10.1016/j.eswa.2017.03.073",
                    url = "http://www.sciencedirect.com/science/article/pii/S0957417417302324",
                    author = "Georgios Douzas and Fernando Bacao"
                    }

    * URL: https://drive.google.com/open?id=1RiPlh4KQ383YTr04-voi3Vq2iBBE-1Ij

Notes:
    * It is not specified how to handle those cases when a cluster contains 1 minority samples, the mean of within-cluster distances is set to 100 in these cases.
ISOMAP_Hybrid
-------------


API
^^^

.. autoclass:: smote_variants.ISOMAP_Hybrid
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.ISOMAP_Hybrid()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/ISOMAP_Hybrid.png


References:
    * BibTex::
        
        @inproceedings{isomap_hybrid,
                         author = {Gu, Qiong and Cai, Zhihua and Zhu, Li},
                         title = {Classification of Imbalanced Data Sets by Using the Hybrid Re-sampling Algorithm Based on Isomap},
                         booktitle = {Proceedings of the 4th International Symposium on Advances in Computation and Intelligence},
                         series = {ISICA '09},
                         year = {2009},
                         isbn = {978-3-642-04842-5},
                         location = {Huangshi, China},
                         pages = {287--296},
                         numpages = {10},
                         url = {http://dx.doi.org/10.1007/978-3-642-04843-2_31},
                         doi = {10.1007/978-3-642-04843-2_31},
                         acmid = {1691478},
                         publisher = {Springer-Verlag},
                         address = {Berlin, Heidelberg},
                         keywords = {Imbalanced data set, Isomap, NCR, Smote, re-sampling},
                        } 

    * URL: https://drive.google.com/open?id=1_j8kYoKt8mFxr8Y_ceNVPlejfTXk6-5w
CE_SMOTE
--------


API
^^^

.. autoclass:: smote_variants.CE_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.CE_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/CE_SMOTE.png


References:
    * BibTex::
        
        @INPROCEEDINGS{ce_smote, 
                            author={S. Chen and G. Guo and L. Chen}, 
                            booktitle={2010 IEEE 24th International Conference on Advanced Information Networking and Applications Workshops}, 
                            title={A New Over-Sampling Method Based on Cluster Ensembles}, 
                            year={2010}, 
                            volume={}, 
                            number={}, 
                            pages={599-604}, 
                            keywords={data mining;Internet;pattern classification;pattern clustering;over sampling method;cluster ensembles;classification method;imbalanced data handling;CE-SMOTE;clustering consistency index;cluster boundary minority samples;imbalanced public data set;Mathematics;Computer science;Electronic mail;Accuracy;Nearest neighbor searches;Application software;Data mining;Conferences;Web sites;Information retrieval;classification;imbalanced data sets;cluster ensembles;over-sampling}, 
                            doi={10.1109/WAINA.2010.40}, 
                            ISSN={}, 
                            month={April}}

    * URL: https://drive.google.com/open?id=1erU3PsoePzxFCyv8aVwNJ2LO1hHX6dTz
Edge_Det_SMOTE
--------------


API
^^^

.. autoclass:: smote_variants.Edge_Det_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.Edge_Det_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/Edge_Det_SMOTE.png


References:
    * BibTex::
        
        @INPROCEEDINGS{Edge_Det_SMOTE, 
                        author={Y. Kang and S. Won}, 
                        booktitle={ICCAS 2010}, 
                        title={Weight decision algorithm for oversampling technique on class-imbalanced learning}, 
                        year={2010}, 
                        volume={}, 
                        number={}, 
                        pages={182-186}, 
                        keywords={edge detection;learning (artificial intelligence);weight decision algorithm;oversampling technique;class-imbalanced learning;class imbalanced data problem;edge detection algorithm;spatial space representation;Classification algorithms;Image edge detection;Training;Noise measurement;Glass;Training data;Machine learning;Imbalanced learning;Classification;Weight decision;Oversampling;Edge detection}, 
                        doi={10.1109/ICCAS.2010.5669889}, 
                        ISSN={}, 
                        month={Oct}}

    * URL: https://drive.google.com/open?id=11eSqSkAzhVTeutlLNqWNo2g3ZYIZbEdM

Notes:
    * This technique is very loosely specified.
CBSO
----


API
^^^

.. autoclass:: smote_variants.CBSO
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.CBSO()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/CBSO.png


References:
    * BibTex::
        
        @InProceedings{cbso,
                        author="Barua, Sukarna
                        and Islam, Md. Monirul
                        and Murase, Kazuyuki",
                        editor="Lu, Bao-Liang
                        and Zhang, Liqing
                        and Kwok, James",
                        title="A Novel Synthetic Minority Oversampling Technique for Imbalanced Data Set Learning",
                        booktitle="Neural Information Processing",
                        year="2011",
                        publisher="Springer Berlin Heidelberg",
                        address="Berlin, Heidelberg",
                        pages="735--744",
                        abstract="Imbalanced data sets contain an unequal distribution of data samples among the classes and pose a challenge to the learning algorithms as it becomes hard to learn the minority class concepts. Synthetic oversampling techniques address this problem by creating synthetic minority samples to balance the data set. However, most of these techniques may create wrong synthetic minority samples which fall inside majority regions. In this respect, this paper presents a novel Cluster Based Synthetic Oversampling (CBSO) algorithm. CBSO adopts its basic idea from existing synthetic oversampling techniques and incorporates unsupervised clustering in its synthetic data generation mechanism. CBSO ensures that synthetic samples created via this method always lie inside minority regions and thus, avoids any wrong synthetic sample creation. Simualtion analyses on some real world datasets show the effectiveness of CBSO showing improvements in various assesment metrics such as overall accuracy, F-measure, and G-mean.",
                        isbn="978-3-642-24958-7"
                        }

    * URL: https://drive.google.com/open?id=16OYKeBf5UPeHJgXCD7-WZtqdc2Am4kBQ

Notes:
    * Clusters containing 1 element induce cloning of samples.
E_SMOTE
-------


API
^^^

.. autoclass:: smote_variants.E_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.E_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/E_SMOTE.png


References:
    * BibTex::
        
        @INPROCEEDINGS{e_smote, 
                        author={T. Deepa and M. Punithavalli}, 
                        booktitle={2011 3rd International Conference on Electronics Computer Technology}, 
                        title={An E-SMOTE technique for feature selection in High-Dimensional Imbalanced Dataset}, 
                        year={2011}, 
                        volume={2}, 
                        number={}, 
                        pages={322-324}, 
                        keywords={bioinformatics;data mining;pattern classification;support vector machines;E-SMOTE technique;feature selection;high-dimensional imbalanced dataset;data mining;bio-informatics;dataset balancing;SVM classification;micro array dataset;Feature extraction;Genetic algorithms;Support vector machines;Data mining;Machine learning;Bioinformatics;Cancer;Imbalanced dataset;Featue Selection;E-SMOTE;Support Vector Machine[SVM]}, 
                        doi={10.1109/ICECTECH.2011.5941710}, 
                        ISSN={}, 
                        month={April}}

    * URL: https://drive.google.com/open?id=1P-4XvnbNuA6OzdeaYBeQEBcUnCgV1R3M

Notes:
    * This technique is basically unreproducible. I try to implement something following the idea of applying some simple genetic algorithm for optimization.
    * In my best understanding, the technique uses evolutionary algorithms to for feature selection and then applies vanilla SMOTE on the selected features only.
DBSMOTE
-------


API
^^^

.. autoclass:: smote_variants.DBSMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.DBSMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/DBSMOTE.png


References:
    * BibTex::
        
        @Article{dbsmote,
                    author="Bunkhumpornpat, Chumphol
                    and Sinapiromsaran, Krung
                    and Lursinsap, Chidchanok",
                    title="DBSMOTE: Density-Based Synthetic Minority Over-sampling TEchnique",
                    journal="Applied Intelligence",
                    year="2012",
                    month="Apr",
                    day="01",
                    volume="36",
                    number="3",
                    pages="664--684",
                    abstract="A dataset exhibits the class imbalance problem when a target class has a very small number of instances relative to other classes. A trivial classifier typically fails to detect a minority class due to its extremely low incidence rate. In this paper, a new over-sampling technique called DBSMOTE is proposed. Our technique relies on a density-based notion of clusters and is designed to over-sample an arbitrarily shaped cluster discovered by DBSCAN. DBSMOTE generates synthetic instances along a shortest path from each positive instance to a pseudo-centroid of a minority-class cluster. Consequently, these synthetic instances are dense near this centroid and are sparse far from this centroid. Our experimental results show that DBSMOTE improves precision, F-value, and AUC more effectively than SMOTE, Borderline-SMOTE, and Safe-Level-SMOTE for imbalanced datasets.",
                    issn="1573-7497",
                    doi="10.1007/s10489-011-0287-y",
                    url="https://doi.org/10.1007/s10489-011-0287-y"
                    }

    * URL: https://drive.google.com/open?id=1FczQWnv7ZveAuLME1flnQw9ogEAcYQ5a

Notes:
    * Standardization is needed to use absolute eps values.
    * The clustering is likely to identify all instances as noise, fixed by recursive call with increaseing eps.
ASMOBD
------


API
^^^

.. autoclass:: smote_variants.ASMOBD
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.ASMOBD()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/ASMOBD.png


References:
    * BibTex::
        
        @INPROCEEDINGS{asmobd, 
                        author={Senzhang Wang and Zhoujun Li and Wenhan Chao and Qinghua Cao}, 
                        booktitle={The 2012 International Joint Conference on Neural Networks (IJCNN)}, 
                        title={Applying adaptive over-sampling technique based on data density and cost-sensitive SVM to imbalanced learning}, 
                        year={2012}, 
                        volume={}, 
                        number={}, 
                        pages={1-8}, 
                        keywords={data analysis;learning (artificial intelligence);sampling methods;smoothing methods;support vector machines;adaptive over-sampling technique;cost-sensitive SVM;imbalanced learning;resampling method;data density information;overfitting;minority sample;learning difficulty;decision region;over generalization;smoothing method;cost-sensitive learning;UCI dataset;G-mean of;receiver operation curve;Smoothing methods;Noise;Support vector machines;Classification algorithms;Interpolation;Measurement;Algorithm design and analysis;over-sampling;Cost-sensitive SVM;imbalanced learning}, 
                        doi={10.1109/IJCNN.2012.6252696}, 
                        ISSN={2161-4407}, 
                        month={June}}

    * URL: https://drive.google.com/open?id=1rF4H2L5W4Y1myX2K3TbKYj1IuwclOWsW

Notes:
    * In order to use absolute thresholds, the data is standardized.
    * The technique has many parameters, not easy to find the right combination.
Assembled_SMOTE
---------------


API
^^^

.. autoclass:: smote_variants.Assembled_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.Assembled_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/Assembled_SMOTE.png


References:
    * BibTex::
        
        @INPROCEEDINGS{assembled_smote, 
                        author={B. Zhou and C. Yang and H. Guo and J. Hu}, 
                        booktitle={The 2013 International Joint Conference on Neural Networks (IJCNN)}, 
                        title={A quasi-linear SVM combined with assembled SMOTE for imbalanced data classification}, 
                        year={2013}, 
                        volume={}, 
                        number={}, 
                        pages={1-7}, 
                        keywords={approximation theory;interpolation;pattern classification;sampling methods;support vector machines;trees (mathematics);quasilinear SVM;assembled SMOTE;imbalanced dataset classification problem;oversampling method;quasilinear kernel function;approximate nonlinear separation boundary;mulitlocal linear boundaries;interpolation;data distribution information;minimal spanning tree;local linear partitioning method;linear separation boundary;synthetic minority class samples;oversampled dataset classification;standard SVM;composite quasilinear kernel function;artificial data datasets;benchmark datasets;classification performance improvement;synthetic minority over-sampling technique;Support vector machines;Kernel;Merging;Standards;Sociology;Statistics;Interpolation}, 
                        doi={10.1109/IJCNN.2013.6707035}, 
                        ISSN={2161-4407}, 
                        month={Aug}}

    * URL: https://drive.google.com/open?id=1r3odAQ9aMPvy373wUFbdfr2r8uV0KLHO

Notes:
    * Absolute value of the angles extracted should be taken. (implemented this way)
    * It is not specified how many samples are generated in the various clusters.
SDSMOTE
-------


API
^^^

.. autoclass:: smote_variants.SDSMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SDSMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SDSMOTE.png


References:
    * BibTex::
        
        @INPROCEEDINGS{sdsmote, 
                        author={K. Li and W. Zhang and Q. Lu and X. Fang}, 
                        booktitle={2014 International Conference on Identification, Information and Knowledge in the Internet of Things}, 
                        title={An Improved SMOTE Imbalanced Data Classification Method Based on Support Degree}, 
                        year={2014}, 
                        volume={}, 
                        number={}, 
                        pages={34-38}, 
                        keywords={data mining;pattern classification;sampling methods;improved SMOTE imbalanced data classification method;support degree;data mining;class distribution;imbalanced data-set classification;over sampling method;minority class sample generation;minority class sample selection;minority class boundary sample identification;Classification algorithms;Training;Bagging;Computers;Testing;Algorithm design and analysis;Data mining;Imbalanced data-sets;Classification;Boundary sample;Support degree;SMOTE}, 
                        doi={10.1109/IIKI.2014.14}, 
                        ISSN={}, 
                        month={Oct}}

    * URL: https://drive.google.com/open?id=1jq20pUZJliHkkndyGYjL_A70kF3nKcNU
DSMOTE
------


API
^^^

.. autoclass:: smote_variants.DSMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.DSMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/DSMOTE.png


References:
    * BibTex::
        
        @INPROCEEDINGS{dsmote, 
                        author={S. Mahmoudi and P. Moradi and F. Akhlaghian and R. Moradi}, 
                        booktitle={2014 4th International Conference on Computer and Knowledge Engineering (ICCKE)}, 
                        title={Diversity and separable metrics in over-sampling technique for imbalanced data classification}, 
                        year={2014}, 
                        volume={}, 
                        number={}, 
                        pages={152-158}, 
                        keywords={learning (artificial intelligence);pattern classification;sampling methods;diversity metric;separable metric;over-sampling technique;imbalanced data classification;class distribution techniques;under-sampling technique;DSMOTE method;imbalanced learning problem;diversity measure;separable measure;Iran University of Medical Science;UCI dataset;Accuracy;Classification algorithms;Vectors;Educational institutions;Euclidean distance;Data mining;Diversity measure;Separable Measure;Over-Sampling;Imbalanced Data;Classification problems}, 
                        doi={10.1109/ICCKE.2014.6993409}, 
                        ISSN={}, 
                        month={Oct}}

    * URL: https://drive.google.com/open?id=1l2rhdGRICI-ttTlMAPYDK_SGPamSR1Fo

Notes:
    * The method is highly inefficient when the number of minority samples is high, time complexity is O(n^3), with 1000 minority samples it takes about 1e9 objective function evaluations to find 1 new sample points. Adding 1000 samples would take about 1e12 evaluations of the objective function, which is unfeasible. We introduce a new parameter, n_step, and during the search for the new sample at most n_step combinations of minority samples are tried.
    * Abnormality of minority points is defined in the paper as D_maj/D_min, high abnormality  means that the minority point is close to other minority points and very far from majority points. This is definitely not abnormality, I have implemented the opposite. 
    * Nothing ensures that the fisher statistics and the variance from the geometric mean remain comparable, which might skew the optimization towards one of the sub-objectives.
    * MinMax normalization doesn't work, each attribute will have a 0 value, which will make the geometric mean of all attribute 0.
G_SMOTE
-------


API
^^^

.. autoclass:: smote_variants.G_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.G_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/G_SMOTE.png


References:
    * BibTex::
        
        @INPROCEEDINGS{g_smote, 
                        author={T. Sandhan and J. Y. Choi}, 
                        booktitle={2014 22nd International Conference on Pattern Recognition}, 
                        title={Handling Imbalanced Datasets by Partially Guided Hybrid Sampling for Pattern Recognition}, 
                        year={2014}, 
                        volume={}, 
                        number={}, 
                        pages={1449-1453}, 
                        keywords={Gaussian processes;learning (artificial intelligence);pattern classification;regression analysis;sampling methods;support vector machines;imbalanced datasets;partially guided hybrid sampling;pattern recognition;real-world domains;skewed datasets;dataset rebalancing;learning algorithm;extremely low minority class samples;classification tasks;extracted hidden patterns;support vector machine;logistic regression;nearest neighbor;Gaussian process classifier;Support vector machines;Proteins;Pattern recognition;Kernel;Databases;Gaussian processes;Vectors;Imbalanced dataset;protein classification;ensemble classifier;bootstrapping;Sat-image classification;medical diagnoses}, 
                        doi={10.1109/ICPR.2014.258}, 
                        ISSN={1051-4651}, 
                        month={Aug}}

    * URL: https://drive.google.com/open?id=1GJ67qd2r0RH3MMJV5XqVhUBDVfnz4ErF

Notes:
    * the non-linear approach is inefficient 
NT_SMOTE
--------


API
^^^

.. autoclass:: smote_variants.NT_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.NT_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/NT_SMOTE.png


References:
    * BibTex::
        
        @INPROCEEDINGS{nt_smote, 
                        author={Y. H. Xu and H. Li and L. P. Le and X. Y. Tian}, 
                        booktitle={2014 Seventh International Joint Conference on Computational Sciences and Optimization}, 
                        title={Neighborhood Triangular Synthetic Minority Over-sampling Technique for Imbalanced Prediction on Small Samples of Chinese Tourism and Hospitality Firms}, 
                        year={2014}, 
                        volume={}, 
                        number={}, 
                        pages={534-538}, 
                        keywords={financial management;pattern classification;risk management;sampling methods;travel industry;Chinese tourism;hospitality firms;imbalanced risk prediction;minority class samples;up-sampling approach;neighborhood triangular synthetic minority over-sampling technique;NT-SMOTE;nearest neighbor idea;triangular area sampling idea;single classifiers;data excavation principles;hospitality industry;missing financial indicators;financial data filtering;financial risk prediction;MDA;DT;LSVM;logit;probit;firm risk prediction;Joints;Optimization;imbalanced datasets;NT-SMOTE;neighborhood triangular;random sampling}, 
                        doi={10.1109/CSO.2014.104}, 
                        ISSN={}, 
                        month={July}}

    * URL: https://drive.google.com/open?id=1iMeem5Ax2AkvwatvMpf1ZGMfHSsI3vQi
Lee
---


API
^^^

.. autoclass:: smote_variants.Lee
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.Lee()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/Lee.png


References:
    * BibTex::
        
        @inproceedings{lee,
                         author = {Lee, Jaedong and Kim, Noo-ri and Lee, Jee-Hyong},
                         title = {An Over-sampling Technique with Rejection for Imbalanced Class Learning},
                         booktitle = {Proceedings of the 9th International Conference on Ubiquitous Information Management and Communication},
                         series = {IMCOM '15},
                         year = {2015},
                         isbn = {978-1-4503-3377-1},
                         location = {Bali, Indonesia},
                         pages = {102:1--102:6},
                         articleno = {102},
                         numpages = {6},
                         url = {http://doi.acm.org/10.1145/2701126.2701181},
                         doi = {10.1145/2701126.2701181},
                         acmid = {2701181},
                         publisher = {ACM},
                         address = {New York, NY, USA},
                         keywords = {data distribution, data preprocessing, imbalanced problem, rejection rule, synthetic minority oversampling technique}
                        } 

    * URL: https://drive.google.com/open?id=1omttVQFQ8oDZHeZ87bUSa5Hr7fqt2Vwf
SPY
---


API
^^^

.. autoclass:: smote_variants.SPY
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SPY()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SPY.png


References:
    * BibTex::
        
        @INPROCEEDINGS{spy, 
                        author={X. T. Dang and D. H. Tran and O. Hirose and K. Satou}, 
                        booktitle={2015 Seventh International Conference on Knowledge and Systems Engineering (KSE)}, 
                        title={SPY: A Novel Resampling Method for Improving Classification Performance in Imbalanced Data}, 
                        year={2015}, 
                        volume={}, 
                        number={}, 
                        pages={280-285}, 
                        keywords={decision making;learning (artificial intelligence);pattern classification;sampling methods;SPY;resampling method;decision-making process;biomedical data classification;class imbalance learning method;SMOTE;oversampling method;UCI machine learning repository;G-mean value;borderline-SMOTE;safe-level-SMOTE;Support vector machines;Training;Bioinformatics;Proteins;Protein engineering;Radio frequency;Sensitivity;Imbalanced dataset;Over-sampling;Under-sampling;SMOTE;borderline-SMOTE}, 
                        doi={10.1109/KSE.2015.24}, 
                        ISSN={}, 
                        month={Oct}}

    * URL: https://drive.google.com/open?id=1B3qUj6lPdO21EjxuVKLHi8OV68gLZUUA
SMOTE_PSOBAT
------------


API
^^^

.. autoclass:: smote_variants.SMOTE_PSOBAT
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SMOTE_PSOBAT()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SMOTE_PSOBAT.png


References:
    * BibTex::
        
        @INPROCEEDINGS{smote_psobat, 
                        author={J. Li and S. Fong and Y. Zhuang}, 
                        booktitle={2015 3rd International Symposium on Computational and Business Intelligence (ISCBI)}, 
                        title={Optimizing SMOTE by Metaheuristics with Neural Network and Decision Tree}, 
                        year={2015}, 
                        volume={}, 
                        number={}, 
                        pages={26-32}, 
                        keywords={data mining;particle swarm optimisation;pattern classification;data mining;classifier;metaherustics;SMOTE parameters;performance indicators;selection optimization;PSO;particle swarm optimization algorithm;BAT;bat-inspired algorithm;metaheuristic optimization algorithms;nearest neighbors;imbalanced dataset problem;synthetic minority over-sampling technique;decision tree;neural network;Classification algorithms;Neural networks;Decision trees;Training;Optimization;Particle swarm optimization;Data mining;SMOTE;Swarm Intelligence;parameter selection optimization}, 
                        doi={10.1109/ISCBI.2015.12}, 
                        ISSN={}, 
                        month={Dec}}

    * URL: https://drive.google.com/open?id=1PQfIJRpKkNVcwQixzJxPN-K1FKEr_oDc

Notes:
    * The parameters of the memetic algorithms are not specified.
    * I have checked multiple paper describing the BAT algorithm, but the meaning of "Generate a new solution by flying randomly" is still unclear. 
    * It is also unclear if best solutions are recorded for each bat, or the entire population.
MDO
---


API
^^^

.. autoclass:: smote_variants.MDO
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.MDO()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/MDO.png


References:
    * BibTex::
        
        @ARTICLE{mdo, 
                    author={L. Abdi and S. Hashemi}, 
                    journal={IEEE Transactions on Knowledge and Data Engineering}, 
                    title={To Combat Multi-Class Imbalanced Problems by Means of Over-Sampling Techniques}, 
                    year={2016}, 
                    volume={28}, 
                    number={1}, 
                    pages={238-251}, 
                    keywords={covariance analysis;learning (artificial intelligence);modelling;pattern classification;sampling methods;statistical distributions;minority class instance modelling;probability contour;covariance structure;MDO;Mahalanobis distance-based oversampling technique;data-oriented technique;model-oriented solution;machine learning algorithm;data skewness;multiclass imbalanced problem;Mathematical model;Training;Accuracy;Eigenvalues and eigenfunctions;Machine learning algorithms;Algorithm design and analysis;Benchmark testing;Multi-class imbalance problems;over-sampling techniques;Mahalanobis distance;Multi-class imbalance problems;over-sampling techniques;Mahalanobis distance}, 
                    doi={10.1109/TKDE.2015.2458858}, 
                    ISSN={1041-4347}, 
                    month={Jan}}

    * URL: https://drive.google.com/open?id=1O_X4rhJcMx5h4eION2WJGHTqLyxoGO9i
Random_SMOTE
------------


API
^^^

.. autoclass:: smote_variants.Random_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.Random_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/Random_SMOTE.png


References:
    * BibTex::
        
        @InProceedings{random_smote,
                        author="Dong, Yanjie
                        and Wang, Xuehua",
                        editor="Xiong, Hui
                        and Lee, W. B.",
                        title="A New Over-Sampling Approach: Random-SMOTE for Learning from Imbalanced Data Sets",
                        booktitle="Knowledge Science, Engineering and Management",
                        year="2011",
                        publisher="Springer Berlin Heidelberg",
                        address="Berlin, Heidelberg",
                        pages="343--352",
                        abstract="For imbalanced data sets, examples of minority class are sparsely distributed in sample space compared with the overwhelming amount of majority class. This presents a great challenge for learning from the minority class. Enlightened by SMOTE, a new over-sampling method, Random-SMOTE, which generates examples randomly in the sample space of minority class is proposed. According to the experiments on real data sets, Random-SMOTE is more effective compared with other random sampling approaches.",
                        isbn="978-3-642-25975-3"
                        }

    * URL: https://drive.google.com/open?id=1_Wd2KaqlIcSmnjvlksYBgu5PsWIhDhbY
ISMOTE
------


API
^^^

.. autoclass:: smote_variants.ISMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.ISMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/ISMOTE.png


References:
    * BibTex::
        
        @InProceedings{ismote,
                        author="Li, Hu
                        and Zou, Peng
                        and Wang, Xiang
                        and Xia, Rongze",
                        editor="Sun, Zengqi
                        and Deng, Zhidong",
                        title="A New Combination Sampling Method for Imbalanced Data",
                        booktitle="Proceedings of 2013 Chinese Intelligent Automation Conference",
                        year="2013",
                        publisher="Springer Berlin Heidelberg",
                        address="Berlin, Heidelberg",
                        pages="547--554",
                        abstract="Imbalanced data is commonly in the real world and brings a lot of challenges. In this paper, we propose a combination sampling method which resamples both minority class and majority class. Improved SMOTE (ISMOTE) is used to do over-sampling on minority class, while distance-based under-sampling (DUS) method is used to do under-sampling on majority class. We adjust the sampling times to search for the optimal results while maintain the dataset size unchanged. Experiments on UCI datasets show that the proposed method performs better than using single over-sampling or under-sampling method.",
                        isbn="978-3-642-38466-0"
                        }

    * URL: https://drive.google.com/open?id=1z5J2-eDZBOobFvYH4jsXmT8-b3eJ0f6W
VIS_RST
-------


API
^^^

.. autoclass:: smote_variants.VIS_RST
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.VIS_RST()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/VIS_RST.png


References:
    * BibTex::
        
        @InProceedings{vis_rst,
                        author="Borowska, Katarzyna
                        and Stepaniuk, Jaros{\l}aw",
                        editor="Saeed, Khalid
                        and Homenda, W{\l}adys{\l}aw",
                        title="Imbalanced Data Classification: A Novel Re-sampling Approach Combining Versatile Improved SMOTE and Rough Sets",
                        booktitle="Computer Information Systems and Industrial Management",
                        year="2016",
                        publisher="Springer International Publishing",
                        address="Cham",
                        pages="31--42",
                        abstract="In recent years, the problem of learning from imbalanced data has emerged as important and challenging. The fact that one of the classes is underrepresented in the data set is not the only reason of difficulties. The complex distribution of data, especially small disjuncts, noise and class overlapping, contributes to the significant depletion of classifier's performance. Hence, the numerous solutions were proposed. They are categorized into three groups: data-level techniques, algorithm-level methods and cost-sensitive approaches. This paper presents a novel data-level method combining Versatile Improved SMOTE and rough sets. The algorithm was applied to the two-class problems, data sets were characterized by the nominal attributes. We evaluated the proposed technique in comparison with other preprocessing methods. The impact of the additional cleaning phase was specifically verified.",
                        isbn="978-3-319-45378-1"
                        }

    * URL: https://drive.google.com/open?id=1mTca65RRZ39SLNOy4hxvh23qlNha8kpj

Notes:
    * Replication of DANGER samples will be removed by the last step of noise filtering.
GASMOTE
-------


API
^^^

.. autoclass:: smote_variants.GASMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.GASMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/GASMOTE.png


References:
    * BibTex::
        
        @Article{gasmote,
                    author="Jiang, Kun
                    and Lu, Jing
                    and Xia, Kuiliang",
                    title="A Novel Algorithm for Imbalance Data Classification Based on Genetic Algorithm Improved SMOTE",
                    journal="Arabian Journal for Science and Engineering",
                    year="2016",
                    month="Aug",
                    day="01",
                    volume="41",
                    number="8",
                    pages="3255--3266",
                    abstract="The classification of imbalanced data has been recognized as a crucial problem in machine learning and data mining. In an imbalanced dataset, there are significantly fewer training instances of one class compared to another class. Hence, the minority class instances are much more likely to be misclassified. In the literature, the synthetic minority over-sampling technique (SMOTE) has been developed to deal with the classification of imbalanced datasets. It synthesizes new samples of the minority class to balance the dataset, by re-sampling the instances of the minority class. Nevertheless, the existing algorithms-based SMOTE uses the same sampling rate for all instances of the minority class. This results in sub-optimal performance. To address this issue, we propose a novel genetic algorithm-based SMOTE (GASMOTE) algorithm. The GASMOTE algorithm uses different sampling rates for different minority class instances and finds the combination of optimal sampling rates. The experimental results on ten typical imbalance datasets show that, compared with SMOTE algorithm, GASMOTE can increase 5.9{\%} on F-measure value and 1.6{\%} on G-mean value, and compared with Borderline-SMOTE algorithm, GASMOTE can increase 3.7{\%} on F-measure value and 2.3{\%} on G-mean value. GASMOTE can be used as a new over-sampling technique to deal with imbalance dataset classification problem. We have particularly applied the GASMOTE algorithm to a practical engineering application: prediction of rockburst in the VCR rockburst datasets. The experiment results indicate that the GASMOTE algorithm can accurately predict the rockburst occurrence and hence provides guidance to the design and construction of safe deep mining engineering structures.",
                    issn="2191-4281",
                    doi="10.1007/s13369-016-2179-2",
                    url="https://doi.org/10.1007/s13369-016-2179-2"
                    }

    * URL: https://drive.google.com/open?id=1VYA2Y_lKXPlMIYNEYO9p2ylymTVcnNR_
A_SUWO
------


API
^^^

.. autoclass:: smote_variants.A_SUWO
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.A_SUWO()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/A_SUWO.png


References:
    * BibTex::
        
        @article{a_suwo,
                    title = "Adaptive semi-unsupervised weighted oversampling (A-SUWO) for imbalanced datasets",
                    journal = "Expert Systems with Applications",
                    volume = "46",
                    pages = "405 - 416",
                    year = "2016",
                    issn = "0957-4174",
                    doi = "https://doi.org/10.1016/j.eswa.2015.10.031",
                    url = "http://www.sciencedirect.com/science/article/pii/S0957417415007356",
                    author = "Iman Nekooeimehr and Susana K. Lai-Yuen",
                    keywords = "Imbalanced dataset, Classification, Clustering, Oversampling"
                    }

    * URL: https://drive.google.com/open?id=14ePxLnx4LlPITR4K_Sjm2PWW41kdczMy

Notes:
    * Equation (7) misses a division by R_j.
    * It is not specified how to sample from clusters with 1 instances.
SMOTE_FRST_2T
-------------


API
^^^

.. autoclass:: smote_variants.SMOTE_FRST_2T
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SMOTE_FRST_2T()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SMOTE_FRST_2T.png


References:
    * BibTex::
        
        @article{smote_frst_2t,
                    title = "Fuzzy-rough imbalanced learning for the diagnosis of High Voltage Circuit Breaker maintenance: The SMOTE-FRST-2T algorithm",
                    journal = "Engineering Applications of Artificial Intelligence",
                    volume = "48",
                    pages = "134 - 139",
                    year = "2016",
                    issn = "0952-1976",
                    doi = "https://doi.org/10.1016/j.engappai.2015.10.009",
                    url = "http://www.sciencedirect.com/science/article/pii/S0952197615002389",
                    author = "E. Ramentol and I. Gondres and S. Lajes and R. Bello and Y. Caballero and C. Cornelis and F. Herrera",
                    keywords = "High Voltage Circuit Breaker (HVCB), Imbalanced learning, Fuzzy rough set theory, Resampling methods"
                    }

    * URL: https://drive.google.com/open?id=1Zmb2MmKGszJB8Q1k7eTZLNG-8KhF4KTc

Notes:
    * Unlucky setting of parameters might result 0 points added, we have fixed this by increasing the gamma_S threshold if the number of samples accepted is low.
    * Similarly, unlucky setting of parameters might result all majority samples turned into minority.
    * In my opinion, in the algorithm presented in the paper the relations are incorrect. The authors talk about accepting samples having POS score below a threshold, and in the algorithm in both places POS >= gamma is used.
AND_SMOTE
---------


API
^^^

.. autoclass:: smote_variants.AND_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.AND_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/AND_SMOTE.png


References:
    * BibTex::
        
        @inproceedings{and_smote,
                         author = {Yun, Jaesub and Ha, Jihyun and Lee, Jong-Seok},
                         title = {Automatic Determination of Neighborhood Size in SMOTE},
                         booktitle = {Proceedings of the 10th International Conference on Ubiquitous Information Management and Communication},
                         series = {IMCOM '16},
                         year = {2016},
                         isbn = {978-1-4503-4142-4},
                         location = {Danang, Viet Nam},
                         pages = {100:1--100:8},
                         articleno = {100},
                         numpages = {8},
                         url = {http://doi.acm.org/10.1145/2857546.2857648},
                         doi = {10.1145/2857546.2857648},
                         acmid = {2857648},
                         publisher = {ACM},
                         address = {New York, NY, USA},
                         keywords = {SMOTE, imbalanced learning, synthetic data generation},
                        } 

    * URL: https://drive.google.com/open?id=1bwj4hQiFnFgfCPDM2e8_GGloZcUd3vBG
NRAS
----


API
^^^

.. autoclass:: smote_variants.NRAS
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.NRAS()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/NRAS.png


References:
    * BibTex::
        
        @article{nras,
                    title = "Noise Reduction A Priori Synthetic Over-Sampling for class imbalanced data sets",
                    journal = "Information Sciences",
                    volume = "408",
                    pages = "146 - 161",
                    year = "2017",
                    issn = "0020-0255",
                    doi = "https://doi.org/10.1016/j.ins.2017.04.046",
                    url = "http://www.sciencedirect.com/science/article/pii/S0020025517307089",
                    author = "William A. Rivera",
                    keywords = "NRAS, SMOTE, OUPS, Class imbalance, Classification"
                    }

    * URL: https://drive.google.com/open?id=1AZ_jRoplDczplH8g1AM3Zn_jujMxaAxA
AMSCO
-----


API
^^^

.. autoclass:: smote_variants.AMSCO
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.AMSCO()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/AMSCO.png


References:
    * BibTex::
        
        @article{amsco,
                    title = "Adaptive multi-objective swarm fusion for imbalanced data classification",
                    journal = "Information Fusion",
                    volume = "39",
                    pages = "1 - 24",
                    year = "2018",
                    issn = "1566-2535",
                    doi = "https://doi.org/10.1016/j.inffus.2017.03.007",
                    url = "http://www.sciencedirect.com/science/article/pii/S1566253517302087",
                    author = "Jinyan Li and Simon Fong and Raymond K. Wong and Victor W. Chu",
                    keywords = "Swarm fusion, Swarm intelligence algorithm, Multi-objective, Crossover rebalancing, Imbalanced data classification"
                    }

    * URL: https://drive.google.com/open?id=1Y90GGJMZeFjp4I_emwk1Z430kjwnNQnt

Notes:
    * It is not clear how the kappa threshold is used, I do use the RA score to drive all the evolution. Particularly:
        
        "In the last phase of each iteration, the average Kappa value
        in current non-inferior set is compare with the latest threshold
        value, the threshold is then increase further if the average value
        increases, and vice versa. By doing so, the non-inferior region will
        be progressively reduced as the Kappa threshold lifts up."
    
    I don't see why would the Kappa threshold lift up if the kappa thresholds
    are decreased if the average Kappa decreases ("vice versa").

    * Due to the interpretation of kappa threshold and the lack of detailed description of the SIS process, the implementation is not exactly what is described in the paper, but something very similar.
SSO
---


API
^^^

.. autoclass:: smote_variants.SSO
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SSO()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SSO.png


References:
    * BibTex::
        
        @InProceedings{sso,
                        author="Rong, Tongwen
                        and Gong, Huachang
                        and Ng, Wing W. Y.",
                        editor="Wang, Xizhao
                        and Pedrycz, Witold
                        and Chan, Patrick
                        and He, Qiang",
                        title="Stochastic Sensitivity Oversampling Technique for Imbalanced Data",
                        booktitle="Machine Learning and Cybernetics",
                        year="2014",
                        publisher="Springer Berlin Heidelberg",
                        address="Berlin, Heidelberg",
                        pages="161--171",
                        abstract="Data level technique is proved to be effective in imbalance learning. The SMOTE is a famous oversampling technique generating synthetic minority samples by linear interpolation between adjacent minorities. However, it becomes inefficiency for datasets with sparse distributions. In this paper, we propose the Stochastic Sensitivity Oversampling (SSO) which generates synthetic samples following Gaussian distributions in the Q-union of minority samples. The Q-union is the union of Q-neighborhoods (hypercubes centered at minority samples) and such that new samples are synthesized around minority samples. Experimental results show that the proposed algorithm performs well on most of datasets, especially those with a sparse distribution.",
                        isbn="978-3-662-45652-1"
                        }

    * URL: https://drive.google.com/open?id=1iW1g0gefhC5bjpXvd9l63N85JgSWTyAc

Notes:
    * In the algorithm step 2d adds a constant to a vector. I have changed it to a componentwise adjustment, and also used the normalized STSM as I don't see any reason why it would be some reasonable, bounded value.
NDO_sampling
------------


API
^^^

.. autoclass:: smote_variants.NDO_sampling
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.NDO_sampling()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/NDO_sampling.png


References:
    * BibTex::
        
        @INPROCEEDINGS{ndo_sampling, 
                        author={L. Zhang and W. Wang}, 
                        booktitle={2011 International Conference of Information Technology, Computer Engineering and Management Sciences}, 
                        title={A Re-sampling Method for Class Imbalance Learning with Credit Data}, 
                        year={2011}, 
                        volume={1}, 
                        number={}, 
                        pages={393-397}, 
                        keywords={data handling;sampling methods;resampling method;class imbalance learning;credit rating;imbalance problem;synthetic minority over-sampling technique;sample distribution;synthetic samples;credit data set;Training;Measurement;Support vector machines;Logistics;Testing;Noise;Classification algorithms;class imbalance;credit rating;SMOTE;sample distribution}, 
                        doi={10.1109/ICM.2011.34}, 
                        ISSN={}, 
                        month={Sept}}

    * URL: https://drive.google.com/open?id=1vrCst6Jk97kTiu-2aJZt3oN5uGHRQA6Q
DSRBF
-----


API
^^^

.. autoclass:: smote_variants.DSRBF
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.DSRBF()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/DSRBF.png


References:
    * BibTex::
        
        @article{dsrbf,
                    title = "A dynamic over-sampling procedure based on sensitivity for multi-class problems",
                    journal = "Pattern Recognition",
                    volume = "44",
                    number = "8",
                    pages = "1821 - 1833",
                    year = "2011",
                    issn = "0031-3203",
                    doi = "https://doi.org/10.1016/j.patcog.2011.02.019",
                    url = "http://www.sciencedirect.com/science/article/pii/S0031320311000823",
                    author = "Francisco Fernández-Navarro and César Hervás-Martínez and Pedro Antonio Gutiérrez",
                    keywords = "Classification, Multi-class, Sensitivity, Accuracy, Memetic algorithm, Imbalanced datasets, Over-sampling method, SMOTE"
                    }

    * URL: https://drive.google.com/open?id=1bUOgi2rFcv55ujfRWuHm_9nHzdg3Uilh

Notes:
    * It is not entirely clear why J-1 output is supposed where J is the number of classes.
    * The fitness function is changed to a balanced mean loss, as I found that it just ignores classification on minority samples (class label +1) in the binary case.
    * The iRprop+ optimization is not implemented.
    * The original paper proposes using SMOTE incrementally. Instead of that, this implementation applies SMOTE to generate all samples needed in the sampling epochs and the evolution of RBF networks is used to select the sampling providing the best results.
Gaussian_SMOTE
--------------


API
^^^

.. autoclass:: smote_variants.Gaussian_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.Gaussian_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/Gaussian_SMOTE.png


References:
    * BibTex::
        
        @article{gaussian_smote,
                  title={Gaussian-Based SMOTE Algorithm for Solving Skewed Class Distributions},
                  author={Hansoo Lee and Jonggeun Kim and Sungshin Kim},
                  journal={Int. J. Fuzzy Logic and Intelligent Systems},
                  year={2017},
                  volume={17},
                  pages={229-234}
                }

    * URL: https://drive.google.com/open?id=12oKlw_GRqsT5-Z4WmvJErBD-vcz5ekwN
kmeans_SMOTE
------------


API
^^^

.. autoclass:: smote_variants.kmeans_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.kmeans_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/kmeans_SMOTE.png


References:
    * BibTex::
        
        @article{kmeans_smote,
                    title = "Improving imbalanced learning through a heuristic oversampling method based on k-means and SMOTE",
                    journal = "Information Sciences",
                    volume = "465",
                    pages = "1 - 20",
                    year = "2018",
                    issn = "0020-0255",
                    doi = "https://doi.org/10.1016/j.ins.2018.06.056",
                    url = "http://www.sciencedirect.com/science/article/pii/S0020025518304997",
                    author = "Georgios Douzas and Fernando Bacao and Felix Last",
                    keywords = "Class-imbalanced learning, Oversampling, Classification, Clustering, Supervised learning, Within-class imbalance"
                    }

    * URL: https://drive.google.com/open?id=1cFpaCsWBXTRYCTIS0hTSOMp_xOwGAPNK
Supervised_SMOTE
----------------


API
^^^

.. autoclass:: smote_variants.Supervised_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.Supervised_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/Supervised_SMOTE.png


References:
    * BibTex::
        
        @article{supervised_smote,
                    author = {Hu, Jun AND He, Xue AND Yu, Dong-Jun AND Yang, Xi-Bei AND Yang, Jing-Yu AND Shen, Hong-Bin},
                    journal = {PLOS ONE},
                    publisher = {Public Library of Science},
                    title = {A New Supervised Over-Sampling Algorithm with Application to Protein-Nucleotide Binding Residue Prediction},
                    year = {2014},
                    month = {09},
                    volume = {9},
                    url = {https://doi.org/10.1371/journal.pone.0107676},
                    pages = {1-10},
                    abstract = {Protein-nucleotide interactions are ubiquitous in a wide variety of biological processes. Accurately identifying interaction residues solely from protein sequences is useful for both protein function annotation and drug design, especially in the post-genomic era, as large volumes of protein data have not been functionally annotated. Protein-nucleotide binding residue prediction is a typical imbalanced learning problem, where binding residues are extremely fewer in number than non-binding residues. Alleviating the severity of class imbalance has been demonstrated to be a promising means of improving the prediction performance of a machine-learning-based predictor for class imbalance problems. However, little attention has been paid to the negative impact of class imbalance on protein-nucleotide binding residue prediction. In this study, we propose a new supervised over-sampling algorithm that synthesizes additional minority class samples to address class imbalance. The experimental results from protein-nucleotide interaction datasets demonstrate that the proposed supervised over-sampling algorithm can relieve the severity of class imbalance and help to improve prediction performance. Based on the proposed over-sampling algorithm, a predictor, called TargetSOS, is implemented for protein-nucleotide binding residue prediction. Cross-validation tests and independent validation tests demonstrate the effectiveness of TargetSOS. The web-server and datasets used in this study are freely available at http://www.csbio.sjtu.edu.cn/bioinf/TargetSOS/.},
                    number = {9},
                    doi = {10.1371/journal.pone.0107676}
                }

    * URL: https://drive.google.com/open?id=1QwAVP9VUBprGFPtrqQra7y-xEBYvqO7Z
SN_SMOTE
--------


API
^^^

.. autoclass:: smote_variants.SN_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.SN_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/SN_SMOTE.png


References:
    * BibTex::
        
        @Article{sn_smote,
                    author="Garc{'i}a, V.
                    and S{'a}nchez, J. S.
                    and Mart{'i}n-F{'e}lez, R.
                    and Mollineda, R. A.",
                    title="Surrounding neighborhood-based SMOTE for learning from imbalanced data sets",
                    journal="Progress in Artificial Intelligence",
                    year="2012",
                    month="Dec",
                    day="01",
                    volume="1",
                    number="4",
                    pages="347--362",
                    abstract="Many traditional approaches to pattern classification assume that the problem classes share similar prior probabilities. However, in many real-life applications, this assumption is grossly violated. Often, the ratios of prior probabilities between classes are extremely skewed. This situation is known as the class imbalance problem. One of the strategies to tackle this problem consists of balancing the classes by resampling the original data set. The SMOTE algorithm is probably the most popular technique to increase the size of the minority class by generating synthetic instances. From the idea of the original SMOTE, we here propose the use of three approaches to surrounding neighborhood with the aim of generating artificial minority instances, but taking into account both the proximity and the spatial distribution of the examples. Experiments over a large collection of databases and using three different classifiers demonstrate that the new surrounding neighborhood-based SMOTE procedures significantly outperform other existing over-sampling algorithms.",
                    issn="2192-6360",
                    doi="10.1007/s13748-012-0027-5",
                    url="https://doi.org/10.1007/s13748-012-0027-5"
                    }

    * URL: https://drive.google.com/open?id=1-cXaoG2z2hoBlI8--Gfe2bOB9lCOIURH
CCR
---


API
^^^

.. autoclass:: smote_variants.CCR
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.CCR()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/CCR.png


References:
    * BibTex::
        
        @article{ccr,
                author = {Koziarski, Michał and Wozniak, Michal},
                year = {2017},
                month = {12},
                pages = {727–736},
                title = {CCR: A combined cleaning and resampling algorithm for imbalanced data classification},
                volume = {27},
                journal = {International Journal of Applied Mathematics and Computer Science}
                }

    * URL: https://drive.google.com/open?id=1-hkZ_pnfHvq4lHwHzC-UxDXae2SzfiuY

Notes:
    * Adapted from https://github.com/michalkoziarski/CCR
ANS
---


API
^^^

.. autoclass:: smote_variants.ANS
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.ANS()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/ANS.png


References:
    * BibTex::
        
        @article{ans,
                 author = {Siriseriwan, W and Sinapiromsaran, Krung},
                 year = {2017},
                 month = {09},
                 pages = {565-576},
                 title = {Adaptive neighbor synthetic minority oversampling technique under 1NN outcast handling},
                 volume = {39},
                 booktitle = {Songklanakarin Journal of Science and Technology}
                 }

    * URL: https://drive.google.com/open?id=1Oz2IloYViHhIbuEBV2GAwaPNB5pgoeNs

Notes:
    * The method is not prepared for the case when there is no c satisfying the condition in line 25 of the algorithm, fixed.
    * The method is not prepared for empty Pused sets, fixed.
cluster_SMOTE
-------------


API
^^^

.. autoclass:: smote_variants.cluster_SMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.cluster_SMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/cluster_SMOTE.png


References:
    * BibTex::
        
        @INPROCEEDINGS{cluster_SMOTE, 
                        author={D. A. Cieslak and N. V. Chawla and A. Striegel}, 
                        booktitle={2006 IEEE International Conference on Granular Computing}, 
                        title={Combating imbalance in network intrusion datasets}, 
                        year={2006}, 
                        volume={}, 
                        number={}, 
                        pages={732-737}, 
                        keywords={Intelligent networks;Intrusion detection;Telecommunication traffic;Data mining;Computer networks;Data security;Machine learning;Counting circuits;Computer security;Humans}, 
                        doi={10.1109/GRC.2006.1635905}, 
                        ISSN={}, 
                        month={May}}

    * URL: https://drive.google.com/open?id=1kDF-WdyMn13h9GNd55b2DLmXt_qtzgBM
NoSMOTE
-------


API
^^^

.. autoclass:: smote_variants.NoSMOTE
    :members:

    .. automethod:: __init__

Example
^^^^^^^

    >>> oversampler= smote_variants.NoSMOTE()
    >>> X_samp, y_samp= oversampler.sample(X, y)


.. image:: figures/base.png
.. image:: figures/NoSMOTE.png


The goal of this class is to provide a functionality to send data through
on any model selection/evaluation pipeline with no oversampling carried
out. It can be used to get baseline estimates on preformance.

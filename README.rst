.. -*- mode: rst -*-

|TravisCI|_ |CircleCI|_ |Codecov|_ |ReadTheDocs|_ |PythonVersion|_ |PyPi|_ |Gitter|_

.. |TravisCI| image:: https://travis-ci.org/gykovacs/smote_variants.svg?branch=master
.. _TravisCI: https://travis-ci.org/gykovacs/smote_variants

.. |CircleCI| image:: https://circleci.com/gh/gykovacs/smote_variants.svg?style=svg
.. _CircleCI: https://circleci.com/gh/gykovacs/smote_variants

.. |Codecov| image:: https://codecov.io/gh/gykovacs/smote_variants/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/gykovacs/smote_variants

.. |ReadTheDocs| image:: https://readthedocs.org/projects/smote-variants/badge/?version=latest
.. _ReadTheDocs: https://smote-variants.readthedocs.io/en/latest/?badge=latest

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-green.svg
.. _PythonVersion: https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-green.svg

.. |PyPi| image:: https://badge.fury.io/py/smote-variants.svg
.. _PyPi: https://badge.fury.io/py/smote-variants

.. |Gitter| image:: https://badges.gitter.im/smote_variants.svg
.. _Gitter: https://gitter.im/smote_variants?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge


SMOTE-variants
==============

Introduction
------------

The package implements 85 variants of the Synthetic Minority Oversampling Technique (SMOTE).
Besides the implementations, an easy to use model selection framework is supplied to enable
the rapid evaluation of oversampling techniques on unseen datasets.

The implemented techniques: [SMOTE]_ , [SMOTE_TomekLinks]_ , [SMOTE_ENN]_ , [Borderline_SMOTE1]_ , [Borderline_SMOTE2]_ , [ADASYN]_ , [AHC]_ , [LLE_SMOTE]_ , [distance_SMOTE]_ , [SMMO]_ , [polynom_fit_SMOTE]_ , [Stefanowski]_ , [ADOMS]_ , [Safe_Level_SMOTE]_ , [MSMOTE]_ , [DE_oversampling]_ , [SMOBD]_ , [SUNDO]_ , [MSYN]_ , [SVM_balance]_ , [TRIM_SMOTE]_ , [SMOTE_RSB]_ , [ProWSyn]_ , [SL_graph_SMOTE]_ , [NRSBoundary_SMOTE]_ , [LVQ_SMOTE]_ , [SOI_CJ]_ , [ROSE]_ , [SMOTE_OUT]_ , [SMOTE_Cosine]_ , [Selected_SMOTE]_ , [LN_SMOTE]_ , [MWMOTE]_ , [PDFOS]_ , [IPADE_ID]_ , [RWO_sampling]_ , [NEATER]_ , [DEAGO]_ , [Gazzah]_ , [MCT]_ , [ADG]_ , [SMOTE_IPF]_ , [KernelADASYN]_ , [MOT2LD]_ , [V_SYNTH]_ , [OUPS]_ , [SMOTE_D]_ , [SMOTE_PSO]_ , [CURE_SMOTE]_ , [SOMO]_ , [ISOMAP_Hybrid]_ , [CE_SMOTE]_ , [Edge_Det_SMOTE]_ , [CBSO]_ , [E_SMOTE]_ , [DBSMOTE]_ , [ASMOBD]_ , [Assembled_SMOTE]_ , [SDSMOTE]_ , [DSMOTE]_ , [G_SMOTE]_ , [NT_SMOTE]_ , [Lee]_ , [SPY]_ , [SMOTE_PSOBAT]_ , [MDO]_ , [Random_SMOTE]_ , [ISMOTE]_ , [VIS_RST]_ , [GASMOTE]_ , [A_SUWO]_ , [SMOTE_FRST_2T]_ , [AND_SMOTE]_ , [NRAS]_ , [AMSCO]_ , [SSO]_ , [NDO_sampling]_ , [DSRBF]_ , [Gaussian_SMOTE]_ , [kmeans_SMOTE]_ , [Supervised_SMOTE]_ , [SN_SMOTE]_ , [CCR]_ , [ANS]_ , [cluster_SMOTE]_ 

Citation
--------

The publication of this work and its derivatives is going on, please come back in a couple of days or weeks for updates.

Documentation
-------------

* For a detailed documentation see http://smote-variants.readthedocs.io.
* For a YouTube tutorial check https://www.youtube.com/watch?v=GSK7akQPM60

The competition
---------------

We have kicked off a competition to find the best general purpose oversampling technique. The competition is ongoing, the preliminary results are available at the page https://smote-variants.readthedocs.io/en/latest/competition.html

All the numerical results are reproducible by the 005_evaluation example script, downloading the database foldings from the link below and following the instructions in the script. Anyone is open to join the competition by implementing an oversampling technique as part of the smote_variants package. The below database foldings can be used to evaluate the technique, and compare the results to the already implemented ones. Once the code is added to a feature branch, the evaluation will be repeated by the organizers and the results added to the rankings page.

* Database foldings: `https://drive.google.com/open?id=1PKw1vETVUzaToomio1-RGzJ9_-buYjOW <https://drive.google.com/open?id=1PKw1vETVUzaToomio1-RGzJ9_-buYjOW>`__

Other downloads
---------------

If someone is interested in the results of the evaluation of 85 oversamplers on 104 imbalanced datasets, the raw and aggregated results as structured pickle files are avaialble at the below links:

* Raw results: `https://drive.google.com/open?id=12CfB3184nchLIwStaHhrjcQK7Ari18Mo <https://drive.google.com/open?id=12CfB3184nchLIwStaHhrjcQK7Ari18Mo>`__
* Aggregated results: `https://drive.google.com/open?id=19JGikRYXQ6-eOxaFVrqkF64zOCiSdT-j <https://drive.google.com/open?id=19JGikRYXQ6-eOxaFVrqkF64zOCiSdT-j>`__

References
----------

.. [SMOTE] Chawla, N. V. and Bowyer, K. W. and Hall, L. O. and Kegelmeyer, W. P., "{SMOTE}: synthetic minority over-sampling technique" , Journal of Artificial Intelligence Research, 2002, pp. 321--357

.. [SMOTE_TomekLinks] Batista, Gustavo E. A. P. A. and Prati, Ronaldo C. and Monard, Maria Carolina, "A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data" , SIGKDD Explor. Newsl., 2004, pp. 20--29

.. [SMOTE_ENN] Batista, Gustavo E. A. P. A. and Prati, Ronaldo C. and Monard, Maria Carolina, "A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data" , SIGKDD Explor. Newsl., 2004, pp. 20--29

.. [Borderline_SMOTE1] Ha, "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning" , Advances in Intelligent Computing, 2005, pp. 878--887

.. [Borderline_SMOTE2] Ha, "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning" , Advances in Intelligent Computing, 2005, pp. 878--887

.. [ADASYN] He, H. and Bai, Y. and Garcia, E. A. and Li, S., "{ADASYN}: adaptive synthetic sampling approach for imbalanced learning" , Proceedings of IJCNN, 2008, pp. 1322--1328

.. [AHC] Gilles Cohen and Mélanie Hilario and Hugo Sax and Stéphane Hugonnet and Antoine Geissbuhler, "Learning from imbalanced data in surveillance of nosocomial infection" , Artificial Intelligence in Medicine, 2006, pp. 7 - 18

.. [LLE_SMOTE] Wang, J. and Xu, M. and Wang, H. and Zhang, J., "Classification of Imbalanced Data by Using the SMOTE Algorithm and Locally Linear Embedding" , 2006 8th international Conference on Signal Processing, 2006, pp. 

.. [distance_SMOTE] de la Calleja, J. and Fuentes, O., "A distance-based over-sampling method for learning from imbalanced data sets" , Proceedings of the Twentieth International Florida Artificial Intelligence, 2007, pp. 634--635

.. [SMMO] de la Calleja, Jorge and Fuentes, Olac and González, Jesús, "Selecting Minority Examples from Misclassified Data for Over-Sampling." , Proceedings of the Twenty-First International Florida Artificial Intelligence Research Society Conference, 2008, pp. 276-281

.. [polynom_fit_SMOTE] Gazzah, S. and Amara, N. E. B., "New Oversampling Approaches Based on Polynomial Fitting for Imbalanced Data Sets" , 2008 The Eighth IAPR International Workshop on Document Analysis Systems, 2008, pp. 677-684

.. [Stefanowski] Stefanowski, Jerzy and Wilk, Szymon, "Selective Pre-processing of Imbalanced Data for Improving Classification Performance" , Proceedings of the 10th International Conference on Data Warehousing and Knowledge Discovery, 2008, pp. 283--292

.. [ADOMS] Tang, S. and Chen, S., "The generation mechanism of synthetic minority class examples" , 2008 International Conference on Information Technology and Applications in Biomedicine, 2008, pp. 444-447

.. [Safe_Level_SMOTE] Bunkhumpornpat, Chumphol and Sinapiromsaran, Krung and Lursinsap, Chidchanok, "Safe-Level-SMOTE: Safe-Level-Synthetic Minority Over-Sampling TEchnique for Handling the Class Imbalanced Problem" , Proceedings of the 13th Pacific-Asia Conference on Advances in Knowledge Discovery and Data Mining, 2009, pp. 475--482

.. [MSMOTE] Hu, Shengguo and Liang, Yanfeng and Ma, Lintao and He, Ying, "MSMOTE: Improving Classification Performance When Training Data is Imbalanced" , Proceedings of the 2009 Second International Workshop on Computer Science and Engineering - Volume 02, 2009, pp. 13--17

.. [DE_oversampling] Chen, L. and Cai, Z. and Chen, L. and Gu, Q., "A Novel Differential Evolution-Clustering Hybrid Resampling Algorithm on Imbalanced Datasets" , 2010 Third International Conference on Knowledge Discovery and Data Mining, 2010, pp. 81-85

.. [SMOBD] Cao, Q. and Wang, S., "Applying Over-sampling Technique Based on Data Density and Cost-sensitive SVM to Imbalanced Learning" , 2011 International Conference on Information Management, Innovation Management and Industrial Engineering, 2011, pp. 543-548

.. [SUNDO] Cateni, S. and Colla, V. and Vannucci, M., "Novel resampling method for the classification of imbalanced datasets for industrial and other real-world problems" , 2011 11th International Conference on Intelligent Systems Design and Applications, 2011, pp. 402-407

.. [MSYN] Fa, "Margin-Based Over-Sampling Method for Learning from Imbalanced Datasets" , Advances in Knowledge Discovery and Data Mining, 2011, pp. 309--320

.. [SVM_balance] Farquad, M.A.H. and Bose, Indranil, "Preprocessing Unbalanced Data Using Support Vector Machine" , Decis. Support Syst., 2012, pp. 226--233

.. [TRIM_SMOTE] Puntumapo, "A Pruning-Based Approach for Searching Precise and Generalized Region for Synthetic Minority Over-Sampling" , Advances in Knowledge Discovery and Data Mining, 2012, pp. 371--382

.. [SMOTE_RSB] Ramento, "SMOTE-RSB*: a hybrid preprocessing approach based on oversampling and undersampling for high imbalanced data-sets using SMOTE and rough sets theory" , Knowledge and Information Systems, 2012, pp. 245--265

.. [ProWSyn] Baru, "ProWSyn: Proximity Weighted Synthetic Oversampling Technique for Imbalanced Data Set Learning" , Advances in Knowledge Discovery and Data Mining, 2013, pp. 317--328

.. [SL_graph_SMOTE] Bunkhumpornpat, Chumpol and Subpaiboonkit, Sitthichoke, "Safe level graph for synthetic minority over-sampling techniques" , 13th International Symposium on Communications and Information Technologies, 2013, pp. 570-575

.. [NRSBoundary_SMOTE] Feng, Hu and Hang, Li, "A Novel Boundary Oversampling Algorithm Based on Neighborhood Rough Set Model: NRSBoundary-SMOTE" , Mathematical Problems in Engineering, 2013, pp. 10

.. [LVQ_SMOTE] Munehiro Nakamura and Yusuke Kajiwara and Atsushi Otsuka and Haruhiko Kimura, "LVQ-SMOTE – Learning Vector Quantization based Synthetic Minority Over–sampling Technique for biomedical data" , BioData Mining, 2013

.. [SOI_CJ] Sánchez, Atlántida I. and Morales, Eduardo and Gonzalez, Jesus, "Synthetic Oversampling of Instances Using Clustering" , International Journal of Artificial Intelligence Tools, 2013, pp. 

.. [ROSE] Menard, "Training and assessing classification rules with imbalanced data" , Data Mining and Knowledge Discovery, 2014, pp. 92--122

.. [SMOTE_OUT] Fajri Koto, "SMOTE-Out, SMOTE-Cosine, and Selected-SMOTE: An enhancement strategy to handle imbalance in data level" , 2014 International Conference on Advanced Computer Science and Information System, 2014, pp. 280-284

.. [SMOTE_Cosine] Fajri Koto, "SMOTE-Out, SMOTE-Cosine, and Selected-SMOTE: An enhancement strategy to handle imbalance in data level" , 2014 International Conference on Advanced Computer Science and Information System, 2014, pp. 280-284

.. [Selected_SMOTE] Fajri Koto, "SMOTE-Out, SMOTE-Cosine, and Selected-SMOTE: An enhancement strategy to handle imbalance in data level" , 2014 International Conference on Advanced Computer Science and Information System, 2014, pp. 280-284

.. [LN_SMOTE] Maciejewski, T. and Stefanowski, J., "Local neighbourhood extension of SMOTE for mining imbalanced data" , 2011 IEEE Symposium on Computational Intelligence and Data Mining (CIDM), 2011, pp. 104-111

.. [MWMOTE] Barua, S. and Islam, M. M. and Yao, X. and Murase, K., "MWMOTE--Majority Weighted Minority Oversampling Technique for Imbalanced Data Set Learning" , IEEE Transactions on Knowledge and Data Engineering, 2014, pp. 405-425

.. [PDFOS] Ming Gao and Xia Hong and Sheng Chen and Chris J. Harris and Emad Khalaf, "PDFOS: PDF estimation based over-sampling for imbalanced two-class problems" , Neurocomputing, 2014, pp. 248 - 259

.. [IPADE_ID] Victoria López and Isaac Triguero and Cristóbal J. Carmona and Salvador García and Francisco Herrera, "Addressing imbalanced classification with instance generation techniques: IPADE-ID" , Neurocomputing, 2014, pp. 15 - 28

.. [RWO_sampling] Zhang, Huaxzhang and Li, Mingfang, "RWO-Sampling: A Random Walk Over-Sampling Approach to Imbalanced Data Classification" , Information Fusion, 2014, pp. 

.. [NEATER] Almogahed, B. A. and Kakadiaris, I. A., "NEATER: Filtering of Over-sampled Data Using Non-cooperative Game Theory" , 2014 22nd International Conference on Pattern Recognition, 2014, pp. 1371-1376

.. [DEAGO] Bellinger, C. and Japkowicz, N. and Drummond, C., "Synthetic Oversampling for Advanced Radioactive Threat Detection" , 2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA), 2015, pp. 948-953

.. [Gazzah] Gazzah, S. and Hechkel, A. and Essoukri Ben Amara, N. , "A hybrid sampling method for imbalanced data" , 2015 IEEE 12th International Multi-Conference on Systems, Signals Devices (SSD15), 2015, pp. 1-6

.. [MCT] Jiang, Liangxiao and Qiu, Chen and Li, Chaoqun, "A Novel Minority Cloning Technique for Cost-Sensitive Learning" , International Journal of Pattern Recognition and Artificial Intelligence, 2015, pp. 1551004

.. [ADG] Pourhabib, A. and Mallick, Bani K. and Ding, Yu, "A Novel Minority Cloning Technique for Cost-Sensitive Learning" , Journal of Machine Learning Research, 2015, pp. 2695--2724

.. [SMOTE_IPF] José A. Sáez and Julián Luengo and Jerzy Stefanowski and Francisco Herrera, "SMOTE–IPF: Addressing the noisy and borderline examples problem in imbalanced classification by a re-sampling method with filtering" , Information Sciences, 2015, pp. 184 - 203

.. [KernelADASYN] Tang, B. and He, H., "KernelADASYN: Kernel based adaptive synthetic data generation for imbalanced learning" , 2015 IEEE Congress on Evolutionary Computation (CEC), 2015, pp. 664-671

.. [MOT2LD] Xi, "A Synthetic Minority Oversampling Method Based on Local Densities in Low-Dimensional Space for Imbalanced Learning" , Database Systems for Advanced Applications, 2015, pp. 3--18

.. [V_SYNTH] Young,Ii, William A. and Nykl, Scott L. and Weckman, Gary R. and Chelberg, David M., "Using Voronoi Diagrams to Improve Classification Performances when Modeling Imbalanced Datasets" , Neural Comput. Appl., 2015, pp. 1041--1054

.. [OUPS] William A. Rivera and Petros Xanthopoulos, "A priori synthetic over-sampling methods for increasing classification sensitivity in imbalanced data sets" , Expert Systems with Applications, 2016, pp. 124 - 135

.. [SMOTE_D] Torre, "SMOTE-D a Deterministic Version of SMOTE" , Pattern Recognition, 2016, pp. 177--188

.. [SMOTE_PSO] Jair Cervantes and Farid Garcia-Lamont and Lisbeth Rodriguez and Asdrúbal López and José Ruiz Castilla and Adrian Trueba, "PSO-based method for SVM classification on skewed data sets" , Neurocomputing, 2017, pp. 187 - 197

.. [CURE_SMOTE] M, "CURE-SMOTE algorithm and hybrid algorithm for feature selection and parameter optimization based on random forests" , BMC Bioinformatics, 2017, pp. 169

.. [SOMO] Georgios Douzas and Fernando Bacao, "Self-Organizing Map Oversampling (SOMO) for imbalanced data set learning" , Expert Systems with Applications, 2017, pp. 40 - 52

.. [ISOMAP_Hybrid] Gu, Qiong and Cai, Zhihua and Zhu, Li, "Classification of Imbalanced Data Sets by Using the Hybrid Re-sampling Algorithm Based on Isomap" , Proceedings of the 4th International Symposium on Advances in Computation and Intelligence, 2009, pp. 287--296

.. [CE_SMOTE] Chen, S. and Guo, G. and Chen, L., "A New Over-Sampling Method Based on Cluster Ensembles" , 2010 IEEE 24th International Conference on Advanced Information Networking and Applications Workshops, 2010, pp. 599-604

.. [Edge_Det_SMOTE] Kang, Y. and Won, S., "Weight decision algorithm for oversampling technique on class-imbalanced learning" , ICCAS 2010, 2010, pp. 182-186

.. [CBSO] Baru, "A Novel Synthetic Minority Oversampling Technique for Imbalanced Data Set Learning" , Neural Information Processing, 2011, pp. 735--744

.. [E_SMOTE] Deepa, T. and Punithavalli, M., "An E-SMOTE technique for feature selection in High-Dimensional Imbalanced Dataset" , 2011 3rd International Conference on Electronics Computer Technology, 2011, pp. 322-324

.. [DBSMOTE] Bunkhumpornpa, "DBSMOTE: Density-Based Synthetic Minority Over-sampling TEchnique" , Applied Intelligence, 2012, pp. 664--684

.. [ASMOBD] Senzhang Wang and Zhoujun Li and Wenhan Chao and Qinghua Cao, "Applying adaptive over-sampling technique based on data density and cost-sensitive SVM to imbalanced learning" , The 2012 International Joint Conference on Neural Networks (IJCNN), 2012, pp. 1-8

.. [Assembled_SMOTE] Zhou, B. and Yang, C. and Guo, H. and Hu, J., "A quasi-linear SVM combined with assembled SMOTE for imbalanced data classification" , The 2013 International Joint Conference on Neural Networks (IJCNN), 2013, pp. 1-7

.. [SDSMOTE] Li, K. and Zhang, W. and Lu, Q. and Fang, X., "An Improved SMOTE Imbalanced Data Classification Method Based on Support Degree" , 2014 International Conference on Identification, Information and Knowledge in the Internet of Things, 2014, pp. 34-38

.. [DSMOTE] Mahmoudi, S. and Moradi, P. and Akhlaghian, F. and Moradi, R., "Diversity and separable metrics in over-sampling technique for imbalanced data classification" , 2014 4th International Conference on Computer and Knowledge Engineering (ICCKE), 2014, pp. 152-158

.. [G_SMOTE] Sandhan, T. and Choi, J. Y., "Handling Imbalanced Datasets by Partially Guided Hybrid Sampling for Pattern Recognition" , 2014 22nd International Conference on Pattern Recognition, 2014, pp. 1449-1453

.. [NT_SMOTE] Xu, Y. H. and Li, H. and Le, L. P. and Tian, X. Y., "Neighborhood Triangular Synthetic Minority Over-sampling Technique for Imbalanced Prediction on Small Samples of Chinese Tourism and Hospitality Firms" , 2014 Seventh International Joint Conference on Computational Sciences and Optimization, 2014, pp. 534-538

.. [Lee] Lee, Jaedong and Kim, Noo-ri and Lee, Jee-Hyong, "An Over-sampling Technique with Rejection for Imbalanced Class Learning" , Proceedings of the 9th International Conference on Ubiquitous Information Management and Communication, 2015, pp. 102:1--102:6

.. [SPY] Dang, X. T. and Tran, D. H. and Hirose, O. and Satou, K., "SPY: A Novel Resampling Method for Improving Classification Performance in Imbalanced Data" , 2015 Seventh International Conference on Knowledge and Systems Engineering (KSE), 2015, pp. 280-285

.. [SMOTE_PSOBAT] Li, J. and Fong, S. and Zhuang, Y., "Optimizing SMOTE by Metaheuristics with Neural Network and Decision Tree" , 2015 3rd International Symposium on Computational and Business Intelligence (ISCBI), 2015, pp. 26-32

.. [MDO] Abdi, L. and Hashemi, S., "To Combat Multi-Class Imbalanced Problems by Means of Over-Sampling Techniques" , IEEE Transactions on Knowledge and Data Engineering, 2016, pp. 238-251

.. [Random_SMOTE] Don, "A New Over-Sampling Approach: Random-SMOTE for Learning from Imbalanced Data Sets" , Knowledge Scienc, 2011, pp. 343--352

.. [ISMOTE] L, "A New Combination Sampling Method for Imbalanced Data" , Proceedings of 2013 Chinese Intelligent Automation Conference, 2013, pp. 547--554

.. [VIS_RST] Borowsk, "Imbalanced Data Classification: A Novel Re-sampling Approach Combining Versatile Improved SMOTE and Rough Sets" , Computer Information Systems and Industrial Management, 2016, pp. 31--42

.. [GASMOTE] Jian, "A Novel Algorithm for Imbalance Data Classification Based on Genetic Algorithm Improved SMOTE" , Arabian Journal for Science and Engineering, 2016, pp. 3255--3266

.. [A_SUWO] Iman Nekooeimehr and Susana K. Lai-Yuen, "Adaptive semi-unsupervised weighted oversampling (A-SUWO) for imbalanced datasets" , Expert Systems with Applications, 2016, pp. 405 - 416

.. [SMOTE_FRST_2T] Ramento, "Fuzzy-rough imbalanced learning for the diagnosis of High Voltage Circuit Breaker maintenance: The SMOTE-FRST-2T algorithm" , Engineering Applications of Artificial Intelligence, 2016, pp. 134 - 139

.. [AND_SMOTE] Yun, Jaesub and Ha, Jihyun and Lee, Jong-Seok, "Automatic Determination of Neighborhood Size in SMOTE" , Proceedings of the 10th International Conference on Ubiquitous Information Management and Communication, 2016, pp. 100:1--100:8

.. [NRAS] William A. Rivera, "Noise Reduction A Priori Synthetic Over-Sampling for class imbalanced data sets" , Information Sciences, 2017, pp. 146 - 161

.. [AMSCO] Jinyan Li and Simon Fong and Raymond K. Wong and Victor W. Chu, "Adaptive multi-objective swarm fusion for imbalanced data classification" , Information Fusion, 2018, pp. 1 - 24

.. [SSO] Ron, "Stochastic Sensitivity Oversampling Technique for Imbalanced Data" , Machine Learning and Cybernetics, 2014, pp. 161--171

.. [NDO_sampling] Zhang, L. and Wang, W., "A Re-sampling Method for Class Imbalance Learning with Credit Data" , 2011 International Conference of Information Technology, Computer Engineering and Management Sciences, 2011, pp. 393-397

.. [DSRBF] Francisco Fernández-Navarro and César Hervás-Martínez and Pedro Antonio Gutiérrez, "A dynamic over-sampling procedure based on sensitivity for multi-class problems" , Pattern Recognition, 2011, pp. 1821 - 1833

.. [Gaussian_SMOTE] Hansoo Lee and Jonggeun Kim and Sungshin Kim, "Gaussian-Based SMOTE Algorithm for Solving Skewed Class Distributions" , Int. J. Fuzzy Logic and Intelligent Systems, 2017, pp. 229-234

.. [kmeans_SMOTE] Georgios Douzas and Fernando Bacao and Felix Last, "Improving imbalanced learning through a heuristic oversampling method based on k-means and SMOTE" , Information Sciences, 2018, pp. 1 - 20

.. [Supervised_SMOTE] Hu, Jun AND He, Xue AND Yu, Dong-Jun AND Yang, Xi-Bei AND Yang, Jing-Yu AND Shen, Hong-Bin, "A New Supervised Over-Sampling Algorithm with Application to Protein-Nucleotide Binding Residue Prediction" , PLOS ONE, 2014, pp. 1-10

.. [SN_SMOTE] Garc{'i}, "Surrounding neighborhood-based SMOTE for learning from imbalanced data sets" , Progress in Artificial Intelligence, 2012, pp. 347--362

.. [CCR] Koziarski, Michał and Wozniak, Michal, "CCR: A combined cleaning and resampling algorithm for imbalanced data classification" , International Journal of Applied Mathematics and Computer Science, 2017, pp. 727–736

.. [ANS] Siriseriwan, W and Sinapiromsaran, Krung, "Adaptive neighbor synthetic minority oversampling technique under 1NN outcast handling" , Songklanakarin Journal of Science and Technology, 2017, pp. 565-576

.. [cluster_SMOTE] Cieslak, D. A. and Chawla, N. V. and Striegel, A., "Combating imbalance in network intrusion datasets" , 2006 IEEE International Conference on Granular Computing, 2006, pp. 732-737

